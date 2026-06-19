from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom


def read_xml_boxes(xml_dir: str | Path) -> dict[str, list[dict]]:
    boxes_by_uid: dict[str, list[dict]] = {}

    for root_dir, _, files in os.walk(xml_dir):
        for filename in files:
            if not filename.lower().endswith(".xml"):
                continue

            xml_uid = Path(filename).stem
            xml_path = Path(root_dir) / filename

            try:
                root = ET.parse(xml_path).getroot()
            except Exception as exc:
                print(f"XML error: {xml_path} {exc}")
                continue

            boxes = []
            for obj in root.findall("object"):
                label = obj.findtext("name")
                if label is None or label.strip() == "" or label == "Unknow":
                    label = "tumor"

                bndbox = obj.find("bndbox")
                if bndbox is None:
                    continue

                boxes.append(
                    {
                        "label": label,
                        "xmin": int(float(bndbox.findtext("xmin"))),
                        "ymin": int(float(bndbox.findtext("ymin"))),
                        "xmax": int(float(bndbox.findtext("xmax"))),
                        "ymax": int(float(bndbox.findtext("ymax"))),
                        "xml_file": filename,
                        "xml_uid": xml_uid,
                    }
                )

            if boxes:
                boxes_by_uid[xml_uid] = boxes

    return boxes_by_uid


def read_dicom_series(dicom_dir: str | Path):
    dicoms = []
    for root, _, files in os.walk(dicom_dir):
        for filename in files:
            path = Path(root) / filename
            try:
                ds = pydicom.dcmread(path, force=True)
                if hasattr(ds, "PixelData"):
                    dicoms.append((path, ds))
            except Exception:
                pass

    if not dicoms:
        raise RuntimeError(f"No DICOM image found in {dicom_dir}")

    def sort_key(item):
        path, ds = item
        if hasattr(ds, "ImagePositionPatient"):
            return float(ds.ImagePositionPatient[2])
        if hasattr(ds, "InstanceNumber"):
            return int(ds.InstanceNumber)
        return path.name

    dicoms = sorted(dicoms, key=sort_key)
    volume = []
    sop_uids = []
    dicom_names = []

    for path, ds in dicoms:
        img = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        volume.append(img * slope + intercept)
        sop_uids.append(str(getattr(ds, "SOPInstanceUID", "")))
        dicom_names.append(path.name)

    return np.stack(volume, axis=0), dicoms, sop_uids, dicom_names


def make_affine_from_dicom(dicoms) -> np.ndarray:
    first_ds = dicoms[0][1]
    row_spacing, col_spacing = [float(x) for x in first_ds.PixelSpacing]
    orientation = np.array(first_ds.ImageOrientationPatient, dtype=float)
    row_cosines = orientation[:3]
    col_cosines = orientation[3:]
    ipp0 = np.array(first_ds.ImagePositionPatient, dtype=float)

    if len(dicoms) > 1:
        ipp1 = np.array(dicoms[1][1].ImagePositionPatient, dtype=float)
        slice_vector = ipp1 - ipp0
    else:
        thickness = float(getattr(first_ds, "SliceThickness", 1.0))
        slice_vector = np.cross(row_cosines, col_cosines) * thickness

    affine = np.eye(4)
    affine[:3, 0] = col_cosines * col_spacing
    affine[:3, 1] = row_cosines * row_spacing
    affine[:3, 2] = slice_vector
    affine[:3, 3] = ipp0
    return affine


def bbox_from_mask_zyx(mask_zyx: np.ndarray) -> dict | None:
    zz, yy, xx = np.where(mask_zyx > 0)
    if len(xx) == 0:
        return None
    return {
        "xmin": int(xx.min()),
        "xmax": int(xx.max()),
        "ymin": int(yy.min()),
        "ymax": int(yy.max()),
        "zmin": int(zz.min()),
        "zmax": int(zz.max()),
    }


def prepare_patient_from_dicom_xml(
    patient_id: str,
    dicom_dir: str | Path,
    xml_dir: str | Path,
    out_patient_dir: str | Path,
) -> dict:
    out_patient_dir = Path(out_patient_dir)
    images_dir = out_patient_dir / "images"
    ann_dir = out_patient_dir / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    ct_zyx, dicoms, sop_uids, _ = read_dicom_series(dicom_dir)
    affine = make_affine_from_dicom(dicoms)
    boxes_by_uid = read_xml_boxes(xml_dir)

    mask_zyx = np.zeros_like(ct_zyx, dtype=np.uint8)
    matched_slices = []

    for z, uid in enumerate(sop_uids):
        boxes = boxes_by_uid.get(uid, [])
        if not boxes:
            continue

        matched_slices.append(z)
        for box in boxes:
            xmin = max(0, box["xmin"])
            ymin = max(0, box["ymin"])
            xmax = min(mask_zyx.shape[2] - 1, box["xmax"])
            ymax = min(mask_zyx.shape[1] - 1, box["ymax"])
            mask_zyx[z, ymin : ymax + 1, xmin : xmax + 1] = 1

    if not matched_slices:
        raise RuntimeError("No XML annotation matches the DICOM slices.")

    ct_xyz = np.transpose(ct_zyx, (2, 1, 0))
    mask_xyz = np.transpose(mask_zyx, (2, 1, 0))

    ct_path = images_dir / f"{patient_id}_ct_phase_00.nii.gz"
    mask_path = ann_dir / f"{patient_id}_tumor_mask_phase_00.nii.gz"
    bbox_path = ann_dir / f"{patient_id}_bbox_phase_00.json"

    nib.save(nib.Nifti1Image(ct_xyz.astype(np.float32), affine), ct_path)
    nib.save(nib.Nifti1Image(mask_xyz.astype(np.uint8), affine), mask_path)

    bbox = bbox_from_mask_zyx(mask_zyx)
    info = {
        "patient_id": patient_id,
        "phase": "phase_00",
        "source": "original_CT",
        "bbox_format": "voxel_index_ZYX",
        "bbox_3d": bbox,
        "annotated_slices": matched_slices,
        "ct_nifti": str(ct_path),
        "mask_nifti": str(mask_path),
    }
    bbox_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info
