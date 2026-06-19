from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from typing import Callable, Optional

import imageio.v2 as imageio
import nibabel as nib
import numpy as np
from PIL import Image

from .radiotherapy_pipeline.deepdrr_runner import XRAY_VIEWS, run_deepdrr
from .radiotherapy_pipeline.dynagan_runner import run_dynagan_4dct
from .schemas import JobParams
from .settings import settings


LogFn = Callable[[str], None]
ProgressFn = Callable[[str, str, int, Optional[dict]], None]


def safe_patient_id(value: str) -> str:
    clean = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value.strip())
    return clean or "patient"


def rel_url(job_dir: Path, path: Path) -> str:
    return f"/jobs/{job_dir.name}/files/{path.relative_to(job_dir).as_posix()}"


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    lo, hi = np.percentile(finite, [1, 99])
    if hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
    out = np.clip((arr - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return (out * 255).astype(np.uint8)


def resize_to_canvas(
    img: Image.Image,
    size: int,
    row_spacing: float = 1.0,
    col_spacing: float = 1.0,
) -> Image.Image:
    row_spacing = max(float(row_spacing), 1e-6)
    col_spacing = max(float(col_spacing), 1e-6)
    physical_h = max(img.height * row_spacing, 1e-6)
    physical_w = max(img.width * col_spacing, 1e-6)
    scale = min(size / physical_w, size / physical_h)
    new_w = max(1, int(round(physical_w * scale)))
    new_h = max(1, int(round(physical_h * scale)))
    resized = img.resize((new_w, new_h), resample=Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (5, 14, 24))
    canvas.paste(resized.convert("RGB"), ((size - new_w) // 2, (size - new_h) // 2))
    return canvas


def save_preview_png(
    arr: np.ndarray,
    path: Path,
    size: int = 420,
    spacing: tuple[float, float] = (1.0, 1.0),
) -> None:
    img = Image.fromarray(normalize_to_uint8(arr)).convert("L")
    canvas = resize_to_canvas(img, size, row_spacing=spacing[0], col_spacing=spacing[1])
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def write_ct_previews(ct_path: Path, preview_dir: Path) -> dict[str, str]:
    ct_img = nib.load(str(ct_path))
    data = ct_img.get_fdata().astype(np.float32)
    sx, sy, sz = [float(value) for value in ct_img.header.get_zooms()[:3]]
    cx, cy, cz = [axis // 2 for axis in data.shape]
    outputs = {
        "axial": preview_dir / "ct_axial.png",
        "coronal": preview_dir / "ct_coronal.png",
        "sagittal": preview_dir / "ct_sagittal.png",
    }
    save_preview_png(np.flipud(np.rot90(data[:, :, cz])), outputs["axial"], spacing=(sy, sx))
    save_preview_png(np.rot90(data[:, cy, :]), outputs["coronal"], spacing=(sz, sx))
    save_preview_png(np.rot90(data[cx, :, :]), outputs["sagittal"], spacing=(sz, sy))
    return {name: str(path) for name, path in outputs.items()}


def write_4dct_gif(fourd_dir: Path, preview_dir: Path, patient_id: str) -> Path | None:
    ct_files = sorted((fourd_dir / "images").glob(f"{patient_id}_ct_phase_*.nii.gz"))
    if not ct_files:
        return None
    frames = []
    z_index = None
    spacing = (1.0, 1.0)
    for ct_path in ct_files:
        ct_img = nib.load(str(ct_path))
        data = ct_img.get_fdata().astype(np.float32)
        if z_index is None:
            z_index = data.shape[2] // 2
            sx, sy = [float(value) for value in ct_img.header.get_zooms()[:2]]
            spacing = (sy, sx)
        frame = Image.fromarray(
            normalize_to_uint8(np.flipud(np.rot90(data[:, :, z_index])))
        ).convert("L")
        frames.append(np.array(resize_to_canvas(frame, 512, row_spacing=spacing[0], col_spacing=spacing[1])))
    if not frames:
        return None
    gif_path = preview_dir / "4dct_axial_cycle.gif"
    preview_dir.mkdir(parents=True, exist_ok=True)
    cycle = frames + frames[-2:0:-1] if len(frames) > 2 else frames
    imageio.mimsave(gif_path, cycle, duration=220, loop=0)
    return gif_path


def collect_deepdrr_gifs(drr_dir: Path) -> list[dict[str, str]]:
    return [
        {"label": path.stem, "url_path": str(path)}
        for path in sorted((drr_dir / "gifs").glob("*.gif"))
    ]


def prepare_uploaded_nifti(
    patient_id: str,
    ct_path: Path,
    mask_path: Path | None,
    prepared_dir: Path,
    has_annotations: bool,
) -> dict:
    images_dir = prepared_dir / "images"
    annotations_dir = prepared_dir / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    ct_img = nib.load(str(ct_path))
    ct_data = ct_img.get_fdata().astype(np.float32)
    prepared_ct = images_dir / f"{patient_id}_ct_phase_00.nii.gz"
    prepared_mask = annotations_dir / f"{patient_id}_tumor_mask_phase_00.nii.gz"
    nib.save(nib.Nifti1Image(ct_data, ct_img.affine, ct_img.header.copy()), prepared_ct)

    if has_annotations and mask_path is not None:
        mask_img = nib.load(str(mask_path))
        mask_data = (mask_img.get_fdata() > 0).astype(np.uint8)
        if mask_data.shape != ct_data.shape:
            raise ValueError(
                f"The mask shape is {mask_data.shape}, but the CT shape is {ct_data.shape}."
            )
        mask_affine = mask_img.affine
        mask_header = mask_img.header.copy()
    else:
        mask_data = np.zeros(ct_data.shape, dtype=np.uint8)
        mask_affine = ct_img.affine
        mask_header = ct_img.header.copy()
    nib.save(nib.Nifti1Image(mask_data, mask_affine, mask_header), prepared_mask)

    metadata = {
        "patient_id": patient_id,
        "input_mode": "nifti",
        "ct_nifti": str(prepared_ct),
        "mask_nifti": str(prepared_mask),
        "has_annotations": bool(has_annotations and mask_path is not None),
        "ct_shape": list(ct_data.shape),
    }
    (prepared_dir / "input_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


def build_xray_views(params: JobParams) -> dict[str, dict]:
    selected = {name: XRAY_VIEWS[name] for name in params.deepdrr.views if name in XRAY_VIEWS}
    for item in params.deepdrr.custom_views:
        name = str(item["name"]).strip()
        if not name:
            continue
        selected[name] = {
            "description": str(item.get("description") or name),
            "alpha": float(item["alpha"]),
            "beta": float(item["beta"]),
            "gamma": float(item["gamma"]),
        }
    if not selected:
        raise ValueError("Select at least one kV/DeepDRR view.")
    return selected


def zip_outputs(job_dir: Path, include_annotations: bool) -> Path:
    zip_path = job_dir / "results.zip"
    if zip_path.exists():
        zip_path.unlink()

    roots = [job_dir / "prepared", job_dir / "4dct", job_dir / "drr"]
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for root in roots:
            if not root.exists():
                continue
            for path in sorted(root.rglob("*")):
                if not path.is_file():
                    continue
                rel = path.relative_to(job_dir)
                rel_parts = set(rel.parts)
                if not include_annotations and (
                    "annotations" in rel_parts or "images_annotated" in rel_parts
                ):
                    continue
                archive.write(path, rel.as_posix())
    return zip_path


def run_pipeline(
    job_dir: Path,
    ct_upload: Path,
    mask_upload: Path | None,
    params: JobParams,
    log: LogFn,
    progress: ProgressFn | None = None,
) -> Path:
    def set_progress(stage: str, state: str, percent: int, artifacts: dict | None = None) -> None:
        if progress is not None:
            progress(stage, state, percent, artifacts)

    patient_id = safe_patient_id(params.dynagan.patient_id)
    prepared_dir = job_dir / "prepared" / patient_id
    fourd_dir = job_dir / "4dct" / patient_id
    drr_dir = job_dir / "drr" / patient_id
    preview_dir = job_dir / "previews"
    for path in [
        prepared_dir / "images",
        prepared_dir / "annotations",
        fourd_dir / "images",
        fourd_dir / "annotations",
        fourd_dir / "dvf",
        drr_dir,
        job_dir / "tmp",
    ]:
        path.mkdir(parents=True, exist_ok=True)

    set_progress("input", "running", 10)
    log("Preparing input CT and optional tumor mask.")
    prepare_uploaded_nifti(
        patient_id,
        ct_upload,
        mask_upload,
        prepared_dir,
        params.has_annotations,
    )
    ct_preview_paths = write_ct_previews(
        prepared_dir / "images" / f"{patient_id}_ct_phase_00.nii.gz",
        preview_dir,
    )
    set_progress(
        "input",
        "done",
        100,
        {
            name: rel_url(job_dir, Path(path))
            for name, path in ct_preview_paths.items()
        },
    )

    if params.run_4dct:
        set_progress("fourdct", "running", 15)
        log("Running Dynagan to generate the respiratory 4DCT library.")
        run_dynagan_4dct(
            patient_id,
            prepared_dir,
            fourd_dir,
            settings.dynagan_dir,
            settings.dynagan_python,
            execute=True,
            alpha_steps=params.dynagan.alpha_steps,
            gpu_id=params.dynagan.gpu_id,
            alpha_min=params.dynagan.alpha_min,
            alpha_max=params.dynagan.alpha_max,
        )
        gif_path = write_4dct_gif(fourd_dir, preview_dir, patient_id)
        set_progress(
            "fourdct",
            "done",
            100,
            {"gif": rel_url(job_dir, gif_path)} if gif_path is not None else None,
        )
    else:
        set_progress("fourdct", "running", 50)
        log("4DCT stage disabled: copying phase 00 CT into the 4DCT output.")
        shutil.copy2(
            prepared_dir / "images" / f"{patient_id}_ct_phase_00.nii.gz",
            fourd_dir / "images" / f"{patient_id}_ct_phase_00.nii.gz",
        )
        shutil.copy2(
            prepared_dir / "annotations" / f"{patient_id}_tumor_mask_phase_00.nii.gz",
            fourd_dir / "annotations" / f"{patient_id}_tumor_mask_phase_00.nii.gz",
        )
        gif_path = write_4dct_gif(fourd_dir, preview_dir, patient_id)
        set_progress(
            "fourdct",
            "done",
            100,
            {"gif": rel_url(job_dir, gif_path)} if gif_path is not None else None,
        )

    if params.run_deepdrr:
        if settings.deepdrr_mode == "singularity" and settings.deepdrr_sif is None:
            raise RuntimeError("DEEPDRR_SIF must point to the DeepDRR .sif container.")
        set_progress("deepdrr", "running", 20)
        log("Running DeepDRR to generate kV images.")
        run_deepdrr(
            patient_id,
            fourd_dir,
            drr_dir,
            settings.deepdrr_sif,
            job_dir,
            execute=True,
            clear_output=True,
            xray_views=build_xray_views(params),
            sensor_width=params.deepdrr.sensor_width,
            sensor_height=params.deepdrr.sensor_height,
            pixel_size=params.deepdrr.pixel_size,
            source_to_detector_distance=params.deepdrr.source_to_detector_distance,
            source_to_isocenter_vertical_distance=params.deepdrr.source_to_isocenter_vertical_distance,
            preview_size=params.deepdrr.preview_size,
            include_annotations=params.deepdrr.include_annotations,
            mode=settings.deepdrr_mode,
        )
        deepdrr_gifs = [
            {"label": item["label"], "url": rel_url(job_dir, Path(item["url_path"]))}
            for item in collect_deepdrr_gifs(drr_dir)
        ]
        set_progress("deepdrr", "done", 100, {"gifs": deepdrr_gifs})
    else:
        log("DeepDRR stage disabled.")
        set_progress("deepdrr", "skipped", 100)

    set_progress("package", "running", 50)
    log("Packaging outputs.")
    zip_path = zip_outputs(job_dir, params.deepdrr.include_annotations)
    set_progress("package", "done", 100, {"zip": rel_url(job_dir, zip_path)})
    return zip_path
