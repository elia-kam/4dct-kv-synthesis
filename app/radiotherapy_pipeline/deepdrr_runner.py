from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path


CONTAINER_COMPYTE = "/usr/local/lib/python3.8/dist-packages/pycuda/compyte"


XRAY_VIEWS = {
    "LAT_PROFIL": {"description": "Lateral / profile view", "alpha": 0, "beta": 0, "gamma": 0},
    "OBL_40": {"description": "40 degree oblique view", "alpha": 40, "beta": 0, "gamma": 0},
    "FACE_AP": {"description": "AP frontal chest view", "alpha": 0, "beta": 90, "gamma": 90},
    "KV_LEFT_45": {"description": "Left 45 degree oblique kV view", "alpha": -45, "beta": 0, "gamma": 0},
    "KV_RIGHT_45": {"description": "Right 45 degree oblique kV view", "alpha": 45, "beta": 0, "gamma": 0},
}


DISPLAY_TRANSFORMS = {
    "LAT_PROFIL": {"rot90": 0, "flip_lr": False, "flip_ud": False},
    "OBL_40": {"rot90": 0, "flip_lr": False, "flip_ud": False},
    "FACE_AP": {"rot90": 0, "flip_lr": False, "flip_ud": True},
    "KV_LEFT_45": {"rot90": 0, "flip_lr": False, "flip_ud": False},
    "KV_RIGHT_45": {"rot90": 0, "flip_lr": False, "flip_ud": False},
}


def singularity_bind_path() -> str:
    return os.environ.get("DEEPDRR_BIND", f"{Path.home()}:{Path.home()}")


def run_cmd(cmd: list[str], execute: bool = False, env: dict | None = None) -> subprocess.CompletedProcess | None:
    print("CMD:", " ".join(map(str, cmd)))
    if not execute:
        print("DRY-RUN: add --execute to run this command.")
        return None
    result = subprocess.run(cmd, text=True, capture_output=True, env=env)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    if result.returncode != 0:
        stdout_tail = "\n".join(result.stdout.splitlines()[-60:])
        stderr_tail = "\n".join(result.stderr.splitlines()[-60:])
        raise RuntimeError(
            "DeepDRR command failed with exit code "
            f"{result.returncode}.\n\nSTDOUT tail:\n{stdout_tail}\n\nSTDERR tail:\n{stderr_tail}"
        )
    return result


def runtime_env(output_root: str | Path) -> dict:
    output_root = Path(output_root)
    cache_root = output_root / "tmp" / "deepdrr_cache"
    env = {
        "PYCUDA_CACHE_DIR": str(cache_root / "pycuda_cache"),
        "XDG_CACHE_HOME": str(cache_root / ".cache"),
        "CUDA_CACHE_PATH": str(cache_root / "cuda_cache"),
        "TMPDIR": str(cache_root / "tmp"),
    }
    for path in env.values():
        Path(path).mkdir(parents=True, exist_ok=True)
    return env


def subprocess_env(output_root: str | Path) -> dict:
    env = dict(os.environ)
    env.update(runtime_env(output_root))
    return env


def prepare_pycuda_compyte_patch(
    sif_path: str | Path,
    patch_root: str | Path,
    execute: bool = False,
) -> Path:
    patch_root = Path(patch_root)
    patched_compyte = patch_root / "pycuda" / "compyte"

    if patched_compyte.exists() and list(patched_compyte.glob("*.py")):
        print("PyCUDA patch already exists:", patched_compyte)
        return patched_compyte

    patched_compyte.mkdir(parents=True, exist_ok=True)
    cmd_list = [
        "singularity",
        "exec",
        "--bind",
        singularity_bind_path(),
        str(sif_path),
        "bash",
        "-lc",
        f"find {CONTAINER_COMPYTE} -maxdepth 1 -name '*.py' -type f",
    ]

    result = run_cmd(cmd_list, execute=execute)
    if result is None:
        return patched_compyte

    files = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not files:
        raise RuntimeError("No pycuda/compyte file found in the SIF.")

    for container_file in files:
        host_file = patched_compyte / Path(container_file).name
        cmd_cat = [
            "singularity",
            "exec",
            "--bind",
            singularity_bind_path(),
            str(sif_path),
            "cat",
            container_file,
        ]
        cat_result = run_cmd(cmd_cat, execute=True)
        assert cat_result is not None
        src = cat_result.stdout
        lines = src.splitlines()
        if "from __future__ import annotations" not in lines[:10]:
            insert_at = 0
            while insert_at < len(lines) and (
                lines[insert_at].startswith("#!") or "coding" in lines[insert_at]
            ):
                insert_at += 1
            lines.insert(insert_at, "from __future__ import annotations")
            src = "\n".join(lines) + "\n"
        host_file.write_text(src, encoding="utf-8")

    print("PyCUDA patch created:", patched_compyte)
    return patched_compyte


def write_deepdrr_config_and_script(
    patient_id: str,
    input_4d_dir: str | Path,
    out_root: str | Path,
    tmp_root: str | Path,
    output_root: str | Path,
    xray_views: dict | None = None,
    sensor_width: int = 1024,
    sensor_height: int = 1024,
    pixel_size: float = 1.0,
    source_to_detector_distance: float = 1020.0,
    source_to_isocenter_vertical_distance: float = 510.0,
    preview_size: int = 512,
    include_annotations: bool = True,
) -> tuple[Path, Path]:
    input_4d_dir = Path(input_4d_dir)
    out_root = Path(out_root)
    tmp_root = Path(tmp_root)
    tmp_root.mkdir(parents=True, exist_ok=True)

    cfg = runtime_env(output_root)
    selected_views = xray_views or XRAY_VIEWS
    selected_transforms = {
        name: DISPLAY_TRANSFORMS.get(name, {"rot90": 0, "flip_lr": False, "flip_ud": False})
        for name in selected_views
    }
    config = {
        "patient_id": patient_id,
        "ct_dir": str(input_4d_dir / "images"),
        "mask_dir": str(input_4d_dir / "annotations"),
        "out_root": str(out_root),
        "tmp_root": str(tmp_root),
        "sensor_width": int(sensor_width),
        "sensor_height": int(sensor_height),
        "pixel_size": float(pixel_size),
        "source_to_detector_distance": float(source_to_detector_distance),
        "source_to_isocenter_vertical_distance": float(source_to_isocenter_vertical_distance),
        "preview_size": int(preview_size),
        "include_annotations": bool(include_annotations),
        "xray_views": selected_views,
        "display_transforms": selected_transforms,
        **cfg,
    }

    config_path = tmp_root / "deepdrr_config.json"
    script_path = tmp_root / "run_deepdrr_generated.py"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    script = r'''
import json
import os
import re
from pathlib import Path

CONFIG_PATH = Path(__file__).with_name("deepdrr_config.json")
cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

for key in ["PYCUDA_CACHE_DIR", "XDG_CACHE_HOME", "CUDA_CACHE_PATH", "TMPDIR"]:
    os.environ[key] = cfg[key]
    Path(cfg[key]).mkdir(parents=True, exist_ok=True)

import imageio.v2 as imageio
import numpy as np
import SimpleITK as sitk
from PIL import Image, ImageDraw
from deepdrr import Volume, MobileCArm
from deepdrr.projector import Projector


def phase_number(path):
    match = re.search(r"phase[_-]?(\d+)", Path(path).name.lower())
    return int(match.group(1)) if match else None


def find_files(folder):
    files = []
    for ext in ["*.mha", "*.mhd", "*.nii", "*.nii.gz", "*.nrrd", "*.nhdr"]:
        files.extend(Path(folder).glob(ext))
    return sorted(files)


def normalize_to_uint8(img):
    img = np.asarray(img, dtype=np.float32)
    lo, hi = np.percentile(img, [1, 99])
    if hi <= lo:
        lo, hi = float(img.min()), float(img.max())
    img = np.clip((img - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return (img * 255).astype(np.uint8)


def apply_transform(arr, transform):
    out = np.asarray(arr)
    rot90 = int(transform.get("rot90", 0)) % 4
    if rot90:
        out = np.rot90(out, k=rot90)
    if bool(transform.get("flip_lr", False)):
        out = np.fliplr(out)
    if bool(transform.get("flip_ud", False)):
        out = np.flipud(out)
    return np.ascontiguousarray(out)


def add_label(img_u8, label):
    img = Image.fromarray(img_u8).convert("RGB")
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, img.width, 42), fill=(0, 0, 0))
    draw.text((10, 12), label, fill=(255, 255, 255))
    return np.array(img)


def bbox_from_projected_mask(mask_2d):
    yy, xx = np.where(mask_2d > 0)
    if len(xx) == 0:
        return None
    return {
        "xmin": int(xx.min()), "xmax": int(xx.max()),
        "ymin": int(yy.min()), "ymax": int(yy.max()),
    }


def centroid_from_projected_mask(mask_2d):
    yy, xx = np.where(mask_2d > 0)
    if len(xx) == 0:
        return None
    return {"x": float(xx.mean()), "y": float(yy.mean())}


def draw_bbox(img_u8, bbox, color=(255, 50, 50), width=4):
    img = Image.fromarray(img_u8).convert("RGB")
    if bbox is not None:
        draw = ImageDraw.Draw(img)
        draw.rectangle(
            (bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]),
            outline=color,
            width=width,
        )
        draw.text((bbox["xmin"] + 4, max(46, bbox["ymin"] - 18)), "TUMOR", fill=color)
    return np.array(img)


def bbox_from_mask(mask_arr):
    zz, yy, xx = np.where(mask_arr > 0)
    if len(xx) == 0:
        return None
    return {
        "xmin": int(xx.min()), "xmax": int(xx.max()),
        "ymin": int(yy.min()), "ymax": int(yy.max()),
        "zmin": int(zz.min()), "zmax": int(zz.max()),
    }


def match_mask(ct_path, mask_files):
    ct_phase = phase_number(ct_path)
    for mask_path in mask_files:
        if phase_number(mask_path) == ct_phase:
            return mask_path
    return None


def project_ct(nifti_path, view_info):
    volume = Volume.from_nifti(str(nifti_path))
    carm = MobileCArm(
        source_to_detector_distance=float(cfg["source_to_detector_distance"]),
        source_to_isocenter_vertical_distance=float(cfg["source_to_isocenter_vertical_distance"]),
        pixel_size=float(cfg["pixel_size"]),
        sensor_height=int(cfg["sensor_height"]),
        sensor_width=int(cfg["sensor_width"]),
    )
    with Projector(volume, carm=carm) as projector:
        volume.orient_patient(head_first=True, supine=True)
        volume.place_center(carm.isocenter_in_world)
        carm.move_to(
            alpha=float(view_info["alpha"]),
            beta=float(view_info["beta"]),
            gamma=float(view_info["gamma"]),
            degrees=True,
        )
        return projector()


def make_projection_mask(mask_path, output_path):
    mask_img = sitk.ReadImage(str(mask_path))
    mask_arr = sitk.GetArrayFromImage(mask_img)
    # DeepDRR projects CT intensities: air background, dense tissue for the tumor.
    projection_arr = np.where(mask_arr > 0, 1000.0, -1000.0).astype(np.float32)
    projection_img = sitk.GetImageFromArray(projection_arr)
    projection_img.CopyInformation(mask_img)
    sitk.WriteImage(projection_img, str(output_path))
    return output_path


def projected_mask_to_binary(mask_projection):
    arr = np.asarray(mask_projection, dtype=np.float32)
    arr = arr - float(np.nanmin(arr))
    peak = float(np.nanmax(arr))
    if peak <= 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    return (arr >= peak * 0.05).astype(np.uint8)


def save_breathing_cycle_gif(path, frames, cycle_seconds=4.0):
    if not frames:
        return
    # Inspiration then expiration, without repeating both endpoints.
    breathing_frames = frames + frames[-2:0:-1]
    duration_ms = max(20, int(round(cycle_seconds * 1000 / len(breathing_frames))))
    pil_frames = [Image.fromarray(frame).convert("P", palette=Image.ADAPTIVE) for frame in breathing_frames]
    pil_frames[0].save(
        str(path),
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


def main():
    patient_id = cfg["patient_id"]
    out_root = Path(cfg["out_root"])
    include_annotations = bool(cfg.get("include_annotations", True))
    preview_size = int(cfg.get("preview_size", 512))
    raw_root = out_root / "images_raw"
    annotated_root = out_root / "images_annotated"
    annotation_root = out_root / "annotations"
    meta_root = out_root / "metadata"
    gif_root = out_root / "gifs"
    projection_mask_root = Path(cfg["tmp_root"]) / "projection_masks"
    base_folders = [raw_root, meta_root, gif_root]
    if include_annotations:
        base_folders.extend([annotated_root, annotation_root, projection_mask_root])
    for folder in base_folders:
        folder.mkdir(parents=True, exist_ok=True)

    ct_files = find_files(cfg["ct_dir"])
    mask_files = find_files(cfg["mask_dir"])
    if not ct_files:
        raise FileNotFoundError(f"No CT found in {cfg['ct_dir']}")

    metadata = {"patient_id": patient_id, "frames": []}
    frames_by_view = {view_name: [] for view_name in cfg["xray_views"]}

    for idx, ct_path in enumerate(ct_files):
        mask_path = match_mask(ct_path, mask_files)
        bbox_3d = None
        if include_annotations and mask_path is not None:
            mask_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path)))
            bbox_3d = bbox_from_mask(mask_arr)
            projection_mask_path = make_projection_mask(
                mask_path, projection_mask_root / f"{patient_id}_projection_mask_phase_{idx:04d}.nii.gz"
            )
        else:
            projection_mask_path = None

        phase_meta = {"phase_index": idx, "source_ct": str(ct_path), "source_mask": str(mask_path) if mask_path else None, "bbox_3d": bbox_3d, "views": {}}

        for view_name, view_info in cfg["xray_views"].items():
            view_dir = raw_root / view_name
            annotated_view_dir = annotated_root / view_name
            annotation_view_dir = annotation_root / view_name
            folders = [view_dir]
            if include_annotations:
                folders.extend([annotated_view_dir, annotation_view_dir])
            for folder in folders:
                folder.mkdir(parents=True, exist_ok=True)

            drr = project_ct(ct_path, view_info)
            img = normalize_to_uint8(drr)
            img = apply_transform(img, cfg["display_transforms"].get(view_name, {}))
            img = np.array(Image.fromarray(img).resize((preview_size, preview_size), resample=Image.BILINEAR))
            preview_img = add_label(img, f"{patient_id} phase {idx:04d} | {view_name}")

            projected_mask = np.zeros((preview_size, preview_size), dtype=np.uint8)
            if projection_mask_path is not None:
                projected_mask = projected_mask_to_binary(project_ct(projection_mask_path, view_info))
                projected_mask = apply_transform(
                    projected_mask, cfg["display_transforms"].get(view_name, {})
                )
                projected_mask = np.array(
                    Image.fromarray(projected_mask).resize((preview_size, preview_size), resample=Image.NEAREST)
                )
            bbox_2d = bbox_from_projected_mask(projected_mask)
            centroid_2d = centroid_from_projected_mask(projected_mask)

            png_path = view_dir / f"{patient_id}_{view_name}_phase_{idx:04d}.png"
            Image.fromarray(img).save(str(png_path))

            view_meta = {
                "png": str(png_path),
                **view_info,
            }
            if include_annotations:
                annotated_img = draw_bbox(preview_img, bbox_2d)
                annotated_png_path = annotated_view_dir / f"{patient_id}_{view_name}_phase_{idx:04d}_annotated.png"
                mask_png_path = annotation_view_dir / f"{patient_id}_{view_name}_phase_{idx:04d}_mask.png"
                annotation_path = annotation_view_dir / f"{patient_id}_{view_name}_phase_{idx:04d}.json"
                Image.fromarray(annotated_img).save(str(annotated_png_path))
                Image.fromarray((projected_mask > 0).astype(np.uint8) * 255).save(str(mask_png_path))
                annotation = {
                    "patient_id": patient_id,
                    "phase_index": idx,
                    "view": view_name,
                    "bbox_format": "pixel_xyxy",
                    "bbox_2d": bbox_2d,
                    "centroid_2d": centroid_2d,
                    "projected_mask_png": str(mask_png_path),
                    "annotated_png": str(annotated_png_path),
                }
                annotation_path.write_text(json.dumps(annotation, indent=2), encoding="utf-8")
                frames_by_view[view_name].append(annotated_img)
                view_meta.update(
                    {
                        "annotated_png": str(annotated_png_path),
                        "annotation_json": str(annotation_path),
                        "projected_mask_png": str(mask_png_path),
                        "bbox_2d": bbox_2d,
                        "centroid_2d": centroid_2d,
                    }
                )
            else:
                frames_by_view[view_name].append(preview_img)
            phase_meta["views"][view_name] = {
                **view_meta,
            }

        metadata["frames"].append(phase_meta)

    for view_name, frames in frames_by_view.items():
        gif_path = gif_root / f"{patient_id}_{view_name}.gif"
        save_breathing_cycle_gif(gif_path, frames, cycle_seconds=4.0)
        metadata[f"gif_{view_name}"] = str(gif_path)

    meta_path = meta_root / f"{patient_id}_deepdrr_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("DeepDRR complete:", out_root)


if __name__ == "__main__":
    main()
'''
    script_path.write_text(textwrap.dedent(script), encoding="utf-8")
    return config_path, script_path


def run_deepdrr(
    patient_id: str,
    input_4d_dir: str | Path,
    out_root: str | Path,
    sif_path: str | Path | None,
    output_root: str | Path,
    execute: bool = False,
    clear_output: bool = False,
    xray_views: dict | None = None,
    sensor_width: int = 1024,
    sensor_height: int = 1024,
    pixel_size: float = 1.0,
    source_to_detector_distance: float = 1020.0,
    source_to_isocenter_vertical_distance: float = 510.0,
    preview_size: int = 512,
    include_annotations: bool = True,
    mode: str = "singularity",
) -> None:
    out_root = Path(out_root)
    output_root = Path(output_root)
    tmp_root = output_root / "tmp" / patient_id / "deepdrr_runtime"
    patch_root = output_root / "tmp" / "deepdrr_sif_patch"

    mode = mode.lower().strip()
    if mode not in {"singularity", "python"}:
        raise ValueError("DeepDRR mode must be 'singularity' or 'python'.")
    if mode == "singularity" and sif_path is None:
        raise ValueError("sif_path is required in singularity mode.")
    patched_compyte = None
    if mode == "singularity":
        patched_compyte = prepare_pycuda_compyte_patch(sif_path, patch_root, execute=execute)
    if execute and clear_output and out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    _, script_path = write_deepdrr_config_and_script(
        patient_id,
        input_4d_dir,
        out_root,
        tmp_root,
        output_root,
        xray_views=xray_views,
        sensor_width=sensor_width,
        sensor_height=sensor_height,
        pixel_size=pixel_size,
        source_to_detector_distance=source_to_detector_distance,
        source_to_isocenter_vertical_distance=source_to_isocenter_vertical_distance,
        preview_size=preview_size,
        include_annotations=include_annotations,
    )
    if mode == "python":
        run_cmd([sys.executable, str(script_path)], execute=execute, env=subprocess_env(output_root))
    else:
        cmd = [
            "singularity",
            "exec",
            "--nv",
            "--bind",
            singularity_bind_path(),
            "--bind",
            f"{patched_compyte}:{CONTAINER_COMPYTE}",
            str(sif_path),
            "python3",
            str(script_path),
        ]
        run_cmd(cmd, execute=execute)
