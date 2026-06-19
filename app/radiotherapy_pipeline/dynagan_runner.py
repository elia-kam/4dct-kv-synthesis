from __future__ import annotations

import glob
import json
import shutil
import subprocess
import textwrap
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

TOTAL_PHASES = 10
# The standard pipeline keeps the original CT and adds nine deformed volumes.
ALPHA_STEPS = TOTAL_PHASES - 1
GPU_ID = "0"


def normalize_gpu_ids(gpu_ids: str | int) -> str:
    value = str(gpu_ids).strip()
    return value or GPU_ID


def primary_gpu_id(gpu_ids: str | int) -> int:
    first_id = normalize_gpu_ids(gpu_ids).split(",", 1)[0].strip()
    try:
        return int(first_id)
    except ValueError:
        return 0


def run_cmd(cmd: list[str], cwd: Path | None = None, execute: bool = False) -> None:
    print("CMD:", " ".join(cmd))
    if execute:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            text=True,
            capture_output=True,
        )
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)
        if result.returncode != 0:
            stdout_tail = "\n".join(result.stdout.splitlines()[-40:])
            stderr_tail = "\n".join(result.stderr.splitlines()[-40:])
            raise RuntimeError(
                "Dynagan command failed with exit code "
                f"{result.returncode}.\n\nSTDOUT tail:\n{stdout_tail}\n\nSTDERR tail:\n{stderr_tail}"
            )
    else:
        print("DRY-RUN: add --execute to run this command.")


def resample_to_shape(data: np.ndarray, target_shape=(128, 128, 128), order=1) -> np.ndarray:
    factors = [target_shape[i] / data.shape[i] for i in range(3)]
    return zoom(data, factors, order=order)


def make_resampled_affine(old_affine: np.ndarray, old_shape, new_shape) -> np.ndarray:
    new_affine = old_affine.copy()
    scale = np.array(old_shape) / np.array(new_shape)
    new_affine[:3, 0] *= scale[0]
    new_affine[:3, 1] *= scale[1]
    new_affine[:3, 2] *= scale[2]
    return new_affine


def prepare_dynagan_inputs(
    prepared_ct: str | Path,
    prepared_mask: str | Path,
    dynagan_dir: str | Path,
    case_id: str = "0001",
) -> tuple[Path, Path]:
    dynagan_dir = Path(dynagan_dir)
    images_ts = dynagan_dir / "datasets" / "imagesTs"
    tumor_dir = dynagan_dir / "datasets" / "tumor"
    images_ts.mkdir(parents=True, exist_ok=True)
    tumor_dir.mkdir(parents=True, exist_ok=True)
    clear_directory(images_ts)
    clear_directory(tumor_dir)

    ct_img = nib.load(str(prepared_ct))
    mask_img = nib.load(str(prepared_mask))
    ct_original = ct_img.get_fdata().astype(np.float32)
    mask_original = mask_img.get_fdata().astype(np.float32)

    ct_128 = resample_to_shape(ct_original, order=1).astype(np.float32)
    mask_128 = (resample_to_shape(mask_original, order=0) > 0.5).astype(np.uint8)
    affine_128 = make_resampled_affine(ct_img.affine, ct_original.shape, ct_128.shape)

    filename = f"LungCT_{case_id}_0000.nii.gz"
    ct_out = images_ts / filename
    mask_out = tumor_dir / filename
    nib.save(nib.Nifti1Image(ct_128, affine_128), ct_out)
    nib.save(nib.Nifti1Image(mask_128, affine_128), mask_out)
    return ct_out, mask_out


def dynagan_test_command(
    dynagan_python: str | Path,
    alpha_steps: int = ALPHA_STEPS,
    gpu_id: str | int = GPU_ID,
    alpha_min: float = 0.0,
    alpha_max: float = 2.0,
) -> list[str]:
    return [
        str(dynagan_python),
        "./test_3D.py",
        "--dataroot",
        "./datasets/",
        "--name",
        "pretrained_model",
        "--model",
        "test",
        "--dataset_mode",
        "test",
        "--num_test",
        "1",
        "--gpu_ids",
        normalize_gpu_ids(gpu_id),
        "--isTumor",
        "True",
        "--alpha_min",
        str(float(alpha_min)),
        "--alpha_max",
        str(float(alpha_max)),
        "--alpha_step",
        str(alpha_steps),
        "--loop",
        "1",
    ]


def python_imports_torch(python_path: str | Path) -> tuple[bool, str]:
    result = subprocess.run(
        [str(python_path), "-c", "import torch; print(torch.__version__)"],
        text=True,
        capture_output=True,
    )
    return result.returncode == 0, result.stderr.strip()


def validate_dynagan_installation(dynagan_dir: str | Path, dynagan_python: str | Path) -> str:
    dynagan_dir = Path(dynagan_dir)
    test_script = dynagan_dir / "test_3D.py"
    if not dynagan_dir.exists():
        raise RuntimeError(
            "Dynagan is not installed at "
            f"{dynagan_dir}. Set DYNAGAN_DIR to a valid Dynagan checkout, "
            "or run the Docker setup so Dynagan is installed inside the image."
        )
    if not test_script.exists():
        raise RuntimeError(
            "Dynagan installation is incomplete: "
            f"{test_script} was not found. DYNAGAN_DIR must point to the Dynagan "
            "repository root containing test_3D.py."
        )
    checkpoint = dynagan_dir / "checkpoints" / "pretrained_model" / "latest_net_G.pth"
    if not checkpoint.exists():
        raise RuntimeError(
            "Dynagan pretrained checkpoint is missing: "
            f"{checkpoint}. Rebuild the Docker image so the pretrained model is "
            "downloaded, or provide the checkpoint at this path."
        )
    if not shutil.which(str(dynagan_python)) and not Path(dynagan_python).exists():
        raise RuntimeError(
            "Dynagan Python executable was not found: "
            f"{dynagan_python}. Set DYNAGAN_PYTHON to the Python environment "
            "where Dynagan dependencies are installed."
        )
    ok, stderr = python_imports_torch(dynagan_python)
    if ok:
        return str(dynagan_python)

    docker_venv_python = Path("/opt/dynagan-venv/bin/python")
    if Path(dynagan_python) != docker_venv_python and docker_venv_python.exists():
        fallback_ok, fallback_stderr = python_imports_torch(docker_venv_python)
        if fallback_ok:
            print(
                "INFO: DYNAGAN_PYTHON could not import torch; "
                f"using Docker Dynagan venv instead: {docker_venv_python}"
            )
            return str(docker_venv_python)
        stderr = f"{stderr}\n\nFallback {docker_venv_python} STDERR:\n{fallback_stderr}"

    raise RuntimeError(
        "Dynagan Python is missing required dependencies. "
        f"Could not import torch with {dynagan_python}.\n"
        f"STDERR:\n{stderr}\n\n"
        "If you are using Docker, rebuild the image with: "
        "sudo docker compose build --no-cache"
    )


def clear_directory(path: Path) -> None:
    if not path.exists():
        return
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def clear_dynagan_patient_outputs(result_root: Path, case_id: str = "0001") -> None:
    clear_directory(result_root / case_id)
    fourdct_file = result_root / f"LungCT_{case_id}_4DCT.nii.gz"
    if fourdct_file.exists():
        fourdct_file.unlink()


def clear_final_patient_outputs(final_dataset_dir: Path) -> None:
    for folder_name in ("images", "annotations", "dvf"):
        clear_directory(final_dataset_dir / folder_name)


def write_high_postprocess_script(
    patient_id: str,
    prepared_ct: str | Path,
    prepared_mask: str | Path,
    dvf_dir: str | Path,
    final_dataset_dir: str | Path,
    dynagan_dir: str | Path,
    expected_dvf_count: int = ALPHA_STEPS,
    gpu_id: str | int = GPU_ID,
) -> Path:
    script_path = Path(dynagan_dir) / f"run_high_postprocess_{patient_id}.py"
    script = f"""
import os

dynagan_threads = int(os.environ.get("DYNAGAN_THREADS", "8"))
os.environ.setdefault("OMP_NUM_THREADS", str(dynagan_threads))
os.environ.setdefault("MKL_NUM_THREADS", str(dynagan_threads))

import glob
import json
import numpy as np
import nibabel as nib
import torch
from scipy import ndimage

from util.spatialTransform import SpatialTransformer

patient_id = {patient_id!r}
prepared_ct = r"{prepared_ct}"
prepared_mask = r"{prepared_mask}"
dvf_dir = r"{dvf_dir}"
final_dataset_dir = r"{final_dataset_dir}"
expected_dvf_count = {expected_dvf_count}
gpu_id = {primary_gpu_id(gpu_id)}

torch.set_num_threads(max(1, dynagan_threads))
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

def select_torch_device(gpu_id):
    if gpu_id < 0:
        print(f"Postprocess device: CPU; torch threads: {{dynagan_threads}}")
        return torch.device("cpu")
    if not torch.cuda.is_available():
        print(f"Postprocess device: CPU; torch threads: {{dynagan_threads}}")
        return torch.device("cpu")
    try:
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{{gpu_id}}")
        test = torch.ones((1,), device=device)
        del test
        torch.cuda.synchronize()
    except Exception as exc:
        print(f"CUDA unavailable for postprocessing, falling back to CPU: {{exc}}")
        return torch.device("cpu")
    print(f"Postprocess device: CUDA {{gpu_id}} {{torch.cuda.get_device_name(gpu_id)}}; torch threads: {{dynagan_threads}}")
    return device

device = select_torch_device(gpu_id)

final_images_dir = os.path.join(final_dataset_dir, "images")
final_annotations_dir = os.path.join(final_dataset_dir, "annotations")
final_dvf_dir = os.path.join(final_dataset_dir, "dvf")
for folder in [final_images_dir, final_annotations_dir, final_dvf_dir]:
    os.makedirs(folder, exist_ok=True)

def bbox_from_mask(mask_arr):
    coords = np.where(mask_arr > 0.5)
    if len(coords[0]) == 0:
        return None
    x, y, z = coords
    return {{"xmin": int(x.min()), "xmax": int(x.max()), "ymin": int(y.min()), "ymax": int(y.max()), "zmin": int(z.min()), "zmax": int(z.max())}}

def resize_dvf_to_original(dvf_arr, original_shape):
    zoom_factors = (original_shape[0] / dvf_arr.shape[0], original_shape[1] / dvf_arr.shape[1], original_shape[2] / dvf_arr.shape[2], 1)
    dvf_high = ndimage.zoom(dvf_arr, zoom=zoom_factors, order=1).astype(np.float32)
    scale = np.array([original_shape[0] / dvf_arr.shape[0], original_shape[1] / dvf_arr.shape[1], original_shape[2] / dvf_arr.shape[2]], dtype=np.float32)
    dvf_high[..., 0] *= scale[0]
    dvf_high[..., 1] *= scale[1]
    dvf_high[..., 2] *= scale[2]
    return dvf_high

def warp_tensor_with_dvf(moving_t, transform, dvf_xyz3, mode="image"):
    field = dvf_xyz3.transpose(3, 0, 1, 2).astype(np.float32)
    field_t = torch.from_numpy(field).unsqueeze(0).to(device)
    with torch.no_grad():
        warped_t = transform(moving_t, field_t)
    warped = warped_t.detach().cpu().numpy()[0, 0]
    del field_t, warped_t
    return (warped > 0.5).astype(np.uint8) if mode == "mask" else warped.astype(np.float32)

ct_img = nib.load(prepared_ct)
mask_img = nib.load(prepared_mask)
ct = ct_img.get_fdata().astype(np.float32)
mask = (mask_img.get_fdata().astype(np.float32) > 0.5).astype(np.float32)
affine = ct_img.affine
header = ct_img.header.copy()
ct_t = torch.from_numpy(ct).unsqueeze(0).unsqueeze(1).to(device)
mask_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(1).to(device)
image_transform = SpatialTransformer(np.asarray(ct.shape)).to(device)
mask_transform = SpatialTransformer(np.asarray(ct.shape), "nearest").to(device)

all_bboxes = {{}}
out_ct_00 = os.path.join(final_images_dir, f"{{patient_id}}_ct_phase_00.nii.gz")
out_mask_00 = os.path.join(final_annotations_dir, f"{{patient_id}}_tumor_mask_phase_00.nii.gz")
nib.save(nib.Nifti1Image(ct.astype(np.float32), affine, header), out_ct_00)
nib.save(nib.Nifti1Image(mask.astype(np.uint8), affine, mask_img.header.copy()), out_mask_00)
all_bboxes["phase_00"] = bbox_from_mask(mask)

dvf_files = sorted(glob.glob(os.path.join(dvf_dir, "*.nii.gz")))
if len(dvf_files) != expected_dvf_count:
    raise RuntimeError(
        f"Unexpected DVF count: {{len(dvf_files)}} file(s), expected {{expected_dvf_count}}. "
        f"Clean {{dvf_dir}} before rerunning."
    )

for idx, dvf_path in enumerate(dvf_files, start=1):
    phase_name = f"phase_{{idx:02d}}"
    dvf = nib.load(dvf_path).get_fdata().astype(np.float32)
    if dvf.ndim == 4 and dvf.shape[0] == 3 and dvf.shape[-1] != 3:
        dvf = np.moveaxis(dvf, 0, -1)
    if dvf.ndim != 4 or dvf.shape[-1] != 3:
        raise RuntimeError(f"Unrecognized DVF format: {{dvf.shape}}")

    dvf_high = resize_dvf_to_original(dvf, ct.shape)
    warped_ct = warp_tensor_with_dvf(ct_t, image_transform, dvf_high, mode="image")
    warped_mask = warp_tensor_with_dvf(mask_t, mask_transform, dvf_high, mode="mask")

    out_dvf = os.path.join(final_dvf_dir, f"{{patient_id}}_dvf_{{phase_name}}.nii.gz")
    out_ct = os.path.join(final_images_dir, f"{{patient_id}}_ct_{{phase_name}}.nii.gz")
    out_mask = os.path.join(final_annotations_dir, f"{{patient_id}}_tumor_mask_{{phase_name}}.nii.gz")
    nib.save(nib.Nifti1Image(dvf_high.astype(np.float32), affine), out_dvf)
    nib.save(nib.Nifti1Image(warped_ct.astype(np.float32), affine, header), out_ct)
    nib.save(nib.Nifti1Image(warped_mask.astype(np.uint8), affine, mask_img.header.copy()), out_mask)
    all_bboxes[phase_name] = bbox_from_mask(warped_mask)
    if device.type == "cuda":
        torch.cuda.empty_cache()

summary_json = os.path.join(final_annotations_dir, f"{{patient_id}}_all_bboxes_4d_HIGH.json")
with open(summary_json, "w", encoding="utf-8") as f:
    json.dump({{"patient_id": patient_id, "phases": all_bboxes, "total_volumes_including_phase_00": len(all_bboxes)}}, f, indent=2)
print("HIGH dataset complete:", final_dataset_dir)
"""
    script_path.write_text(textwrap.dedent(script), encoding="utf-8")
    return script_path


def run_dynagan_4dct(
    patient_id: str,
    prepared_patient_dir: str | Path,
    final_dataset_dir: str | Path,
    dynagan_dir: str | Path,
    dynagan_python: str | Path,
    execute: bool = False,
    alpha_steps: int = ALPHA_STEPS,
    gpu_id: str | int = GPU_ID,
    alpha_min: float = 0.0,
    alpha_max: float = 2.0,
) -> None:
    prepared_patient_dir = Path(prepared_patient_dir)
    dynagan_dir = Path(dynagan_dir)
    final_dataset_dir = Path(final_dataset_dir)
    prepared_ct = prepared_patient_dir / "images" / f"{patient_id}_ct_phase_00.nii.gz"
    prepared_mask = prepared_patient_dir / "annotations" / f"{patient_id}_tumor_mask_phase_00.nii.gz"
    result_root = dynagan_dir / "results" / "pretrained_model"

    if execute:
        dynagan_python = validate_dynagan_installation(dynagan_dir, dynagan_python)
        print("INFO: cleaning previous Dynagan results:", result_root / "0001")
        clear_dynagan_patient_outputs(result_root)
        print("INFO: cleaning previous 4DCT outputs:", final_dataset_dir)
        clear_final_patient_outputs(final_dataset_dir)
        prepare_dynagan_inputs(prepared_ct, prepared_mask, dynagan_dir)
    else:
        print("DRY-RUN: would prepare 128^3 Dynagan inputs from:")
        print("  CT:", prepared_ct)
        print("  Mask:", prepared_mask)

    run_cmd(
        dynagan_test_command(
            dynagan_python,
            alpha_steps=alpha_steps,
            gpu_id=gpu_id,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
        ),
        cwd=dynagan_dir,
        execute=execute,
    )

    dvf_dir = result_root / "0001" / "dvf"
    if execute:
        script_path = write_high_postprocess_script(
            patient_id,
            prepared_ct,
            prepared_mask,
            dvf_dir,
            final_dataset_dir,
            dynagan_dir,
            expected_dvf_count=alpha_steps,
            gpu_id=gpu_id,
        )
    else:
        script_path = dynagan_dir / f"run_high_postprocess_{patient_id}.py"
        print("DRY-RUN: would create the HIGH postprocess script:", script_path)

    run_cmd([str(dynagan_python), str(script_path)], cwd=dynagan_dir, execute=execute)
