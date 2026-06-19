from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from .jobs import create_job_dir, read_status, run_job_background
from .pipeline import safe_patient_id
from .radiotherapy_pipeline.deepdrr_runner import XRAY_VIEWS
from .schemas import DeepDRRParams, DynaganParams, JobParams
from .settings import settings

APP_DIR = Path(__file__).resolve().parent

app = FastAPI(title="CT 4DCT kV Generator", version="0.1.0")
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")
templates = Jinja2Templates(directory=APP_DIR / "templates")


def parse_custom_views(raw: str) -> tuple[dict, ...]:
    raw = raw.strip()
    if not raw:
        return ()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid custom views JSON: {exc}") from exc
    if not isinstance(data, list):
        raise HTTPException(status_code=400, detail="Custom views must be a JSON list.")
    required = {"name", "alpha", "beta", "gamma"}
    for item in data:
        if not isinstance(item, dict) or not required.issubset(item):
            raise HTTPException(
                status_code=400,
                detail="Each custom view must contain name, alpha, beta, and gamma.",
            )
    return tuple(data)


def selected_views(view_names: list[str] | None) -> tuple[str, ...]:
    if not view_names:
        return ("FACE_AP", "KV_LEFT_45", "KV_RIGHT_45")
    unknown = sorted(set(view_names) - set(XRAY_VIEWS))
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown view(s): {unknown}")
    return tuple(view_names)


def validate_gpu_ids(raw: str) -> str:
    value = raw.strip() or "0"
    parts = [part.strip() for part in value.split(",")]
    try:
        ids = [int(part) for part in parts]
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="GPU IDs must be an integer like 0, a comma list like 0,1, or -1 for CPU.",
        ) from exc
    if any(part == "" for part in parts) or not ids:
        raise HTTPException(status_code=400, detail="GPU IDs cannot be empty.")
    if -1 in ids and len(ids) > 1:
        raise HTTPException(status_code=400, detail="Use -1 alone for CPU mode.")
    return ",".join(str(gpu_id) for gpu_id in ids)


async def save_upload(upload: UploadFile, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as out:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    return destination


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"views": XRAY_VIEWS, "settings": settings},
    )


@app.post("/jobs")
async def create_job(
    patient_id: str = Form("patient"),
    ct_file: UploadFile = File(...),
    mask_file: Optional[UploadFile] = File(None),
    run_4dct: bool = Form(True),
    run_deepdrr: bool = Form(True),
    include_annotations: bool = Form(True),
    alpha_min: float = Form(0.0),
    alpha_max: float = Form(2.0),
    phase_count: int = Form(10),
    gpu_id: str = Form("0"),
    view_names: Optional[List[str]] = Form(None),
    custom_views_json: str = Form(""),
    sensor_width: int = Form(1024),
    sensor_height: int = Form(1024),
    pixel_size: float = Form(1.0),
    source_to_detector_distance: float = Form(1020.0),
    source_to_isocenter_vertical_distance: float = Form(510.0),
    preview_size: int = Form(512),
) -> RedirectResponse:
    if phase_count < 2 or phase_count > 30:
        raise HTTPException(status_code=400, detail="phase_count must be between 2 and 30.")
    if preview_size < 128 or preview_size > 2048:
        raise HTTPException(status_code=400, detail="preview_size must be between 128 and 2048.")

    clean_patient_id = safe_patient_id(patient_id)
    job_dir = create_job_dir()
    uploads_dir = job_dir / "uploads"
    ct_upload = await save_upload(ct_file, uploads_dir / (ct_file.filename or "ct.nii.gz"))
    mask_upload = None
    if mask_file is not None and mask_file.filename:
        mask_upload = await save_upload(mask_file, uploads_dir / mask_file.filename)

    params = JobParams(
        dynagan=DynaganParams(
            patient_id=clean_patient_id,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            phase_count=phase_count,
            gpu_id=validate_gpu_ids(gpu_id),
        ),
        deepdrr=DeepDRRParams(
            views=selected_views(view_names),
            custom_views=parse_custom_views(custom_views_json),
            sensor_width=sensor_width,
            sensor_height=sensor_height,
            pixel_size=pixel_size,
            source_to_detector_distance=source_to_detector_distance,
            source_to_isocenter_vertical_distance=source_to_isocenter_vertical_distance,
            preview_size=preview_size,
            include_annotations=include_annotations and mask_upload is not None,
        ),
        run_4dct=run_4dct,
        run_deepdrr=run_deepdrr,
        has_annotations=mask_upload is not None,
    )
    shutil.copy2(job_dir / "status.json", job_dir / "status.initial.json")
    (job_dir / "params.json").write_text(json.dumps(params, default=lambda o: o.__dict__, indent=2), encoding="utf-8")
    run_job_background(job_dir, ct_upload, mask_upload, params)
    return RedirectResponse(url=f"/jobs/{job_dir.name}", status_code=303)


@app.get("/jobs/{job_id}", response_class=HTMLResponse)
async def job_page(request: Request, job_id: str) -> HTMLResponse:
    status = read_status(job_id)
    if status["status"] == "missing":
        raise HTTPException(status_code=404, detail="Job not found.")
    return templates.TemplateResponse(
        request=request,
        name="job.html",
        context={"job_id": job_id, "status": status},
    )


@app.get("/api/jobs/{job_id}")
async def job_status(job_id: str) -> dict:
    status = read_status(job_id)
    if status["status"] == "missing":
        raise HTTPException(status_code=404, detail="Job not found.")
    return status


@app.get("/jobs/{job_id}/files/{file_path:path}")
async def job_file(job_id: str, file_path: str) -> FileResponse:
    job_dir = (settings.jobs_dir / job_id).resolve()
    requested = (job_dir / file_path).resolve()
    if not requested.is_file() or job_dir not in requested.parents:
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(requested)


@app.get("/jobs/{job_id}/download")
async def download(job_id: str) -> FileResponse:
    job_dir = settings.jobs_dir / job_id
    zip_path = job_dir / "results.zip"
    if not zip_path.exists():
        raise HTTPException(status_code=404, detail="Results are not available yet.")
    return FileResponse(zip_path, filename=f"{job_id}_ct4d_kv_results.zip")


@app.get("/healthz")
async def healthz() -> dict:
    return {"ok": True}
