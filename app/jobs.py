from __future__ import annotations

import json
import copy
import os
import threading
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .pipeline import run_pipeline
from .schemas import JobParams
from .settings import settings

LOCK = threading.Lock()

DEFAULT_STAGES = {
    "input": {"label": "Input CT", "status": "queued", "progress": 0, "artifacts": {}},
    "fourdct": {"label": "4DCT generation", "status": "queued", "progress": 0, "artifacts": {}},
    "deepdrr": {"label": "DeepDRR kV images", "status": "queued", "progress": 0, "artifacts": {}},
    "package": {"label": "Result package", "status": "queued", "progress": 0, "artifacts": {}},
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_job_dir() -> Path:
    settings.jobs_dir.mkdir(parents=True, exist_ok=True)
    job_dir = settings.jobs_dir / uuid.uuid4().hex
    (job_dir / "uploads").mkdir(parents=True, exist_ok=False)
    write_status(
        job_dir,
        {
            "status": "queued",
            "progress": 0,
            "created_at": utc_now(),
            "logs": [],
            "stages": copy.deepcopy(DEFAULT_STAGES),
        },
    )
    return job_dir


def write_status(job_dir: Path, payload: dict) -> None:
    with LOCK:
        status_path = job_dir / "status.json"
        tmp_path = job_dir / "status.json.tmp"
        tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp_path, status_path)


def read_status(job_id: str) -> dict:
    job_dir = settings.jobs_dir / job_id
    status_path = job_dir / "status.json"
    if not status_path.exists():
        return {"status": "missing", "logs": []}
    try:
        return json.loads(status_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fallback_path = job_dir / "status.initial.json"
        if fallback_path.exists():
            try:
                status = json.loads(fallback_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                status = {"logs": [], "stages": copy.deepcopy(DEFAULT_STAGES)}
        else:
            status = {"logs": [], "stages": copy.deepcopy(DEFAULT_STAGES)}
        status.update(
            {
                "status": "failed",
                "error": (
                    "Job status file is corrupted, likely because the Docker data "
                    "volume ran out of space while writing results. Free Docker "
                    "space and start a new job."
                ),
                "traceback": f"Could not parse {status_path}: {exc}",
            }
        )
        return status


def append_log(job_dir: Path, message: str) -> None:
    status = read_status(job_dir.name)
    status.setdefault("logs", []).append({"time": utc_now(), "message": message})
    write_status(job_dir, status)


def update_stage(
    job_dir: Path,
    stage: str,
    stage_status: str,
    progress: int,
    artifacts: dict | None = None,
) -> None:
    status = read_status(job_dir.name)
    stages = status.setdefault("stages", copy.deepcopy(DEFAULT_STAGES))
    current = stages.setdefault(
        stage,
        {"label": stage, "status": "queued", "progress": 0, "artifacts": {}},
    )
    current["status"] = stage_status
    current["progress"] = max(0, min(100, int(progress)))
    if artifacts:
        current.setdefault("artifacts", {}).update(artifacts)
    weights = {"input": 15, "fourdct": 35, "deepdrr": 35, "package": 15}
    overall = 0.0
    for key, weight in weights.items():
        item = stages.get(key, {})
        overall += (float(item.get("progress", 0)) / 100.0) * weight
    status["progress"] = int(round(overall))
    write_status(job_dir, status)


def run_job_background(
    job_dir: Path,
    ct_upload: Path,
    mask_upload: Path | None,
    params: JobParams,
) -> None:
    def target() -> None:
        status = read_status(job_dir.name)
        status.update({"status": "running", "progress": 0, "started_at": utc_now()})
        write_status(job_dir, status)
        try:
            zip_path = run_pipeline(
                job_dir,
                ct_upload,
                mask_upload,
                params,
                lambda msg: append_log(job_dir, msg),
                lambda stage, state, pct, artifacts=None: update_stage(
                    job_dir, stage, state, pct, artifacts
                ),
            )
            status = read_status(job_dir.name)
            status.update(
                {
                    "status": "done",
                    "progress": 100,
                    "finished_at": utc_now(),
                    "download": f"/jobs/{job_dir.name}/download",
                    "zip_path": str(zip_path),
                }
            )
            write_status(job_dir, status)
        except Exception as exc:
            status = read_status(job_dir.name)
            active_stage = None
            for key, stage in status.get("stages", {}).items():
                if stage.get("status") == "running":
                    active_stage = key
                    stage["status"] = "failed"
                    break
            status.update(
                {
                    "status": "failed",
                    "finished_at": utc_now(),
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "failed_stage": active_stage,
                }
            )
            write_status(job_dir, status)

    thread = threading.Thread(target=target, name=f"job-{job_dir.name}", daemon=True)
    thread.start()
