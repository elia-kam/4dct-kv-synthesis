from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    dynagan_dir: Path
    dynagan_python: str
    deepdrr_mode: str
    deepdrr_sif: Path | None
    output_retention_hours: int

    @classmethod
    def from_env(cls) -> "Settings":
        data_dir = Path(os.environ.get("APP_DATA_DIR", "/data")).resolve()
        deepdrr_sif_raw = os.environ.get("DEEPDRR_SIF", "").strip()
        dynagan_python = resolve_executable(os.environ.get("DYNAGAN_PYTHON", sys.executable))
        return cls(
            data_dir=data_dir,
            dynagan_dir=Path(os.environ.get("DYNAGAN_DIR", "/opt/Dynagan")).resolve(),
            dynagan_python=dynagan_python,
            deepdrr_mode=os.environ.get("DEEPDRR_MODE", "python").strip().lower(),
            deepdrr_sif=Path(deepdrr_sif_raw).resolve() if deepdrr_sif_raw else None,
            output_retention_hours=int(os.environ.get("OUTPUT_RETENTION_HOURS", "24")),
        )

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"


def resolve_executable(value: str) -> str:
    value = value.strip()
    if not value:
        return sys.executable
    if "/" in value:
        return str(Path(value).expanduser().resolve())
    return shutil.which(value) or value


settings = Settings.from_env()
