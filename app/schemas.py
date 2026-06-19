from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DynaganParams:
    patient_id: str
    alpha_min: float
    alpha_max: float
    phase_count: int
    gpu_id: str

    @property
    def alpha_steps(self) -> int:
        return max(1, self.phase_count - 1)


@dataclass(frozen=True)
class DeepDRRParams:
    views: tuple[str, ...]
    custom_views: tuple[dict, ...]
    sensor_width: int
    sensor_height: int
    pixel_size: float
    source_to_detector_distance: float
    source_to_isocenter_vertical_distance: float
    preview_size: int
    include_annotations: bool


@dataclass(frozen=True)
class JobParams:
    dynagan: DynaganParams
    deepdrr: DeepDRRParams
    run_4dct: bool
    run_deepdrr: bool
    has_annotations: bool
