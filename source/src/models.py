from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DataSourceSettings:
    output_interval_s: float = 5.76928006151677e-6
    step_origin: float = 57103.0
    time_origin_s: float = 0.0
    first_row_zero: bool = True
    start_angle_deg: float = 0.0
    angle_direction: int = 1

    def validate(self) -> None:
        if not np.isfinite(self.output_interval_s) or self.output_interval_s <= 0:
            raise ValueError("探针响应输出间隔必须是大于 0 的有限数。")
        if self.angle_direction not in (-1, 1):
            raise ValueError("周向排列方向必须为递增或递减。")


@dataclass
class ProbeDataset:
    path: Path
    file_hash: str
    steps: np.ndarray
    time_s: np.ndarray
    probe_names: list[str]
    original_headers: list[str]
    angles_deg: np.ndarray
    values: np.ndarray
    unit: str
    settings: DataSourceSettings
    quality: dict[str, Any] = field(default_factory=dict)

    @property
    def sample_rate_hz(self) -> float:
        return 1.0 / self.settings.output_interval_s

    @property
    def probe_count(self) -> int:
        return len(self.probe_names)

    @property
    def sample_count(self) -> int:
        return int(self.steps.size)


@dataclass
class SystemAnalysisResult:
    method: str
    window_times: np.ndarray
    window_starts: np.ndarray
    window_ends: np.ndarray
    feature: np.ndarray
    risk_score: np.ndarray
    probe_alarm: np.ndarray
    system_alarm: np.ndarray
    valid: np.ndarray
    threshold: float
    centers: np.ndarray
    scales: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    auxiliary: dict[str, Any] = field(default_factory=dict)

    @property
    def first_alarm_time(self) -> float | None:
        indices = np.flatnonzero(self.system_alarm)
        return None if indices.size == 0 else float(self.window_times[indices[0]])

    @property
    def peak_risk(self) -> float:
        finite = self.risk_score[np.isfinite(self.risk_score)]
        return float(np.max(finite)) if finite.size else float("nan")
