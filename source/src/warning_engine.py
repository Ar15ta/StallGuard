from __future__ import annotations

import numpy as np


def fit_robust_baseline(feature: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    baseline = np.asarray(feature, dtype=float)
    if baseline.ndim == 1:
        baseline = baseline[:, None]
    centers = np.nanmedian(baseline, axis=0)
    mad = np.nanmedian(np.abs(baseline - centers), axis=0)
    scales = 1.4826 * mad
    std = np.nanstd(baseline, axis=0)
    scales = np.where(np.isfinite(scales) & (scales > 1e-12), scales, std)
    magnitude = np.maximum(np.abs(centers), 1.0)
    scales = np.where(np.isfinite(scales) & (scales > magnitude * 1e-12), scales, magnitude * 1e-12)
    return centers, scales


def score_with_baseline(
    feature: np.ndarray, centers: np.ndarray, scales: np.ndarray, direction: str = "both"
) -> np.ndarray:
    data = np.asarray(feature, dtype=float)
    if direction not in {"high", "low", "both"}:
        raise ValueError("判据方向必须为 high、low 或 both。")
    if direction == "high":
        risk = (data - centers) / scales
    elif direction == "low":
        risk = (centers - data) / scales
    else:
        risk = np.abs(data - centers) / scales
    return np.where(np.isfinite(data), risk, np.nan)


def robust_risk_scores(
    feature: np.ndarray,
    baseline_fraction: float = 0.2,
    direction: str = "high",
    minimum_baseline_windows: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.asarray(feature, dtype=float)
    if data.ndim == 1:
        data = data[:, None]
    if direction not in {"high", "low", "both"}:
        raise ValueError("判据方向必须为 high、low 或 both。")
    baseline_count = max(int(np.ceil(data.shape[0] * baseline_fraction)), minimum_baseline_windows)
    baseline_count = min(baseline_count, data.shape[0])
    centers, scales = fit_robust_baseline(data[:baseline_count])
    risk = score_with_baseline(data, centers, scales, direction)
    return risk, centers, scales


def apply_hysteresis(
    risk_score: np.ndarray,
    threshold: float = 3.0,
    trigger_windows: int = 3,
    release_windows: int = 5,
    release_ratio: float = 0.7,
) -> np.ndarray:
    risk = np.asarray(risk_score, dtype=float)
    if risk.ndim == 1:
        risk = risk[:, None]
    if threshold <= 0 or trigger_windows < 1 or release_windows < 1:
        raise ValueError("预警阈值和持续窗口参数必须大于 0。")
    alarm = np.zeros(risk.shape, dtype=bool)
    for probe in range(risk.shape[1]):
        active = False
        high_count = low_count = 0
        for row, value in enumerate(risk[:, probe]):
            if not np.isfinite(value):
                high_count = low_count = 0
                alarm[row, probe] = active
                continue
            if not active:
                high_count = high_count + 1 if value >= threshold else 0
                if high_count >= trigger_windows:
                    active = True
                    low_count = 0
            else:
                low_count = low_count + 1 if value < threshold * release_ratio else 0
                if low_count >= release_windows:
                    active = False
                    high_count = 0
            alarm[row, probe] = active
    return alarm


def aggregate_probe_alarms(probe_alarm: np.ndarray, minimum_probes: int = 1) -> np.ndarray:
    alarms = np.asarray(probe_alarm, dtype=bool)
    if alarms.ndim != 2:
        raise ValueError("探针预警结果必须为二维。")
    if not 1 <= int(minimum_probes) <= alarms.shape[1]:
        raise ValueError("K-of-N 中的 K 必须位于 1 和探针数之间。")
    return np.count_nonzero(alarms, axis=1) >= int(minimum_probes)
