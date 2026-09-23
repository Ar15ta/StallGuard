from __future__ import annotations

import numpy as np
from scipy import signal

from .common import standardize_window, window_indices, window_times


def normalized_acf(values: np.ndarray, max_lag: int) -> np.ndarray:
    series = np.asarray(values, dtype=float)
    correlations = signal.correlate(series, series, mode="full", method="fft")[series.size - 1 :]
    overlap = np.arange(series.size, 0, -1, dtype=float)
    correlations = correlations / overlap
    zero = correlations[0]
    if not np.isfinite(zero) or zero <= 0:
        return np.full(max_lag + 1, np.nan)
    return correlations[: max_lag + 1] / zero


def autocorrelation_features(
    values: np.ndarray,
    time_s: np.ndarray,
    window_size: int,
    step_size: int,
    min_lag: int = 2,
    max_lag: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    data = np.asarray(values, dtype=float)
    if data.ndim == 1:
        data = data[:, None]
    if not 1 <= int(min_lag) < int(max_lag) < int(window_size):
        raise ValueError("自相关参数必须满足 1 ≤ 最小 lag < 最大 lag < 窗口长度。")
    starts, ends = window_indices(data.shape[0], int(window_size), int(step_size))
    feature = np.full((starts.size, data.shape[1]), np.nan)
    peak_lag = np.full_like(feature, np.nan)
    last_acf = np.full((int(max_lag) + 1, data.shape[1]), np.nan)
    for row, (start, end) in enumerate(zip(starts, ends, strict=True)):
        for probe in range(data.shape[1]):
            normalized = standardize_window(data[start:end, probe])
            if normalized is None:
                continue
            acf = normalized_acf(normalized, int(max_lag))
            search = acf[int(min_lag) : int(max_lag) + 1]
            if np.any(np.isfinite(search)):
                local = int(np.nanargmax(search))
                feature[row, probe] = search[local]
                peak_lag[row, probe] = int(min_lag) + local
            if row == starts.size - 1:
                last_acf[:, probe] = acf
    auxiliary = {
        "peak_lag": peak_lag,
        "last_acf": last_acf,
        "min_lag": int(min_lag),
        "max_lag": int(max_lag),
    }
    return feature, window_times(time_s, starts, ends), starts, ends, auxiliary
