from __future__ import annotations

import numpy as np


def window_indices(sample_count: int, window_size: int, step_size: int) -> tuple[np.ndarray, np.ndarray]:
    if window_size < 16:
        raise ValueError("分析窗口至少需要 16 点。")
    if step_size < 1:
        raise ValueError("窗口步长至少需要 1 点。")
    if sample_count < window_size:
        raise ValueError(f"有效数据不足：分析需要 {window_size} 点。")
    starts = np.arange(0, sample_count - window_size + 1, step_size, dtype=int)
    return starts, starts + window_size


def window_times(time_s: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    centers = starts + (ends - starts - 1) // 2
    return np.asarray(time_s, dtype=float)[centers]


def standardize_window(window: np.ndarray) -> np.ndarray | None:
    finite = np.isfinite(window)
    if finite.sum() < max(8, int(window.size * 0.8)):
        return None
    if not np.all(finite):
        x = np.arange(window.size)
        window = np.interp(x, x[finite], window[finite])
    centered = window - np.mean(window)
    scale = np.std(centered)
    if not np.isfinite(scale) or scale <= 1e-14:
        return None
    return centered / scale
