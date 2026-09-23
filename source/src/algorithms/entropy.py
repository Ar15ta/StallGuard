from __future__ import annotations

import math

import numpy as np

from .common import standardize_window, window_indices, window_times


def normalized_permutation_entropy(values: np.ndarray, order: int = 5, delay: int = 1) -> float:
    series = np.asarray(values, dtype=float)
    if order < 3 or order > 7:
        raise ValueError("排列熵嵌入维数必须在 3～7 之间。")
    if delay < 1:
        raise ValueError("排列熵延迟必须大于等于 1。")
    pattern_count = series.size - (order - 1) * delay
    if pattern_count < max(10, math.factorial(order) // 2):
        return float("nan")
    offsets = np.arange(order) * delay
    embedded = series[np.arange(pattern_count)[:, None] + offsets[None, :]]
    patterns = np.argsort(embedded, axis=1, kind="stable")
    _, counts = np.unique(patterns, axis=0, return_counts=True)
    probabilities = counts.astype(float) / counts.sum()
    entropy = -np.sum(probabilities * np.log(probabilities))
    return float(entropy / np.log(math.factorial(order)))


def permutation_entropy_features(
    values: np.ndarray,
    time_s: np.ndarray,
    window_size: int,
    step_size: int,
    order: int = 5,
    delay: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    data = np.asarray(values, dtype=float)
    if data.ndim == 1:
        data = data[:, None]
    starts, ends = window_indices(data.shape[0], int(window_size), int(step_size))
    feature = np.full((starts.size, data.shape[1]), np.nan)
    for row, (start, end) in enumerate(zip(starts, ends, strict=True)):
        for probe in range(data.shape[1]):
            normalized = standardize_window(data[start:end, probe])
            if normalized is not None:
                feature[row, probe] = normalized_permutation_entropy(normalized, int(order), int(delay))
    return (
        feature,
        window_times(time_s, starts, ends),
        starts,
        ends,
        {"order": int(order), "delay": int(delay)},
    )
