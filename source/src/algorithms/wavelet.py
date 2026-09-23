from __future__ import annotations

import numpy as np
import pywt

from .common import standardize_window, window_indices, window_times


def wavelet_energy_features(
    values: np.ndarray,
    time_s: np.ndarray,
    window_size: int,
    step_size: int,
    wavelet: str = "db4",
    level: int = 4,
    detail_level: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Return the selected DWT detail-energy ratio for every window and probe."""
    data = np.asarray(values, dtype=float)
    if data.ndim == 1:
        data = data[:, None]
    starts, ends = window_indices(data.shape[0], int(window_size), int(step_size))
    wavelet_obj = pywt.Wavelet(wavelet)
    max_level = pywt.dwt_max_level(window_size, wavelet_obj.dec_len)
    actual_level = min(int(level), int(max_level))
    if actual_level < 1:
        raise ValueError("窗口过短，无法进行所选小波分解。")
    if not 1 <= int(detail_level) <= actual_level:
        raise ValueError(f"细节尺度必须在 1～{actual_level} 之间。")
    feature = np.full((starts.size, data.shape[1]), np.nan)
    energies = np.full((starts.size, data.shape[1], actual_level + 1), np.nan)
    for row, (start, end) in enumerate(zip(starts, ends, strict=True)):
        for probe in range(data.shape[1]):
            normalized = standardize_window(data[start:end, probe])
            if normalized is None:
                continue
            coeffs = pywt.wavedec(normalized, wavelet_obj, level=actual_level, mode="symmetric")
            coefficient_energy = np.asarray([np.sum(coeff * coeff) for coeff in coeffs], dtype=float)
            total = float(np.sum(coefficient_energy))
            if total <= 0:
                continue
            energies[row, probe, :] = coefficient_energy / total
            feature[row, probe] = coefficient_energy[-int(detail_level)] / total
    auxiliary = {
        "energy_ratios": energies,
        "actual_level": actual_level,
        "wavelet": wavelet,
        "detail_level": int(detail_level),
    }
    return feature, window_times(time_s, starts, ends), starts, ends, auxiliary
