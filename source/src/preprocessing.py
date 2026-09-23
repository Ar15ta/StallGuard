from __future__ import annotations

import numpy as np
from scipy import signal


def _as_2d(values: np.ndarray) -> np.ndarray:
    data = np.asarray(values, dtype=float)
    return data[:, None] if data.ndim == 1 else data


def interpolate_missing(values: np.ndarray) -> np.ndarray:
    result = _as_2d(values).copy()
    x = np.arange(result.shape[0])
    for column in range(result.shape[1]):
        valid = np.isfinite(result[:, column])
        if valid.sum() >= 2:
            result[:, column] = np.interp(x, x[valid], result[valid, column])
    return result


def _design_sos(
    sample_rate_hz: float,
    mode: str,
    low_cutoff_hz: float | None,
    high_cutoff_hz: float | None,
    order: int,
) -> np.ndarray:
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("采样率必须大于 0。")
    if not 2 <= int(order) <= 8:
        raise ValueError("滤波器阶数必须在 2～8 之间。")
    nyquist = sample_rate_hz / 2.0
    if mode == "lowpass":
        if low_cutoff_hz is None or not 0 < low_cutoff_hz < nyquist:
            raise ValueError("低通截止频率必须在 0 与 Nyquist 频率之间。")
        cutoff: float | list[float] = float(low_cutoff_hz)
        btype = "lowpass"
    elif mode == "bandpass":
        if low_cutoff_hz is None or high_cutoff_hz is None or not 0 < low_cutoff_hz < high_cutoff_hz < nyquist:
            raise ValueError("带通频率必须满足 0 < 低截止 < 高截止 < Nyquist。")
        cutoff = [float(low_cutoff_hz), float(high_cutoff_hz)]
        btype = "bandpass"
    else:
        raise ValueError(f"未知滤波模式：{mode}")
    return signal.butter(int(order), cutoff, btype=btype, fs=sample_rate_hz, output="sos")


def apply_filter(
    values: np.ndarray,
    sample_rate_hz: float,
    mode: str = "off",
    low_cutoff_hz: float | None = None,
    high_cutoff_hz: float | None = None,
    order: int = 4,
    detrend: bool = False,
) -> np.ndarray:
    data = interpolate_missing(values)
    if detrend:
        data = signal.detrend(data, axis=0, type="linear")
    if mode == "off":
        return data
    sos = _design_sos(sample_rate_hz, mode, low_cutoff_hz, high_cutoff_hz, order)
    try:
        return signal.sosfiltfilt(sos, data, axis=0)
    except ValueError as exc:
        raise ValueError("数据过短，无法执行零相位滤波。") from exc


def _rolling_stats(data: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    count = np.isfinite(data).astype(float)
    clean = np.where(np.isfinite(data), data, 0.0)
    zeros = np.zeros((1, data.shape[1]))
    c_count = np.vstack([zeros, np.cumsum(count, axis=0)])
    c_sum = np.vstack([zeros, np.cumsum(clean, axis=0)])
    c_sum2 = np.vstack([zeros, np.cumsum(clean * clean, axis=0)])
    n = c_count[ends] - c_count[starts]
    sums = c_sum[ends] - c_sum[starts]
    sums2 = c_sum2[ends] - c_sum2[starts]
    with np.errstate(invalid="ignore", divide="ignore"):
        means = sums / n
        variances = sums2 / n - means * means
    rms = np.sqrt(np.maximum(variances, 0.0))
    rms[n < 2] = np.nan
    return rms


def rolling_rms(values: np.ndarray, half_window: int = 50) -> np.ndarray:
    data = _as_2d(values)
    if half_window < 1:
        raise ValueError("RMS 半窗口必须至少为 1 点。")
    indices = np.arange(data.shape[0])
    starts = np.maximum(0, indices - int(half_window))
    ends = np.minimum(data.shape[0], indices + int(half_window) + 1)
    return _rolling_stats(data, starts, ends)


def causal_rolling_rms(values: np.ndarray, window_size: int = 101) -> np.ndarray:
    data = _as_2d(values)
    if window_size < 2:
        raise ValueError("在线 RMS 窗口必须至少为 2 点。")
    ends = np.arange(1, data.shape[0] + 1)
    starts = np.maximum(0, ends - int(window_size))
    return _rolling_stats(data, starts, ends)


class OnlineSignalFilter:
    def __init__(
        self,
        probe_count: int,
        sample_rate_hz: float,
        mode: str = "off",
        low_cutoff_hz: float | None = None,
        high_cutoff_hz: float | None = None,
        order: int = 4,
    ) -> None:
        self.probe_count = int(probe_count)
        self.sample_rate_hz = float(sample_rate_hz)
        self.mode = mode
        self.low_cutoff_hz = low_cutoff_hz
        self.high_cutoff_hz = high_cutoff_hz
        self.order = int(order)
        self.sos = None if mode == "off" else _design_sos(sample_rate_hz, mode, low_cutoff_hz, high_cutoff_hz, order)
        self.reset()

    def reset(self) -> None:
        if self.sos is None:
            self.zi = None
        else:
            self.zi = np.zeros((self.sos.shape[0], 2, self.probe_count), dtype=float)

    def process(self, values: np.ndarray) -> np.ndarray:
        data = interpolate_missing(values)
        if data.shape[1] != self.probe_count:
            raise ValueError("在线滤波数据的探针维数不一致。")
        if self.sos is None:
            return data
        filtered, self.zi = signal.sosfilt(self.sos, data, axis=0, zi=self.zi)
        return filtered


def normalize_rms(rms: np.ndarray, pressure: np.ndarray) -> tuple[np.ndarray, str]:
    rms_data = np.asarray(rms, dtype=float)
    pressure_data = np.asarray(pressure, dtype=float)
    means = np.abs(np.nanmean(pressure_data, axis=0))
    finite_scale = np.nanmax(np.abs(pressure_data), axis=0)
    safe = np.isfinite(means) & (means > np.maximum(finite_scale * 1e-12, 1e-12))
    if np.all(safe):
        return rms_data / means, "mean_pressure"
    peaks = np.nanmax(rms_data, axis=0)
    peaks = np.where(np.isfinite(peaks) & (peaks > 0), peaks, 1.0)
    return rms_data / peaks, "probe_peak_fallback"


def circumferential_amplitude(
    normalized_rms: np.ndarray, angles_deg: np.ndarray, amplitude_scale_deg: float
) -> tuple[np.ndarray, float]:
    values = np.asarray(normalized_rms, dtype=float)
    angles = np.asarray(angles_deg, dtype=float)
    if values.ndim != 2 or values.shape[1] != angles.size:
        raise ValueError("RMS 数据列数必须与探针角度数量一致。")
    finite = values[np.isfinite(values)]
    global_max = float(np.max(finite)) if finite.size else 0.0
    if global_max <= 0:
        return np.broadcast_to(angles, values.shape).copy(), global_max
    return angles[None, :] + values / global_max * float(amplitude_scale_deg), global_max


def window_mask_to_samples(
    sample_count: int, starts: np.ndarray, ends: np.ndarray, window_mask: np.ndarray
) -> np.ndarray:
    mask = np.asarray(window_mask, dtype=bool)
    if mask.ndim == 1:
        mask = mask[:, None]
    output = np.zeros((sample_count, mask.shape[1]), dtype=bool)
    for row, (start, end) in enumerate(zip(starts, ends, strict=True)):
        output[int(start) : int(end), :] |= mask[row]
    return output
