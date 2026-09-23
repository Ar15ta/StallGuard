from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .analysis_service import _single_feature
from .warning_engine import fit_robust_baseline, score_with_baseline


@dataclass
class OnlineSnapshot:
    state: str
    sample_end: int
    time_s: float
    feature: np.ndarray
    risk_score: np.ndarray
    probe_alarm: np.ndarray
    system_alarm: bool
    calibration_progress: float


class OnlineWarningEngine:
    """Incremental, fixed-baseline warning engine used by TCP acquisition."""

    def __init__(
        self,
        probe_count: int,
        method: str,
        params: dict[str, Any],
        calibration_windows: int = 20,
        baseline_profile: dict[str, Any] | None = None,
    ) -> None:
        if probe_count < 1:
            raise ValueError("至少需要一个探针。")
        if calibration_windows < 10:
            raise ValueError("在线标定至少需要 10 个分析窗口。")
        self.probe_count = int(probe_count)
        self.method = method
        self.params = dict(params)
        self.window_size = int(params["window_size"])
        self.step_size = int(params["step_size"])
        self.calibration_windows = int(calibration_windows)
        self.minimum_probes = min(int(params.get("minimum_probes", 2)), probe_count)
        self.threshold = float(params.get("threshold", 3.0))
        self.trigger_windows = int(params.get("trigger_windows", 3))
        self.release_windows = int(params.get("release_windows", 5))
        self.release_ratio = float(params.get("release_ratio", 0.7))
        self.direction = str(params.get("direction", "both"))
        if self.window_size < 16 or self.step_size < 1:
            raise ValueError("在线窗口或步长无效。")
        self.reset()
        if baseline_profile is not None:
            self.load_calibration_profile(baseline_profile)

    @property
    def state(self) -> str:
        if not self._calibrated:
            return "CALIBRATING"
        return "ALARM" if self._system_alarm else "MONITORING"

    @property
    def calibrated(self) -> bool:
        return self._calibrated

    @property
    def first_alarm_time(self) -> float | None:
        return self._first_alarm_time

    def reset(self) -> None:
        self._buffer_times = np.empty(0, dtype=float)
        self._buffer_values = np.empty((0, self.probe_count), dtype=float)
        self._buffer_start = 0
        self._total_samples = 0
        self._next_window_end = self.window_size
        self._baseline_rows = {name: [] for name in self._methods_to_run()}
        self._centers: dict[str, np.ndarray] = {}
        self._scales: dict[str, np.ndarray] = {}
        self._calibrated = False
        self._high_counts = np.zeros(self.probe_count, dtype=int)
        self._low_counts = np.zeros(self.probe_count, dtype=int)
        self._probe_alarm = np.zeros(self.probe_count, dtype=bool)
        self._system_alarm = False
        self._first_alarm_time = None

    def _methods_to_run(self) -> tuple[str, ...]:
        return (self.method,)

    def update(self, time_s: np.ndarray, values: np.ndarray) -> list[OnlineSnapshot]:
        times = np.asarray(time_s, dtype=float)
        data = np.asarray(values, dtype=float)
        if data.ndim == 1:
            data = data[:, None]
        if times.ndim != 1 or data.shape != (times.size, self.probe_count):
            raise ValueError("在线数据的时间和探针数组形状不一致。")
        if times.size == 0:
            return []
        if np.any(np.diff(times) <= 0) or (self._buffer_times.size and times[0] <= self._buffer_times[-1]):
            raise ValueError("在线数据时间必须严格递增。")
        self._buffer_times = np.concatenate([self._buffer_times, times])
        self._buffer_values = np.vstack([self._buffer_values, data])
        self._total_samples += times.size
        snapshots: list[OnlineSnapshot] = []
        while self._next_window_end <= self._total_samples:
            sample_end = self._next_window_end
            sample_start = sample_end - self.window_size
            local_start = sample_start - self._buffer_start
            local_end = sample_end - self._buffer_start
            window_times = self._buffer_times[local_start:local_end]
            window_values = self._buffer_values[local_start:local_end]
            snapshot = self._process_window(sample_end, window_times, window_values)
            snapshots.append(snapshot)
            self._next_window_end += self.step_size
        keep_from = max(0, self._next_window_end - self.window_size - self._buffer_start)
        if keep_from:
            self._buffer_times = self._buffer_times[keep_from:]
            self._buffer_values = self._buffer_values[keep_from:, :]
            self._buffer_start += keep_from
        return snapshots

    def _process_window(
        self, sample_end: int, window_times: np.ndarray, window_values: np.ndarray
    ) -> OnlineSnapshot:
        features: dict[str, np.ndarray] = {}
        for method in self._methods_to_run():
            local_params = dict(self.params)
            local_params["window_size"] = self.window_size
            local_params["step_size"] = self.window_size
            feature, *_ = _single_feature(method, window_values, window_times, local_params)
            features[method] = np.asarray(feature[0], dtype=float)

        risk = np.full(self.probe_count, np.nan)
        if not self._calibrated:
            for method, feature in features.items():
                self._baseline_rows[method].append(feature.copy())
            count = len(self._baseline_rows[self.method])
            progress = min(count / self.calibration_windows, 1.0)
            if count >= self.calibration_windows:
                for method in self._methods_to_run():
                    baseline = np.vstack(self._baseline_rows[method])
                    self._centers[method], self._scales[method] = fit_robust_baseline(baseline)
                self._calibrated = True
        else:
            selected = features[self.method]
            risk = score_with_baseline(
                selected, self._centers[self.method], self._scales[self.method], self.direction
            )
            self._advance_alarm_state(risk)
            progress = 1.0

        display_feature = self._display_feature(features)
        time_value = float(window_times[-1])
        if self._system_alarm and self._first_alarm_time is None:
            self._first_alarm_time = time_value
        return OnlineSnapshot(
            state=self.state,
            sample_end=int(sample_end),
            time_s=time_value,
            feature=display_feature.copy(),
            risk_score=risk.copy(),
            probe_alarm=self._probe_alarm.copy(),
            system_alarm=bool(self._system_alarm),
            calibration_progress=float(progress),
        )

    def _display_feature(self, features: dict[str, np.ndarray]) -> np.ndarray:
        return features[self.method]

    def _advance_alarm_state(self, risk_score: np.ndarray) -> None:
        for probe, value in enumerate(risk_score):
            if not np.isfinite(value):
                self._high_counts[probe] = 0
                self._low_counts[probe] = 0
                continue
            if not self._probe_alarm[probe]:
                self._high_counts[probe] = self._high_counts[probe] + 1 if value >= self.threshold else 0
                self._low_counts[probe] = 0
                if self._high_counts[probe] >= self.trigger_windows:
                    self._probe_alarm[probe] = True
            else:
                self._low_counts[probe] = self._low_counts[probe] + 1 if value < self.threshold * self.release_ratio else 0
                self._high_counts[probe] = 0
                if self._low_counts[probe] >= self.release_windows:
                    self._probe_alarm[probe] = False
        self._system_alarm = bool(np.count_nonzero(self._probe_alarm) >= self.minimum_probes)

    def calibration_profile(self) -> dict[str, Any] | None:
        if not self._calibrated:
            return None
        return {
            "version": 1,
            "method": self.method,
            "probe_count": self.probe_count,
            "params": self.params,
            "centers": {name: values.tolist() for name, values in self._centers.items()},
            "scales": {name: values.tolist() for name, values in self._scales.items()},
            "calibration_windows": self.calibration_windows,
        }

    def load_calibration_profile(self, profile: dict[str, Any]) -> None:
        if int(profile.get("version", -1)) != 1:
            raise ValueError("不支持的基线文件版本。")
        if profile.get("method") != self.method:
            raise ValueError("基线分析类型与当前判据类型不一致。")
        if int(profile.get("probe_count", -1)) != self.probe_count:
            raise ValueError("基线探针数与当前数据不一致。")
        saved_params = profile.get("params")
        if not isinstance(saved_params, dict):
            raise ValueError("基线文件缺少算法参数。")
        for key in self._feature_parameter_names():
            if saved_params.get(key) != self.params.get(key):
                raise ValueError(f"基线参数 {key} 与当前配置不一致。")
        centers = profile.get("centers")
        scales = profile.get("scales")
        if not isinstance(centers, dict) or not isinstance(scales, dict):
            raise ValueError("基线文件缺少中心或尺度数据。")
        loaded_centers: dict[str, np.ndarray] = {}
        loaded_scales: dict[str, np.ndarray] = {}
        for method in self._methods_to_run():
            center = np.asarray(centers.get(method), dtype=float)
            scale = np.asarray(scales.get(method), dtype=float)
            if center.shape != (self.probe_count,) or scale.shape != (self.probe_count,):
                raise ValueError(f"基线 {method} 的探针维数无效。")
            if not np.all(np.isfinite(center)) or not np.all(np.isfinite(scale)) or np.any(scale <= 0):
                raise ValueError(f"基线 {method} 包含无效数值。")
            loaded_centers[method] = center.copy()
            loaded_scales[method] = scale.copy()
        self._centers = loaded_centers
        self._scales = loaded_scales
        self._calibrated = True

    def _feature_parameter_names(self) -> tuple[str, ...]:
        names = ["window_size"]
        methods = self._methods_to_run()
        if "wavelet" in methods:
            names.extend(["wavelet", "wavelet_level", "detail_level"])
        if "entropy" in methods:
            names.extend(["entropy_order", "entropy_delay"])
        if "autocorrelation" in methods:
            names.extend(["min_lag", "max_lag"])
        return tuple(names)
