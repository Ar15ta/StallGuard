from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .actuation import wrap_signed_angle


@dataclass(frozen=True)
class StallLocationEstimate:
    angle_deg: float
    angular_speed_deg_s: float | None
    confidence: float
    source: str


class StallPositionTracker:
    """Estimate a rotating disturbance from successive probe alarm rising edges."""

    def __init__(self) -> None:
        self.configure(np.empty(0, dtype=float))

    def configure(self, probe_angles_deg: np.ndarray) -> None:
        angles = np.asarray(probe_angles_deg, dtype=float)
        if angles.ndim != 1 or np.any(~np.isfinite(angles)):
            raise ValueError("探针角度必须是一维有限数组。")
        self.probe_angles_deg = np.mod(angles, 360.0)
        self._pressure_center = np.zeros(angles.size, dtype=float)
        self._pressure_scale = np.ones(angles.size, dtype=float)
        self._has_pressure_baseline = False
        self.reset()

    def set_pressure_baseline(self, center: np.ndarray, scale: np.ndarray) -> None:
        center_values = np.asarray(center, dtype=float)
        scale_values = np.asarray(scale, dtype=float)
        expected = self.probe_angles_deg.shape
        if center_values.shape != expected or scale_values.shape != expected:
            raise ValueError("压力定位基线维度与探针数量不一致。")
        if np.any(~np.isfinite(center_values)) or np.any(~np.isfinite(scale_values)) or np.any(scale_values <= 0):
            raise ValueError("压力定位基线包含无效数值。")
        self._pressure_center = center_values.copy()
        self._pressure_scale = scale_values.copy()
        self._has_pressure_baseline = True

    def reset(self) -> None:
        self._previous_alarm = np.zeros(self.probe_angles_deg.size, dtype=bool)
        self._events: list[tuple[float, int, float]] = []
        self._anchor_time: float | None = None
        self._anchor_angle: float | None = None
        self._angular_speed: float | None = None
        self._pair_confidence = 0.0
        self._pressure_angle: float | None = None
        self._pressure_time: float | None = None
        self._pressure_speed: float | None = None

    def update(
        self,
        timestamp_s: float,
        risk_score: np.ndarray,
        probe_alarm: np.ndarray,
        system_alarm: bool,
        pressure_profile: np.ndarray | None = None,
    ) -> StallLocationEstimate | None:
        alarms = np.asarray(probe_alarm, dtype=bool)
        risk = np.asarray(risk_score, dtype=float)
        if alarms.shape != self.probe_angles_deg.shape or risk.shape != alarms.shape:
            raise ValueError("定位输入维度与探针角度不一致。")

        rising = np.flatnonzero(alarms & ~self._previous_alarm)
        for index in rising:
            self._events.append((float(timestamp_s), int(index), float(self.probe_angles_deg[index])))
        self._previous_alarm = alarms.copy()
        self._fit_latest_pair()

        if not system_alarm:
            if not np.any(alarms):
                self.reset()
            return None

        if self._angular_speed is not None and self._anchor_time is not None and self._anchor_angle is not None:
            age = max(0.0, float(timestamp_s) - self._anchor_time)
            predicted = (self._anchor_angle + self._angular_speed * age) % 360.0
            confidence = self._decayed_confidence(age)
            return StallLocationEstimate(float(predicted), self._angular_speed, confidence, "探针时差")

        pressure_location = self._pressure_location(timestamp_s, pressure_profile)
        if pressure_location is not None:
            return pressure_location

        fallback = self._risk_centroid(risk, alarms)
        if fallback is None:
            return None
        return StallLocationEstimate(fallback, None, 0.35, "风险周向重心")

    def _fit_latest_pair(self) -> None:
        if len(self._events) < 2:
            return
        second = self._events[-1]
        for first in reversed(self._events[:-1]):
            delta_t = second[0] - first[0]
            if delta_t <= np.finfo(float).eps or first[1] == second[1]:
                continue
            delta_angle = wrap_signed_angle(second[2] - first[2])
            if abs(delta_angle) <= np.finfo(float).eps:
                continue
            self._angular_speed = delta_angle / delta_t
            self._anchor_time = second[0]
            self._anchor_angle = second[2]
            expected_gap = self._typical_probe_gap()
            gap_error = abs(abs(delta_angle) - expected_gap) / max(expected_gap, 1.0)
            self._pair_confidence = float(np.clip(0.95 - 0.45 * gap_error, 0.45, 0.95))
            return

    def _typical_probe_gap(self) -> float:
        if self.probe_angles_deg.size < 2:
            return 180.0
        ordered = np.sort(self.probe_angles_deg)
        gaps = np.diff(np.r_[ordered, ordered[0] + 360.0])
        positive = gaps[gaps > np.finfo(float).eps]
        return float(np.median(positive)) if positive.size else 180.0

    def _decayed_confidence(self, age_s: float) -> float:
        if self._angular_speed is None:
            return 0.0
        crossing_time = self._typical_probe_gap() / max(abs(self._angular_speed), np.finfo(float).eps)
        grace = 2.0 * crossing_time
        excess = max(0.0, age_s - grace)
        return float(self._pair_confidence * np.exp(-excess / max(4.0 * crossing_time, 1e-6)))

    def _pressure_location(
        self, timestamp_s: float, pressure_profile: np.ndarray | None
    ) -> StallLocationEstimate | None:
        if pressure_profile is None:
            return None
        pressure = np.asarray(pressure_profile, dtype=float)
        if pressure.shape != self.probe_angles_deg.shape:
            raise ValueError("压力定位输入维度与探针角度不一致。")
        finite = np.isfinite(pressure)
        if np.count_nonzero(finite) < 2:
            return None
        if self._has_pressure_baseline:
            spatial = (pressure - self._pressure_center) / self._pressure_scale
        else:
            spatial = pressure.copy()
        spatial = spatial - np.nanmedian(spatial[finite])
        contrast = np.where(finite, np.abs(spatial), 0.0)
        noise_floor = float(np.nanmedian(contrast[finite]))
        weights = np.maximum(contrast - noise_floor, 0.0)
        if not np.any(weights > 0):
            weights = contrast
        total = float(np.sum(weights))
        if total <= np.finfo(float).eps:
            return None
        radians = np.deg2rad(self.probe_angles_deg)
        vector = np.sum(weights * np.exp(1j * radians))
        if abs(vector) <= np.finfo(float).eps:
            return None
        measured = float(np.rad2deg(np.angle(vector)) % 360.0)
        concentration = float(np.clip(abs(vector) / total, 0.0, 1.0))
        strength = float(np.nanmax(contrast[finite]))
        confidence = float(np.clip(0.25 + 0.45 * concentration + 0.1 * min(strength / 3.0, 1.0), 0.25, 0.8))

        now = float(timestamp_s)
        speed: float | None = None
        if self._pressure_angle is None or self._pressure_time is None or now <= self._pressure_time:
            angle = measured
        else:
            delta_t = now - self._pressure_time
            delta_angle = wrap_signed_angle(measured - self._pressure_angle)
            smoothing = 0.45
            angle = (self._pressure_angle + smoothing * delta_angle) % 360.0
            instantaneous_speed = wrap_signed_angle(angle - self._pressure_angle) / delta_t
            if self._pressure_speed is None:
                speed = instantaneous_speed
            else:
                speed = 0.3 * instantaneous_speed + 0.7 * self._pressure_speed
        self._pressure_angle = float(angle)
        self._pressure_time = now
        self._pressure_speed = speed
        return StallLocationEstimate(float(angle), speed, confidence, "压力周向差异")

    def _risk_centroid(self, risk: np.ndarray, alarms: np.ndarray) -> float | None:
        finite = np.isfinite(risk)
        if np.any(finite):
            baseline = float(np.nanmin(risk[finite]))
            weights = np.where(finite, np.maximum(risk - baseline, 0.0), 0.0)
        else:
            weights = np.zeros_like(risk)
        weights = weights + alarms.astype(float)
        if not np.any(weights > 0):
            return None
        radians = np.deg2rad(self.probe_angles_deg)
        vector = np.sum(weights * np.exp(1j * radians))
        if abs(vector) <= np.finfo(float).eps:
            active = np.flatnonzero(alarms)
            return None if active.size == 0 else float(self.probe_angles_deg[active[0]])
        return float(np.rad2deg(np.angle(vector)) % 360.0)
