from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np


CHANNEL_COUNT = 16


def wrap_signed_angle(angle_deg: float) -> float:
    """Wrap an angle to [-180, 180)."""

    return float((float(angle_deg) + 180.0) % 360.0 - 180.0)


def circular_distance(first_deg: float, second_deg: float) -> float:
    return abs(wrap_signed_angle(float(first_deg) - float(second_deg)))


@dataclass(frozen=True)
class ActuatorChannelCommand:
    channel: int
    circumferential_angle_deg: float
    enabled: bool
    active: bool
    nozzle_offset_deg: float
    target_angle_deg: float | None
    timestamp_s: float
    confidence: float

    def as_dict(self) -> dict[str, float | int | bool | None]:
        return asdict(self)


@dataclass(frozen=True)
class ActuatorFrame:
    timestamp_s: float
    target_angle_deg: float | None
    confidence: float
    commands: tuple[ActuatorChannelCommand, ...]

    @property
    def active_channels(self) -> tuple[int, ...]:
        return tuple(command.channel for command in self.commands if command.active)

    def as_dict(self) -> dict[str, object]:
        return {
            "type": "actuator_frame",
            "timestamp_s": self.timestamp_s,
            "target_angle_deg": self.target_angle_deg,
            "confidence": self.confidence,
            "commands": [command.as_dict() for command in self.commands],
        }


ChannelOutput = Callable[[ActuatorChannelCommand], None]
FrameOutput = Callable[[ActuatorFrame], None]


class PulseScheduler:
    """Physical-time scheduler for repeated detect-aim-fire control pulses."""

    def __init__(self, duration_ms: float = 100.0) -> None:
        self._pulse_end_s: float | None = None
        self.set_duration_ms(duration_ms)

    @property
    def pulse_end_s(self) -> float | None:
        return self._pulse_end_s

    def set_duration_ms(self, duration_ms: float) -> None:
        if not np.isfinite(duration_ms) or duration_ms <= 0:
            raise ValueError("喷气持续时间必须大于0。")
        self.duration_s = float(duration_ms) / 1000.0

    def pulse_due(self, timestamp_s: float) -> bool:
        return self._pulse_end_s is None or float(timestamp_s) >= self._pulse_end_s

    def start(self, timestamp_s: float) -> float:
        self._pulse_end_s = float(timestamp_s) + self.duration_s
        return self._pulse_end_s

    def reset(self) -> None:
        self._pulse_end_s = None


class ActuatorController:
    """Sixteen-channel active-control command router.

    Hardware integration deliberately lives behind callbacks. Register one callback per
    physical channel, or one frame callback for a PLC/UDP adapter. Channel numbers are
    one-based in the public interface.
    """

    def __init__(self, channel_count: int = CHANNEL_COUNT) -> None:
        if channel_count < 2:
            raise ValueError("动作机构数量至少为2。")
        self.channel_count = int(channel_count)
        self.channel_angles_deg = np.arange(self.channel_count, dtype=float) * (360.0 / self.channel_count)
        # Safe commissioning default: no physical output until the operator enables it.
        self.enabled = np.zeros(self.channel_count, dtype=bool)
        self.manual_offsets_deg = np.zeros(self.channel_count, dtype=float)
        self.activation_counts = np.zeros(self.channel_count, dtype=int)
        self._channel_outputs: list[ChannelOutput | None] = [None] * self.channel_count
        self._frame_outputs: list[FrameOutput] = []
        self.history: list[ActuatorFrame] = []
        self.last_frame = self._build_frame(0.0, None, 0.0, ())

    def set_enabled(self, index: int, enabled: bool) -> None:
        self._validate_index(index)
        self.enabled[index] = bool(enabled)

    def set_manual_offset(self, index: int, angle_deg: float) -> None:
        self._validate_index(index)
        if not np.isfinite(angle_deg):
            raise ValueError("喷口角度必须为有限数。")
        self.manual_offsets_deg[index] = wrap_signed_angle(angle_deg)

    def set_all_enabled(self, enabled: bool) -> None:
        self.enabled[:] = bool(enabled)

    def reset_activation_counts(self) -> None:
        self.activation_counts[:] = 0

    def register_channel_output(self, channel: int, callback: ChannelOutput | None) -> None:
        """Bind a physical output callback to channel 1..N."""

        index = int(channel) - 1
        self._validate_index(index)
        self._channel_outputs[index] = callback

    def register_frame_output(self, callback: FrameOutput) -> None:
        self._frame_outputs.append(callback)

    def command_target(self, timestamp_s: float, target_angle_deg: float, confidence: float) -> ActuatorFrame:
        if not np.isfinite(timestamp_s) or not np.isfinite(target_angle_deg):
            raise ValueError("动作指令的时间与目标角度必须为有限数。")
        available = np.flatnonzero(self.enabled)
        if available.size < 2:
            return self.stop(timestamp_s)
        target = float(target_angle_deg) % 360.0
        nearest = sorted(
            available.tolist(),
            key=lambda index: (circular_distance(self.channel_angles_deg[index], target), index),
        )[:2]
        frame = self._build_frame(timestamp_s, target, confidence, tuple(nearest))
        previous = self.last_frame
        selection_changed = previous.active_channels != frame.active_channels
        target_changed = (
            previous.target_angle_deg is None
            or circular_distance(previous.target_angle_deg, target) >= 1.0
        )
        refresh_due = float(timestamp_s) - previous.timestamp_s >= 0.02
        if selection_changed or target_changed or refresh_due:
            self._emit(frame)
        else:
            self.last_frame = frame
        return frame

    def stop(self, timestamp_s: float) -> ActuatorFrame:
        frame = self._build_frame(float(timestamp_s), None, 0.0, ())
        # Avoid flooding hardware adapters with identical stop frames.
        if self.last_frame.active_channels or not self.history:
            self._emit(frame)
        else:
            self.last_frame = frame
        return frame

    def _build_frame(
        self,
        timestamp_s: float,
        target_angle_deg: float | None,
        confidence: float,
        active_indices: tuple[int, ...],
    ) -> ActuatorFrame:
        active = set(active_indices)
        commands: list[ActuatorChannelCommand] = []
        for index, position in enumerate(self.channel_angles_deg):
            nozzle_offset = self.manual_offsets_deg[index]
            if index in active and target_angle_deg is not None:
                nozzle_offset = self._aim_offset(position, target_angle_deg)
            commands.append(
                ActuatorChannelCommand(
                    channel=index + 1,
                    circumferential_angle_deg=float(position),
                    enabled=bool(self.enabled[index]),
                    active=index in active,
                    nozzle_offset_deg=float(nozzle_offset),
                    target_angle_deg=None if target_angle_deg is None else float(target_angle_deg),
                    timestamp_s=float(timestamp_s),
                    confidence=float(np.clip(confidence, 0.0, 1.0)),
                )
            )
        return ActuatorFrame(
            timestamp_s=float(timestamp_s),
            target_angle_deg=None if target_angle_deg is None else float(target_angle_deg),
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            commands=tuple(commands),
        )

    @staticmethod
    def _aim_offset(actuator_angle_deg: float, target_angle_deg: float, target_radius: float = 0.42) -> float:
        actuator_angle = np.deg2rad(actuator_angle_deg)
        target_angle = np.deg2rad(target_angle_deg)
        actuator = np.array([np.cos(actuator_angle), np.sin(actuator_angle)])
        target = target_radius * np.array([np.cos(target_angle), np.sin(target_angle)])
        direction = target - actuator
        aim_angle = np.rad2deg(np.arctan2(direction[1], direction[0]))
        inward_angle = actuator_angle_deg + 180.0
        return wrap_signed_angle(float(aim_angle) - inward_angle)

    def _emit(self, frame: ActuatorFrame) -> None:
        previous_active = {channel - 1 for channel in self.last_frame.active_channels}
        current_active = {channel - 1 for channel in frame.active_channels}
        for index in current_active - previous_active:
            self.activation_counts[index] += 1
        self.last_frame = frame
        self.history.append(frame)
        if len(self.history) > 5000:
            del self.history[:1000]
        for command, callback in zip(frame.commands, self._channel_outputs, strict=True):
            if callback is not None:
                callback(command)
        for callback in tuple(self._frame_outputs):
            callback(frame)

    def _validate_index(self, index: int) -> None:
        if not 0 <= int(index) < self.channel_count:
            raise IndexError("动作机构通道超出范围。")
