from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import numpy as np

from src.algorithms.autocorrelation import autocorrelation_features
from src.algorithms.entropy import normalized_permutation_entropy
from src.actuation import ActuatorController, PulseScheduler
from src.io_csv import generate_angles, load_cfd_csv
from src.models import DataSourceSettings
from src.online_engine import OnlineWarningEngine
from src.preprocessing import causal_rolling_rms
from src.stall_tracking import StallPositionTracker
from src.warning_engine import aggregate_probe_alarms, apply_hysteresis


ROOT = Path(__file__).resolve().parents[2]


class CoreTests(unittest.TestCase):
    def test_csv_contract_and_time_origin(self) -> None:
        dataset = load_cfd_csv(ROOT / "release_v022" / "pressure_u1_initialization.csv", DataSourceSettings())
        self.assertEqual(dataset.probe_count, 5)
        self.assertEqual(dataset.probe_names[0], "Monitor Point 1")
        self.assertEqual(dataset.unit, "Pa")
        self.assertEqual(dataset.time_s[0], 0.0)
        self.assertTrue(np.allclose(dataset.angles_deg, [0, 72, 144, 216, 288]))

    def test_angle_direction(self) -> None:
        self.assertTrue(np.allclose(generate_angles(5, 10, -1), [10, 298, 226, 154, 82]))

    def test_csv_probe_count_is_dynamic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "many_probes.csv"
            headers = ["step"] + [f"probe_{index + 1} [Pa]" for index in range(37)]
            rows = [",".join(headers)]
            for step in range(4):
                rows.append(",".join([str(step)] + [str(step + index / 10) for index in range(37)]))
            path.write_text("\n".join(rows), encoding="utf-8")
            dataset = load_cfd_csv(path, DataSourceSettings(output_interval_s=0.01))
            self.assertEqual(dataset.probe_count, 37)
            self.assertEqual(dataset.values.shape, (4, 37))
            self.assertTrue(np.isclose(dataset.angles_deg[1], 360.0 / 37.0))

    def test_entropy_bounds(self) -> None:
        ordered = np.arange(1000, dtype=float)
        random = np.random.default_rng(4).normal(size=1000)
        self.assertLess(normalized_permutation_entropy(ordered, 4), normalized_permutation_entropy(random, 4))

    def test_autocorrelation_shape(self) -> None:
        time_s = np.arange(256) * 0.001
        values = np.column_stack([np.sin(np.arange(256) * 0.2), np.cos(np.arange(256) * 0.1)])
        feature, times, starts, ends, auxiliary = autocorrelation_features(values, time_s, 64, 16, 2, 20)
        self.assertEqual(feature.shape, (13, 2))
        self.assertEqual(times.shape, (13,))
        self.assertEqual(auxiliary["last_acf"].shape, (21, 2))
        self.assertTrue(np.all(ends - starts == 64))

    def test_hysteresis_and_k_of_n(self) -> None:
        risk = np.array([[0, 0], [4, 4], [4, 4], [1, 1], [1, 1], [1, 1]], dtype=float)
        probes = apply_hysteresis(risk, threshold=3, trigger_windows=2, release_windows=3)
        system = aggregate_probe_alarms(probes, 2)
        self.assertTrue(system[2])
        self.assertFalse(system[-1])

    def test_online_chunking_is_invariant(self) -> None:
        rng = np.random.default_rng(8)
        values = rng.normal(size=(240, 3))
        times = np.arange(240) * 0.01
        params = {
            "window_size": 32, "step_size": 8, "minimum_probes": 2,
            "threshold": 3.0, "trigger_windows": 2, "release_windows": 3,
            "release_ratio": 0.7, "direction": "high", "min_lag": 2, "max_lag": 10,
        }
        whole = OnlineWarningEngine(3, "autocorrelation", params, 10)
        chunked = OnlineWarningEngine(3, "autocorrelation", params, 10)
        expected = whole.update(times, values)
        actual = []
        for left, right in ((0, 17), (17, 61), (61, 122), (122, 183), (183, 240)):
            actual.extend(chunked.update(times[left:right], values[left:right]))
        self.assertEqual(len(expected), len(actual))
        for first, second in zip(expected, actual, strict=True):
            self.assertEqual(first.state, second.state)
            self.assertEqual(first.sample_end, second.sample_end)
            self.assertTrue(np.allclose(first.feature, second.feature, equal_nan=True))
            self.assertTrue(np.allclose(first.risk_score, second.risk_score, equal_nan=True))

    def test_causal_rms(self) -> None:
        result = causal_rolling_rms(np.array([1.0, 2.0, 3.0, 4.0]), 3)[:, 0]
        self.assertTrue(np.isnan(result[0]))
        self.assertTrue(np.isclose(result[-1], np.std([2.0, 3.0, 4.0])))

    def test_stall_location_from_successive_probe_alarms(self) -> None:
        tracker = StallPositionTracker()
        tracker.configure(np.array([0.0, 72.0, 144.0, 216.0, 288.0]))
        tracker.update(1.0, np.array([4.0, 1.0, 0.0, 0.0, 0.0]), np.array([1, 0, 0, 0, 0]), False)
        estimate = tracker.update(
            1.2,
            np.array([5.0, 4.0, 0.0, 0.0, 0.0]),
            np.array([1, 1, 0, 0, 0]),
            True,
        )
        self.assertIsNotNone(estimate)
        self.assertTrue(np.isclose(estimate.angular_speed_deg_s, 360.0))
        self.assertTrue(np.isclose(estimate.angle_deg, 72.0))
        advanced = tracker.update(
            1.3,
            np.array([5.0, 4.0, 0.0, 0.0, 0.0]),
            np.array([1, 1, 0, 0, 0]),
            True,
        )
        self.assertIsNotNone(advanced)
        self.assertTrue(np.isclose(advanced.angle_deg, 108.0))

    def test_stall_location_falls_back_when_alarms_are_simultaneous(self) -> None:
        tracker = StallPositionTracker()
        tracker.configure(np.array([0.0, 72.0, 144.0, 216.0, 288.0]))
        estimate = tracker.update(
            2.0,
            np.array([5.0, 5.0, 0.0, 0.0, 0.0]),
            np.array([1, 1, 0, 0, 0]),
            True,
        )
        self.assertIsNotNone(estimate)
        self.assertIsNone(estimate.angular_speed_deg_s)
        self.assertEqual(estimate.source, "风险周向重心")

    def test_pressure_contrast_tracks_location_without_alarm_edge_timing(self) -> None:
        tracker = StallPositionTracker()
        tracker.configure(np.array([0.0, 90.0, 180.0, 270.0]))
        tracker.set_pressure_baseline(np.zeros(4), np.ones(4))
        alarms = np.ones(4, dtype=bool)
        first = tracker.update(
            1.0,
            np.ones(4),
            alarms,
            True,
            pressure_profile=np.array([-4.0, 0.0, 0.0, 0.0]),
        )
        second = tracker.update(
            1.01,
            np.ones(4),
            alarms,
            True,
            pressure_profile=np.array([0.0, -4.0, 0.0, 0.0]),
        )
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertEqual(first.source, "压力周向差异")
        self.assertEqual(second.source, "压力周向差异")
        self.assertFalse(np.isclose(first.angle_deg, second.angle_deg))
        self.assertIsNotNone(second.angular_speed_deg_s)

    def test_actuator_controller_selects_two_nearest_enabled_channels(self) -> None:
        controller = ActuatorController(16)
        controller.set_all_enabled(True)
        received = []
        controller.register_frame_output(received.append)
        frame = controller.command_target(0.25, 10.0, 0.9)
        self.assertEqual(frame.active_channels, (1, 2))
        self.assertEqual(received[-1].active_channels, (1, 2))
        self.assertTrue(all(frame.commands[index].active for index in (0, 1)))
        self.assertTrue(np.array_equal(controller.activation_counts[:2], [1, 1]))
        controller.command_target(0.26, 11.0, 0.9)
        self.assertTrue(np.array_equal(controller.activation_counts[:2], [1, 1]))
        stopped = controller.stop(0.3)
        self.assertEqual(stopped.active_channels, ())
        controller.command_target(0.31, 10.0, 0.9)
        self.assertTrue(np.array_equal(controller.activation_counts[:2], [2, 2]))
        controller.reset_activation_counts()
        self.assertFalse(np.any(controller.activation_counts))

    def test_actuator_requires_two_enabled_outputs(self) -> None:
        controller = ActuatorController(16)
        controller.set_enabled(0, True)
        frame = controller.command_target(1.0, 0.0, 0.5)
        self.assertEqual(frame.active_channels, ())

    def test_repeated_control_pulse_uses_physical_time(self) -> None:
        scheduler = PulseScheduler(100.0)
        self.assertTrue(scheduler.pulse_due(1.0))
        self.assertTrue(np.isclose(scheduler.start(1.0), 1.1))
        self.assertFalse(scheduler.pulse_due(1.099))
        self.assertTrue(scheduler.pulse_due(1.1))
        scheduler.set_duration_ms(250.0)
        self.assertTrue(np.isclose(scheduler.start(2.0), 2.25))
        scheduler.reset()
        self.assertTrue(scheduler.pulse_due(2.01))


if __name__ == "__main__":
    unittest.main()
