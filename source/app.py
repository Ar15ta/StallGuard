from __future__ import annotations

import logging
import os
import sys
import traceback
from pathlib import Path

from PySide6.QtCore import QStandardPaths
from PySide6.QtWidgets import QApplication, QMessageBox

from src.config import resource_path
from src.ui.main_window import MainWindow


def configure_logging() -> None:
    location = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppLocalDataLocation)
    log_dir = Path(location) / "logs" if location else Path.cwd() / "logs"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "stall_warning.log"
    except OSError:
        log_path = Path.cwd() / "stall_warning.log"
    try:
        file_handler: logging.Handler = logging.FileHandler(log_path, encoding="utf-8")
    except OSError:
        try:
            file_handler = logging.FileHandler(Path.cwd() / f"stall_warning_{os.getpid()}.log", encoding="utf-8")
        except OSError:
            file_handler = logging.NullHandler()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[file_handler, logging.StreamHandler()],
    )


def _smoke_test(window: MainWindow, csv_path: Path) -> int:
    import numpy as np

    from src.io_csv import load_cfd_csv
    from src.models import DataSourceSettings
    from src.online_engine import OnlineSnapshot, OnlineWarningEngine

    dataset = load_cfd_csv(csv_path, DataSourceSettings())
    window._adopt_dataset(dataset)
    params = window._machine_params()
    params["window_size"] = min(1024, dataset.sample_count)
    params["step_size"] = max(1, params["window_size"] // 4)
    engine = OnlineWarningEngine(dataset.probe_count, str(window.machine_method.currentData()), params, calibration_windows=10)
    commissioning_rows = params["window_size"] + 9 * params["step_size"]
    snapshots = engine.update(dataset.time_s[:commissioning_rows], dataset.values[:commissioning_rows])
    if not engine.calibrated or not snapshots or engine.calibration_profile() is None:
        raise RuntimeError("Smoke-test online calibration failed")
    profile = engine.calibration_profile()
    profile["deployment_context"] = window._profile_context()
    window.online_baseline_profile = profile
    window._prepare_sensor_session(capacity=2048)
    window._handle_sensor_message(
        {
            "type": "meta",
            "probe_names": dataset.probe_names,
            "output_interval_s": dataset.settings.output_interval_s,
            "total_rows": 2048,
        }
    )
    window._handle_sensor_message(
        {"type": "samples", "steps": dataset.steps[:2048].tolist(), "values": dataset.values[:2048].tolist()}
    )
    window._refresh_live_plots(force=True)
    if (
        window.live_count != 2048
        or window.cwt_frequencies_hz.size != 64
        or window.cwt_sample_count != min(window.live_count, window.window_size.value())
    ):
        raise RuntimeError("Smoke-test online display path failed")
    if dataset.probe_count > 1:
        window.probe_selector.setCurrentIndex(1)
        window._plot_cwt(force=True)
        if dataset.probe_names[1] not in window.cwt_plot.plotItem.titleLabel.text:
            raise RuntimeError("Smoke-test CWT probe switching failed")
    # Explicitly exercise both time-domain branches and their axis policies.
    window.view_mode.setCurrentIndex(window.view_mode.findData("pressure"))
    window._plot_live_time_domain()
    window.view_mode.setCurrentIndex(window.view_mode.findData("rms"))
    window._plot_live_time_domain()
    # Exercise the active-control panel, output routing and animation data path.
    window._set_all_actuators_enabled(True)
    synthetic_risk = np.array([5.0, 4.0] + [0.0] * max(0, dataset.probe_count - 2), dtype=float)
    synthetic_alarm = np.array([True, True] + [False] * max(0, dataset.probe_count - 2), dtype=bool)
    synthetic_risk = synthetic_risk[: dataset.probe_count]
    synthetic_alarm = synthetic_alarm[: dataset.probe_count]
    first_alarm = synthetic_alarm.copy()
    if first_alarm.size > 1:
        first_alarm[1:] = False
    first = OnlineSnapshot("MONITORING", 1, 1.0, synthetic_risk, synthetic_risk, first_alarm, False, 1.0)
    window._update_active_control(first)
    second = OnlineSnapshot("ALARM", 2, 1.2, synthetic_risk, synthetic_risk, synthetic_alarm, True, 1.0)
    window._update_active_control(second)
    if len(window.actuator_controller.last_frame.active_channels) != 2:
        raise RuntimeError("Smoke-test active-control routing failed")
    within_pulse = OnlineSnapshot("ALARM", 3, 1.25, synthetic_risk, synthetic_risk, synthetic_alarm, True, 1.0)
    previous_target = window.actuator_controller.last_frame.target_angle_deg
    window._update_active_control(within_pulse)
    if window._pulse_sequence != 1:
        raise RuntimeError("Smoke-test pulse hold failed")
    if window.actuator_controller.last_frame.target_angle_deg == previous_target:
        raise RuntimeError("Smoke-test in-pulse position tracking failed")
    next_pulse = OnlineSnapshot("ALARM", 4, 1.31, synthetic_risk, synthetic_risk, synthetic_alarm, True, 1.0)
    window._update_active_control(next_pulse)
    if window._pulse_sequence != 2:
        raise RuntimeError("Smoke-test pulse reacquisition failed")
    window._sensor_disconnected()
    if window.actuator_controller.last_frame.active_channels:
        raise RuntimeError("Smoke-test disconnect fail-safe failed")
    window.close()
    return 0


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("StallWarning")
    app.setOrganizationName("StallGuard")
    app.setStyle("Fusion")
    configure_logging()
    try:
        app.setStyleSheet(resource_path("assets/theme.qss").read_text(encoding="utf-8"))
    except OSError:
        logging.getLogger(__name__).warning("Theme file unavailable")

    def report_unhandled(exc_type, exc_value, exc_tb) -> None:
        text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        logging.getLogger(__name__).critical("Unhandled exception\n%s", text)
        QMessageBox.critical(None, "未处理异常", f"程序发生错误：\n{exc_value}")

    sys.excepthook = report_unhandled
    window = MainWindow()
    if "--smoke-test" in sys.argv:
        try:
            argument_index = sys.argv.index("--smoke-test")
            return _smoke_test(window, Path(sys.argv[argument_index + 1]))
        except Exception:
            logging.getLogger(__name__).exception("Source smoke test failed")
            window.close()
            return 1
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
