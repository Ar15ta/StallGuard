from __future__ import annotations

import json
import logging
import time
from bisect import bisect_left
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyqtgraph as pg
import pywt
from PySide6.QtCore import QRectF, Qt, QTimer
from PySide6.QtGui import QAction
from PySide6.QtNetwork import QHostAddress, QTcpServer, QTcpSocket
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..analysis_service import METHOD_LABELS
from ..actuation import ActuatorController, PulseScheduler
from ..config import load_defaults
from ..io_csv import load_cfd_csv
from ..models import ProbeDataset
from ..online_engine import OnlineSnapshot, OnlineWarningEngine
from ..preprocessing import (
    OnlineSignalFilter,
    causal_rolling_rms,
    circumferential_amplitude,
    normalize_rms,
)
from ..streaming import SampleBatch
from ..stall_tracking import StallLocationEstimate, StallPositionTracker
from .actuator_panel import ActuatorPanel
from .data_source_dialog import DataSourceDialog, confirm_dataset_mapping
from ..warning_engine import fit_robust_baseline

LOGGER = logging.getLogger(__name__)
PALETTE = ("#38BDF8", "#22C55E", "#F59E0B", "#A78BFA", "#EC4899", "#14B8A6", "#F97316")


def _spin(minimum: int, maximum: int, value: int, tooltip: str) -> QSpinBox:
    widget = QSpinBox()
    widget.setRange(minimum, maximum)
    widget.setValue(value)
    widget.setToolTip(tooltip)
    return widget


class MainWindow(QMainWindow):
    """Industrial desktop UI for baseline commissioning and live monitoring."""

    def __init__(self) -> None:
        super().__init__()
        self.defaults = load_defaults()
        self.dataset: ProbeDataset | None = None
        self.online_engine: OnlineWarningEngine | None = None
        self.online_baseline_profile: dict[str, Any] | None = None
        self.signal_filter: OnlineSignalFilter | None = None
        self.server: QTcpServer | None = None
        self.sensor_socket: QTcpSocket | None = None
        self._tcp_buffer = b""
        self.tcp_meta_valid = False
        self.live_time_origin_step: float | None = None
        self._initialize_live_buffers(0)
        self.live_risk_times: list[float] = []
        self.live_risk_rows: list[np.ndarray] = []
        self.last_snapshot: OnlineSnapshot | None = None
        self.last_stall_estimate: StallLocationEstimate | None = None
        self._last_active_channels: tuple[int, ...] = ()
        self.actuator_controller = ActuatorController(16)
        self.pulse_scheduler = PulseScheduler(100.0)
        self._pulse_sequence = 0
        self.stall_tracker = StallPositionTracker()
        self.cwt_frequencies_hz = np.empty(0, dtype=float)
        self.cwt_sample_count = 0
        self._plots_dirty = False
        self._last_cwt_wall_time = 0.0
        self._display_window_s = 0.1

        self.setWindowTitle("StallGuard · 失速预警系统")
        self.resize(1520, 940)
        self._build_menu()
        self._build_ui()
        self._render_timer = QTimer(self)
        self._render_timer.setInterval(max(33, int(self.defaults["online"].get("timer_interval_ms", 66))))
        self._render_timer.timeout.connect(self._refresh_live_plots)
        self._render_timer.start()
        self.statusBar().showMessage("请选择正常数据段进行初始化标定")

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("文件")
        initialize = QAction("初始化标定", self)
        initialize.triggered.connect(self.initialize_from_normal_segment)
        file_menu.addAction(initialize)
        export = QAction("导出监听结果", self)
        export.triggered.connect(self.export_results)
        file_menu.addAction(export)

    def _build_ui(self) -> None:
        root = QWidget()
        outer = QVBoxLayout(root)
        outer.addWidget(self._build_header())
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_control_scroll())
        splitter.addWidget(self._build_plot_area())
        splitter.addWidget(self._build_status_panel())
        splitter.setSizes([300, 850, 350])
        outer.addWidget(splitter, 1)
        self.setCentralWidget(root)

    def _build_header(self) -> QWidget:
        frame = QFrame()
        frame.setObjectName("Card")
        layout = QHBoxLayout(frame)
        title_box = QVBoxLayout()
        title = QLabel("StallGuard 失速预警系统")
        title.setObjectName("ProductTitle")
        subtitle = QLabel("在线探针监测 · 稳健基线 · K-of-N 报警")
        subtitle.setObjectName("Muted")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        layout.addLayout(title_box)
        layout.addStretch()
        self.connection_label = QLabel("未初始化")
        self.connection_label.setObjectName("Muted")
        layout.addWidget(self.connection_label)
        return frame

    def _build_control_scroll(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.addWidget(self._build_data_group())
        layout.addWidget(self._build_filter_group())
        layout.addWidget(self._build_analysis_group())
        layout.addWidget(self._build_rms_group())
        layout.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        scroll.setMinimumWidth(285)
        return scroll

    def _build_data_group(self) -> QWidget:
        group = QGroupBox("数据与监听")
        layout = QVBoxLayout(group)
        self.initialize_button = QPushButton("初始化标定")
        self.initialize_button.setObjectName("Primary")
        self.initialize_button.setToolTip("选择一段已知正常数据，冻结当前判据的中心值和波动尺度。")
        self.initialize_button.clicked.connect(self.initialize_from_normal_segment)
        self.stop_button = QPushButton("停止监听")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_sensor_listener)
        self.export_button = QPushButton("导出监听结果")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_results)
        self.stream_progress = QProgressBar()
        self.stream_progress.setRange(0, 0)
        self.stream_progress.setFormat("等待初始化")
        layout.addWidget(self.initialize_button)
        layout.addWidget(self.stop_button)
        layout.addWidget(self.export_button)
        layout.addWidget(self.stream_progress)
        return group

    def _build_filter_group(self) -> QWidget:
        defaults = self.defaults["filter"]
        group = QGroupBox("实时滤波")
        form = QFormLayout(group)
        self.filter_mode = QComboBox()
        self.filter_mode.addItem("关闭", "off")
        self.filter_mode.addItem("低通", "lowpass")
        self.filter_mode.addItem("带通", "bandpass")
        self.filter_mode.setToolTip("Butterworth 实时因果滤波；关闭时直接使用输入压力。")
        self.filter_order = _spin(2, 8, int(defaults["order"]), "Butterworth 滤波器阶数；阶数越高，过渡带越陡。")
        self.low_cutoff = QDoubleSpinBox()
        self.low_cutoff.setRange(0.001, 1e9)
        self.low_cutoff.setValue(1000.0)
        self.low_cutoff.setSuffix(" Hz")
        self.low_cutoff.setToolTip("低通模式的截止频率；带通模式下作为低截止频率。")
        self.high_cutoff = QDoubleSpinBox()
        self.high_cutoff.setRange(0.001, 1e9)
        self.high_cutoff.setValue(10000.0)
        self.high_cutoff.setSuffix(" Hz")
        self.high_cutoff.setToolTip("带通模式的高截止频率，必须小于 Nyquist 频率。")
        form.addRow("模式", self.filter_mode)
        form.addRow("阶数", self.filter_order)
        form.addRow("低截止", self.low_cutoff)
        form.addRow("高截止", self.high_cutoff)
        return group

    def _build_analysis_group(self) -> QWidget:
        defaults = self.defaults["analysis"]
        group = QGroupBox("机器判据")
        form = QFormLayout(group)
        self.machine_method = QComboBox()
        self.machine_method.addItem("时序自相关", "autocorrelation")
        self.machine_method.addItem("数据熵", "entropy")
        self.machine_method.setToolTip("决定红绿报警灯所采用的唯一机器判据。")
        self.window_size = _spin(64, 65536, int(defaults["window_size"]), "每次计算判据所使用的连续样本数。")
        self.overlap = _spin(0, 95, int(defaults["overlap_percent"]), "相邻分析窗口的重叠比例；越高更新越密集。")
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.1, 20.0)
        self.threshold.setDecimals(2)
        self.threshold.setValue(float(defaults["threshold"]))
        self.threshold.setToolTip("相对正常基线的稳健标准化偏差阈值；连续超限后触发报警。")
        self.minimum_probes = _spin(1, 128, int(defaults["minimum_probes"]), "同时报警的最少探针数；系统红灯采用 K-of-N 规则。")
        self.trigger_windows = _spin(1, 50, int(defaults["trigger_windows"]), "连续达到阈值多少个分析窗口后进入报警。")
        self.release_windows = _spin(1, 100, int(defaults["release_windows"]), "连续回落多少个分析窗口后解除报警。")
        self.entropy_order = _spin(3, 7, int(defaults["entropy_order"]), "排列熵的嵌入维数，常用范围为 3～7。")
        self.entropy_delay = _spin(1, 20, int(defaults["entropy_delay"]), "排列向量相邻元素之间的样本延迟。")
        self.min_lag = _spin(1, 10000, int(defaults["min_lag"]), "自相关峰值搜索的最小正滞后点数。")
        self.max_lag = _spin(2, 20000, int(defaults["max_lag"]), "自相关峰值搜索的最大正滞后点数，须小于窗口长度。")
        for label, widget in (
            ("判据", self.machine_method), ("窗口点数", self.window_size), ("重叠", self.overlap),
            ("灵敏度阈值", self.threshold), ("最少报警探针", self.minimum_probes),
            ("触发窗口", self.trigger_windows), ("解除窗口", self.release_windows),
            ("熵嵌入维数", self.entropy_order), ("熵延迟", self.entropy_delay),
            ("最小 lag", self.min_lag), ("最大 lag", self.max_lag),
        ):
            form.addRow(label, widget)
        self.machine_method.currentIndexChanged.connect(self._sync_machine_method_controls)
        self.window_size.valueChanged.connect(self._sync_lag_limits)
        self._sync_machine_method_controls()
        return group

    def _build_rms_group(self) -> QWidget:
        defaults = self.defaults["rms"]
        group = QGroupBox("显示")
        form = QFormLayout(group)
        self.view_mode = QComboBox()
        self.view_mode.addItem("周向 RMS", "rms")
        self.view_mode.addItem("单探针压力", "pressure")
        self.view_mode.setToolTip("切换周向归一 RMS 视图或单个探针的原始压力时域图。")
        self.probe_selector = QComboBox()
        self.probe_selector.setToolTip("单探针压力视图和小波图所使用的探针。")
        self.rms_half = _spin(1, 5000, int(defaults["half_window"]), "RMS 计算的半窗口点数，总窗口为 2N+1。")
        self.downsample = _spin(1, 1000, int(defaults["downsample"]), "仅影响绘图抽点，不改变分析和报警计算。")
        self.amplitude_scale = QDoubleSpinBox()
        self.amplitude_scale.setRange(1.0, 180.0)
        self.amplitude_scale.setValue(float(defaults["amplitude_scale_deg"]))
        self.amplitude_scale.setSuffix(" °")
        self.amplitude_scale.setToolTip("将归一 RMS 映射为周向角度方向的显示振幅。")
        self.cwt_window_count = _spin(1, 3, 1, "小波图仅保留最近若干个相邻分析窗口；1表示当前分析窗。")
        form.addRow("时域视图", self.view_mode)
        form.addRow("探针", self.probe_selector)
        form.addRow("RMS 半窗口", self.rms_half)
        form.addRow("绘图抽点", self.downsample)
        form.addRow("振幅尺度", self.amplitude_scale)
        form.addRow("小波保留窗口", self.cwt_window_count)
        self.view_mode.currentIndexChanged.connect(self._plot_live_time_domain)
        self.probe_selector.currentIndexChanged.connect(self._plot_live_time_domain)
        self.probe_selector.currentIndexChanged.connect(self._cwt_setting_changed)
        self.rms_half.valueChanged.connect(self._plot_live_time_domain)
        self.downsample.valueChanged.connect(self._plot_live_time_domain)
        self.amplitude_scale.valueChanged.connect(self._plot_live_time_domain)
        self.cwt_window_count.valueChanged.connect(self._cwt_setting_changed)
        return group

    def _configure_plot(self, plot: pg.PlotWidget) -> None:
        plot.showGrid(x=True, y=True, alpha=0.18)
        plot.setBackground("#0B1220")
        plot.setLabel("bottom", "物理时间", units="s")
        plot.addLegend(offset=(8, 8))

    def _build_plot_area(self) -> QWidget:
        self.analysis_tabs = QTabWidget()
        online = QWidget()
        online_layout = QVBoxLayout(online)
        self.time_plot = pg.PlotWidget(title="实时波形（等待数据）")
        self.analysis_plot = pg.PlotWidget(title="在线分析（等待首个完整窗口）")
        self._configure_plot(self.time_plot)
        self._configure_plot(self.analysis_plot)
        online_layout.addWidget(self.time_plot, 3)
        online_layout.addWidget(self.analysis_plot, 2)
        self.analysis_tabs.addTab(online, "在线监测")
        wavelet = QWidget()
        wavelet_layout = QVBoxLayout(wavelet)
        self.cwt_plot = pg.PlotWidget(title="小波分析")
        self.cwt_plot.setBackground("#0B1220")
        self.cwt_plot.setLabel("bottom", "物理时间", units="s")
        self.cwt_plot.setLabel("left", "频率", units="Hz")
        wavelet_layout.addWidget(self.cwt_plot)
        self.analysis_tabs.addTab(wavelet, "小波分析")
        self.analysis_tabs.currentChanged.connect(self._analysis_tab_changed)
        return self.analysis_tabs

    def _build_status_panel(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(330)
        layout = QVBoxLayout(panel)
        card = QFrame()
        card.setObjectName("Card")
        card_layout = QVBoxLayout(card)
        self.alarm_light = QLabel("●")
        self.alarm_light.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.alarm_light.setStyleSheet("font-size:72px;color:#22C55E")
        self.state_text = QLabel("未初始化")
        self.state_text.setObjectName("StateText")
        self.state_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.risk_metric_label = QLabel("—")
        self.risk_metric_label.setObjectName("Metric")
        self.risk_metric_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(self.alarm_light)
        card_layout.addWidget(self.state_text)
        card_layout.addWidget(self.risk_metric_label)
        self.event_list = QListWidget()
        layout.addWidget(card)
        layout.addWidget(QLabel("事件记录"))
        layout.addWidget(self.event_list)
        self.event_list.setMaximumHeight(125)
        self.tcp_status = QLabel("监听未启动")
        self.tcp_status.setWordWrap(True)
        layout.addWidget(self.tcp_status)
        self.actuator_panel = ActuatorPanel(self.actuator_controller)
        self.actuator_panel.enabled_changed.connect(self._set_actuator_enabled)
        self.actuator_panel.angle_changed.connect(self._set_actuator_angle)
        self.actuator_panel.all_enabled_changed.connect(self._set_all_actuators_enabled)
        layout.addWidget(self.actuator_panel, 1)
        return panel

    def _sync_machine_method_controls(self, *_args) -> None:
        entropy = self.machine_method.currentData() == "entropy"
        self.entropy_order.setEnabled(entropy)
        self.entropy_delay.setEnabled(entropy)
        self.min_lag.setEnabled(not entropy)
        self.max_lag.setEnabled(not entropy)

    def _sync_lag_limits(self, value: int) -> None:
        self.max_lag.setMaximum(max(2, value - 1))
        if self.max_lag.value() >= value:
            self.max_lag.setValue(max(2, value - 1))

    def _set_machine_controls_locked(self, locked: bool) -> None:
        for widget in (
            self.machine_method, self.window_size, self.overlap, self.threshold, self.minimum_probes,
            self.trigger_windows, self.release_windows, self.entropy_order, self.entropy_delay,
            self.min_lag, self.max_lag,
        ):
            widget.setEnabled(not locked)
        for widget in (self.filter_mode, self.filter_order, self.low_cutoff, self.high_cutoff):
            widget.setEnabled(not locked)
        if not locked:
            self._sync_machine_method_controls()

    def _machine_direction(self) -> str:
        return "high"

    def _machine_direction_label(self) -> str:
        return "上升异常"

    def _machine_params(self) -> dict[str, Any]:
        window = self.window_size.value()
        step = max(1, round(window * (100 - self.overlap.value()) / 100))
        return {
            "window_size": window,
            "step_size": step,
            "threshold": self.threshold.value(),
            "direction": self._machine_direction(),
            "trigger_windows": self.trigger_windows.value(),
            "release_windows": self.release_windows.value(),
            "release_ratio": float(self.defaults["analysis"]["release_ratio"]),
            "minimum_probes": self.minimum_probes.value(),
            "entropy_order": self.entropy_order.value(),
            "entropy_delay": self.entropy_delay.value(),
            "min_lag": self.min_lag.value(),
            "max_lag": min(self.max_lag.value(), window - 1),
        }

    def _profile_context(self) -> dict[str, Any]:
        if self.dataset is None:
            return {}
        return {
            "probe_names": self.dataset.probe_names,
            "angles_deg": self.dataset.angles_deg.tolist(),
            "output_interval_s": self.dataset.settings.output_interval_s,
            "unit": self.dataset.unit,
            "filter": {
                "mode": str(self.filter_mode.currentData()),
                "order": self.filter_order.value(),
                "low_cutoff_hz": self.low_cutoff.value(),
                "high_cutoff_hz": self.high_cutoff.value(),
            },
        }

    def _new_signal_filter(self) -> OnlineSignalFilter:
        if self.dataset is None:
            raise RuntimeError("尚未加载数据。")
        mode = str(self.filter_mode.currentData())
        return OnlineSignalFilter(
            self.dataset.probe_count,
            self.dataset.sample_rate_hz,
            mode=mode,
            low_cutoff_hz=self.low_cutoff.value() if mode != "off" else None,
            high_cutoff_hz=self.high_cutoff.value() if mode == "bandpass" else None,
            order=self.filter_order.value(),
        )

    def initialize_from_normal_segment(self, *_args) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择正常数据段", "", "CSV 数据 (*.csv);;所有文件 (*)")
        if not path:
            return
        dialog = DataSourceDialog(self.defaults["data_source"], self)
        if dialog.exec() != DataSourceDialog.DialogCode.Accepted:
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            dataset = load_cfd_csv(path, dialog.settings())
            if not confirm_dataset_mapping(dataset, self):
                return
            self._adopt_dataset(dataset)
            params = self._machine_params()
            available = 1 + max(0, (dataset.sample_count - params["window_size"]) // params["step_size"])
            calibration_windows = min(int(self.defaults["online"]["calibration_windows"]), available)
            if calibration_windows < 10:
                raise ValueError("正常数据段过短，至少需要生成 10 个标定窗口。")
            commissioning_rows = params["window_size"] + (calibration_windows - 1) * params["step_size"]
            engine = OnlineWarningEngine(
                dataset.probe_count,
                str(self.machine_method.currentData()),
                params,
                calibration_windows=calibration_windows,
            )
            calibration_filter = self._new_signal_filter()
            calibrated_values = calibration_filter.process(dataset.values[:commissioning_rows])
            engine.update(dataset.time_s[:commissioning_rows], calibrated_values)
            profile = engine.calibration_profile()
            if profile is None:
                raise RuntimeError("初始化数据未能完成基线标定。")
            pressure_center, pressure_scale = fit_robust_baseline(calibrated_values)
            profile["pressure_center"] = pressure_center.tolist()
            profile["pressure_scale"] = pressure_scale.tolist()
            self.stall_tracker.set_pressure_baseline(pressure_center, pressure_scale)
            profile["deployment_context"] = self._profile_context()
            self.online_baseline_profile = profile
            self._set_machine_controls_locked(True)
            self._start_sensor_listener()
            self.event_list.addItem(f"初始化标定完成：{Path(path).name}")
        except Exception as exc:
            LOGGER.exception("Initialization failed")
            QMessageBox.critical(self, "初始化失败", str(exc))
        finally:
            QApplication.restoreOverrideCursor()

    def _adopt_dataset(self, dataset: ProbeDataset) -> None:
        self.dataset = dataset
        self.stall_tracker.configure(dataset.angles_deg)
        pressure_center, pressure_scale = fit_robust_baseline(dataset.values)
        self.stall_tracker.set_pressure_baseline(pressure_center, pressure_scale)
        self.probe_selector.clear()
        self.probe_selector.addItems(dataset.probe_names)
        self.minimum_probes.setMaximum(dataset.probe_count)
        self.minimum_probes.setValue(min(self.minimum_probes.value(), dataset.probe_count))
        self.connection_label.setText(f"{dataset.probe_count} 探针 · {dataset.sample_rate_hz:,.1f} Hz")

    def _start_sensor_listener(self) -> None:
        self.stop_sensor_listener(quiet=True)
        self._prepare_sensor_session()
        self.server = QTcpServer(self)
        self.server.newConnection.connect(self._accept_sensor_connection)
        if not self.server.listen(QHostAddress.SpecialAddress.LocalHost, 8765):
            message = self.server.errorString()
            self.server.deleteLater()
            self.server = None
            raise RuntimeError(f"无法监听 127.0.0.1:8765：{message}")
        self.stop_button.setEnabled(True)
        self.tcp_status.setText("正在监听 127.0.0.1:8765")
        self._set_state("MONITORING", "等待传感器连接")

    def _prepare_sensor_session(self, capacity: int = 4096) -> None:
        if self.dataset is None or self.online_baseline_profile is None:
            raise RuntimeError("请先完成初始化标定。")
        self._initialize_live_buffers(self.dataset.probe_count, capacity)
        self.live_risk_times = []
        self.live_risk_rows = []
        self.last_snapshot = None
        self.last_stall_estimate = None
        self._last_active_channels = ()
        self._pulse_sequence = 0
        self.pulse_scheduler.reset()
        self.stall_tracker.reset()
        idle_frame = self.actuator_controller.stop(0.0)
        self.actuator_controller.reset_activation_counts()
        self.actuator_panel.show_control(None, idle_frame)
        self.live_time_origin_step = None
        self.tcp_meta_valid = False
        params = self._machine_params()
        self.online_engine = OnlineWarningEngine(
            self.dataset.probe_count,
            str(self.machine_method.currentData()),
            params,
            calibration_windows=max(10, int(self.online_baseline_profile.get("calibration_windows", 10))),
            baseline_profile=self.online_baseline_profile,
        )
        self.signal_filter = self._new_signal_filter()
        pressure_center = np.asarray(self.online_baseline_profile.get("pressure_center", []), dtype=float)
        pressure_scale = np.asarray(self.online_baseline_profile.get("pressure_scale", []), dtype=float)
        if pressure_center.shape == (self.dataset.probe_count,) and pressure_scale.shape == (self.dataset.probe_count,):
            self.stall_tracker.set_pressure_baseline(pressure_center, pressure_scale)
        self._plots_dirty = True
        self._last_cwt_wall_time = 0.0

    def _accept_sensor_connection(self) -> None:
        if self.server is None:
            return
        if self.sensor_socket is not None:
            self.sensor_socket.disconnectFromHost()
        self.sensor_socket = self.server.nextPendingConnection()
        self.sensor_socket.readyRead.connect(self._read_sensor_socket)
        self.sensor_socket.disconnected.connect(self._sensor_disconnected)
        self._tcp_buffer = b""
        self.tcp_meta_valid = False
        self.tcp_status.setText("传感器已连接，等待数据契约")

    def _read_sensor_socket(self) -> None:
        if self.sensor_socket is None:
            return
        self._tcp_buffer += bytes(self.sensor_socket.readAll())
        while b"\n" in self._tcp_buffer:
            raw, self._tcp_buffer = self._tcp_buffer.split(b"\n", 1)
            if not raw.strip():
                continue
            try:
                self._handle_sensor_message(json.loads(raw.decode("utf-8")))
            except Exception as exc:
                self._sensor_protocol_error(str(exc))
                return

    def _handle_sensor_message(self, message: dict[str, Any]) -> None:
        if self.dataset is None:
            raise RuntimeError("尚未初始化在线数据套约。")
        message_type = message.get("type")
        if message_type == "meta":
            if message.get("probe_names") != self.dataset.probe_names:
                raise ValueError("在线探针名称或顺序与初始化数据不一致。")
            interval = float(message.get("output_interval_s", np.nan))
            if not np.isclose(interval, self.dataset.settings.output_interval_s):
                raise ValueError("在线响应输出间隔与初始化数据不一致。")
            total_rows = max(0, int(message.get("total_rows", 0)))
            if total_rows:
                self._ensure_live_capacity(total_rows)
                self.stream_progress.setRange(0, total_rows)
            self.tcp_meta_valid = True
            self.tcp_status.setText(
                f"在线采集中 · 机器判据 {METHOD_LABELS[str(self.machine_method.currentData())]}"
                f"（{self._machine_direction_label()}）"
            )
            self.event_list.addItem("数据契约校验通过，开始在线判定")
            return
        if not self.tcp_meta_valid:
            raise ValueError("在线数据必须先发送 meta 握手。")
        if message_type == "sample":
            steps = np.asarray([message["step"]], dtype=float)
            values = np.asarray([message["values"]], dtype=float)
        elif message_type == "samples":
            steps = np.asarray(message["steps"], dtype=float)
            values = np.asarray(message["values"], dtype=float)
        else:
            raise ValueError(f"未知在线消息类型：{message_type}")
        if values.ndim != 2 or values.shape != (steps.size, self.dataset.probe_count):
            raise ValueError("在线样本的探针维度与初始化数据不一致。")
        if steps.size == 0:
            return
        if self.live_count == 0:
            self.live_time_origin_step = float(steps[0])
        time_s = (steps - float(self.live_time_origin_step)) * self.dataset.settings.output_interval_s
        self._consume_live_batch(SampleBatch(steps=steps, time_s=time_s, values=values))
        self.export_button.setEnabled(self.live_count > 0)
        if self.stream_progress.maximum() > 0:
            self.stream_progress.setValue(min(self.live_count, self.stream_progress.maximum()))
            self.stream_progress.setFormat(f"在线接收 {self.live_count:,}/{self.stream_progress.maximum():,} 行")
        else:
            self.stream_progress.setFormat(f"在线接收 {self.live_count:,} 行")

    def _sensor_protocol_error(self, message: str) -> None:
        self.event_list.addItem("协议错误：" + message)
        self.tcp_status.setText("传感器数据错误")
        if self.sensor_socket is not None:
            self.sensor_socket.disconnectFromHost()

    def _sensor_disconnected(self) -> None:
        self.tcp_status.setText("传感器已断开；监听端仍在等待连接")
        timestamp_s = self.last_snapshot.time_s if self.last_snapshot is not None else 0.0
        had_active_output = bool(self.actuator_controller.last_frame.active_channels)
        frame = self.actuator_controller.stop(timestamp_s)
        self.pulse_scheduler.reset()
        self._pulse_sequence = 0
        self.last_stall_estimate = None
        self._last_active_channels = ()
        self.actuator_panel.show_control(None, frame)
        self._set_state("STOPPED", "数据流停止")
        if had_active_output:
            self.event_list.addItem(f"{timestamp_s:.6f} s · 数据流结束，主动控制输出已撤销")
        self.sensor_socket = None
        self.tcp_meta_valid = False

    def stop_sensor_listener(self, *_args, quiet: bool = False) -> None:
        if self.sensor_socket is not None:
            self.sensor_socket.disconnectFromHost()
            self.sensor_socket.deleteLater()
            self.sensor_socket = None
        if self.server is not None:
            self.server.close()
            self.server.deleteLater()
            self.server = None
        self.stop_button.setEnabled(False) if hasattr(self, "stop_button") else None
        timestamp_s = self.last_snapshot.time_s if self.last_snapshot is not None else 0.0
        frame = self.actuator_controller.stop(timestamp_s)
        self.pulse_scheduler.reset()
        self._pulse_sequence = 0
        self.last_stall_estimate = None
        self._last_active_channels = ()
        if hasattr(self, "actuator_panel"):
            self.actuator_panel.show_control(None, frame)
        if not quiet and hasattr(self, "tcp_status"):
            self.tcp_status.setText("监听已停止")
            self._set_state("STOPPED", "监听已停止")

    def _initialize_live_buffers(self, probe_count: int, capacity: int = 4096) -> None:
        capacity = max(1, int(capacity))
        self.live_capacity = capacity
        self.live_count = 0
        self.live_steps = np.empty(capacity, dtype=float)
        self.live_times = np.empty(capacity, dtype=float)
        self.live_values = np.empty((capacity, probe_count), dtype=float)
        self.live_filtered = np.empty((capacity, probe_count), dtype=float) if probe_count else None
        self.live_alarm_samples = np.zeros((capacity, probe_count), dtype=bool) if probe_count else None

    def _ensure_live_capacity(self, required: int) -> None:
        if self.dataset is None or required <= self.live_capacity:
            return
        new_capacity = max(required, self.live_capacity * 2)
        old_count = self.live_count
        arrays = (self.live_steps, self.live_times)
        self.live_steps = np.resize(arrays[0], new_capacity)
        self.live_times = np.resize(arrays[1], new_capacity)
        self.live_values = np.resize(self.live_values, (new_capacity, self.dataset.probe_count))
        self.live_filtered = np.resize(self.live_filtered, (new_capacity, self.dataset.probe_count))
        self.live_alarm_samples = np.resize(self.live_alarm_samples, (new_capacity, self.dataset.probe_count))
        self.live_alarm_samples[old_count:] = False
        self.live_capacity = new_capacity

    def _consume_live_batch(self, batch: SampleBatch) -> list[OnlineSnapshot]:
        if self.dataset is None or self.online_engine is None or self.signal_filter is None:
            raise RuntimeError("在线会话尚未准备完成。")
        start = self.live_count
        end = start + batch.steps.size
        self._ensure_live_capacity(end)
        filtered = self.signal_filter.process(batch.values)
        self.live_steps[start:end] = batch.steps
        self.live_times[start:end] = batch.time_s
        self.live_values[start:end] = batch.values
        self.live_filtered[start:end] = filtered
        self.live_alarm_samples[start:end] = self.last_snapshot.probe_alarm if self.last_snapshot else False
        self.live_count = end
        snapshots = self.online_engine.update(batch.time_s, filtered)
        for snapshot in snapshots:
            self.last_snapshot = snapshot
            self.live_risk_times.append(snapshot.time_s)
            self.live_risk_rows.append(snapshot.risk_score.copy())
            sample_index = max(start, snapshot.sample_end - 1)
            if sample_index < end:
                self.live_alarm_samples[sample_index:end] = snapshot.probe_alarm
            self._update_active_control(snapshot)
        self._update_live_status(self.last_snapshot)
        self._plots_dirty = True
        return snapshots

    def _refresh_live_plots(self, force: bool = False) -> None:
        """Render at a bounded frame rate while analysis continues for every window."""
        if not force and not self._plots_dirty:
            return
        self._plots_dirty = False
        self._plot_live_time_domain()
        self._plot_live_analysis()
        if force or self.analysis_tabs.currentIndex() == 1:
            self._plot_cwt(force=force)

    def _analysis_tab_changed(self, index: int) -> None:
        if index == 1:
            self._plot_cwt(force=True)

    def _cwt_setting_changed(self, *_args) -> None:
        if self.analysis_tabs.currentIndex() == 1:
            self._plot_cwt(force=True)

    def _update_active_control(self, snapshot: OnlineSnapshot) -> None:
        pressure_profile = self._pressure_profile_for_snapshot(snapshot.sample_end)
        estimate = self.stall_tracker.update(
            snapshot.time_s,
            snapshot.risk_score,
            snapshot.probe_alarm,
            snapshot.system_alarm,
            pressure_profile=pressure_profile,
        )
        self.last_stall_estimate = estimate
        self.pulse_scheduler.set_duration_ms(self.actuator_panel.pulse_duration_ms.value())
        pulse_started = False
        if snapshot.system_alarm and estimate is not None:
            if self.pulse_scheduler.pulse_due(snapshot.time_s):
                # End the previous pulse, reacquire the latest disturbance position,
                # then issue a fresh command even when the same channels remain nearest.
                self.actuator_controller.stop(snapshot.time_s)
                frame = self.actuator_controller.command_target(
                    snapshot.time_s,
                    estimate.angle_deg,
                    estimate.confidence,
                )
                self.pulse_scheduler.start(snapshot.time_s)
                self._pulse_sequence += 1
                pulse_started = True
            else:
                # Continue following the latest position inside the pulse instead
                # of freezing the nozzle aim until the next pulse boundary.
                frame = self.actuator_controller.command_target(
                    snapshot.time_s,
                    estimate.angle_deg,
                    estimate.confidence,
                )
        else:
            frame = self.actuator_controller.stop(snapshot.time_s)
            self.pulse_scheduler.reset()
            self._pulse_sequence = 0
        self.actuator_panel.show_control(
            estimate,
            frame,
            self._pulse_sequence if frame.active_channels else None,
        )
        active = frame.active_channels
        if pulse_started:
            if len(active) == 2:
                message = f"{active[0]}号、{active[1]}号动作机构反馈控制中"
                self.event_list.addItem(
                    f"{snapshot.time_s:.6f} s · 第{self._pulse_sequence}次喷气 · {message}"
                )
            else:
                self.event_list.addItem(
                    f"{snapshot.time_s:.6f} s · 第{self._pulse_sequence}次重定位 · 可用动作机构不足2个，未输出"
                )
            while self.event_list.count() > 500:
                self.event_list.takeItem(0)
        elif active != self._last_active_channels:
            if len(active) == 2:
                self.event_list.addItem(
                    f"{snapshot.time_s:.6f} s · 定位更新 · {active[0]}号、{active[1]}号动作机构输出"
                )
            elif self._last_active_channels:
                self.event_list.addItem(f"{snapshot.time_s:.6f} s · 主动控制指令已撤销")
        while self.event_list.count() > 500:
            self.event_list.takeItem(0)
        if active != self._last_active_channels:
            self._last_active_channels = active

    def _pressure_profile_for_snapshot(self, sample_end: int) -> np.ndarray | None:
        if self.live_filtered is None or self.live_count < 1:
            return None
        end = min(max(0, int(sample_end)), self.live_count)
        if end < 1:
            return None
        start = max(0, end - 256)
        return np.nanmedian(self.live_filtered[start:end], axis=0)

    def _set_actuator_enabled(self, index: int, enabled: bool) -> None:
        self.actuator_controller.set_enabled(index, enabled)
        self._refresh_actuator_configuration()

    def _set_actuator_angle(self, index: int, angle_deg: float) -> None:
        self.actuator_controller.set_manual_offset(index, angle_deg)
        self._refresh_actuator_configuration()

    def _set_all_actuators_enabled(self, enabled: bool) -> None:
        self.actuator_controller.set_all_enabled(enabled)
        self._refresh_actuator_configuration()

    def _refresh_actuator_configuration(self) -> None:
        if self.last_snapshot is not None and self.last_snapshot.system_alarm and self.last_stall_estimate is not None:
            self.actuator_controller.stop(self.last_snapshot.time_s)
            frame = self.actuator_controller.command_target(
                self.last_snapshot.time_s,
                self.last_stall_estimate.angle_deg,
                self.last_stall_estimate.confidence,
            )
            self.pulse_scheduler.set_duration_ms(self.actuator_panel.pulse_duration_ms.value())
            self.pulse_scheduler.start(self.last_snapshot.time_s)
            self._pulse_sequence += 1
            self.actuator_panel.show_control(self.last_stall_estimate, frame, self._pulse_sequence)
        else:
            frame = self.actuator_controller.stop(self.last_snapshot.time_s if self.last_snapshot is not None else 0.0)
            self.actuator_panel.show_control(None, frame)
        self.actuator_panel.refresh_configuration()

    def _update_live_status(self, snapshot: OnlineSnapshot | None) -> None:
        if snapshot is None:
            return
        if snapshot.system_alarm:
            self._set_state("ALARM", "失速预警")
        else:
            self._set_state("MONITORING", "运行正常")
        finite = snapshot.risk_score[np.isfinite(snapshot.risk_score)]
        self.risk_metric_label.setText(f"{np.max(finite):.2f} σ" if finite.size else "基线就绪")

    def _plot_live_analysis(self) -> None:
        self.analysis_plot.clear()
        if self.analysis_plot.plotItem.legend is not None:
            self.analysis_plot.plotItem.legend.clear()
        if self.dataset is None or self.online_engine is None or not self.live_risk_rows:
            self.analysis_plot.setTitle("在线分析（等待首个完整窗口）")
            return
        x_range = self.time_plot.viewRange()[0]
        first = max(0, bisect_left(self.live_risk_times, float(x_range[0])) - 1)
        times = np.asarray(self.live_risk_times[first:], dtype=float)
        deviations = np.vstack(self.live_risk_rows[first:])
        for probe, (name, angle) in enumerate(zip(self.dataset.probe_names, self.dataset.angles_deg, strict=True)):
            self.analysis_plot.plot(times, deviations[:, probe], pen=pg.mkPen(PALETTE[probe % len(PALETTE)], width=1.5), name=f"{name} ({angle:.1f}°)")
        self.analysis_plot.addLine(y=self.online_engine.threshold, pen=pg.mkPen("#F59E0B", width=2))
        self.analysis_plot.setTitle(f"判据：{METHOD_LABELS[self.online_engine.method]}")
        self.analysis_plot.setLabel("left", "判据标准化偏差")
        if times.size > 1:
            self.analysis_plot.setXRange(float(x_range[0]), float(x_range[1]), padding=0.0)

    def _plot_live_time_domain(self, *_args) -> None:
        self.time_plot.clear()
        if self.time_plot.plotItem.legend is not None:
            self.time_plot.plotItem.legend.clear()
        if self.dataset is None or self.live_filtered is None or self.live_count < 1:
            self.time_plot.setTitle("实时波形（等待数据）")
            return
        factor = max(1, self.downsample.value())
        window_s = self._display_window_s
        visible_start = int(
            np.searchsorted(
                self.live_times[: self.live_count],
                self.live_times[self.live_count - 1] - window_s,
                side="left",
            )
        )
        if self.view_mode.currentData() == "pressure":
            indices = np.arange(visible_start, self.live_count, factor)
            if indices.size == 0 or indices[-1] != self.live_count - 1:
                indices = np.append(indices, self.live_count - 1)
            x = self.live_times[indices]
            alarm_data = self.live_alarm_samples[indices]
            probe = max(0, self.probe_selector.currentIndex())
            y, alarm = self.live_filtered[indices, probe], alarm_data[:, probe]
            self.time_plot.plot(x, np.where(alarm, np.nan, y), pen=pg.mkPen("#38BDF8", width=1.2), name="滤波压力")
            self.time_plot.plot(x, np.where(alarm, y, np.nan), pen=pg.mkPen("#EF4444", width=2.2), name="预警区间")
            self.time_plot.setLabel("left", "压力", units=self.dataset.unit if self.dataset.unit != "未指定" else None)
            self.time_plot.setTitle(f"实时 · {self.dataset.probe_names[probe]} · 压力时域图")
            self.time_plot.enableAutoRange(axis="y", enable=True)
        else:
            rms_window = 2 * self.rms_half.value() + 1
            calculation_start = max(0, visible_start - (rms_window - 1))
            local_values = self.live_filtered[calculation_start : self.live_count]
            relative_start = visible_start - calculation_start
            relative_indices = np.arange(relative_start, local_values.shape[0], factor)
            if relative_indices.size == 0 or relative_indices[-1] != local_values.shape[0] - 1:
                relative_indices = np.append(relative_indices, local_values.shape[0] - 1)
            indices = calculation_start + relative_indices
            x = self.live_times[indices]
            alarm_data = self.live_alarm_samples[indices]
            if local_values.shape[0] < 2:
                amplitude = np.broadcast_to(
                    self.dataset.angles_deg,
                    (local_values.shape[0], self.dataset.probe_count),
                ).copy()
            else:
                live_rms = causal_rolling_rms(local_values, rms_window)
                normalized, _ = normalize_rms(live_rms, local_values)
                amplitude, _ = circumferential_amplitude(normalized, self.dataset.angles_deg, self.amplitude_scale.value())
            for probe, (name, angle) in enumerate(zip(self.dataset.probe_names, self.dataset.angles_deg, strict=True)):
                y, alarm = amplitude[relative_indices, probe], alarm_data[:, probe]
                self.time_plot.addLine(y=float(angle), pen=pg.mkPen("#334155", width=0.7))
                self.time_plot.plot(x, np.where(alarm, np.nan, y), pen=pg.mkPen(PALETTE[probe % len(PALETTE)], width=1), name=f"{name} · {angle:.1f}°", skipFiniteCheck=True)
                self.time_plot.plot(x, np.where(alarm, y, np.nan), pen=pg.mkPen("#EF4444", width=2.3), skipFiniteCheck=True)
            ticks = sorted((float(angle), f"{angle:.1f}°") for angle in self.dataset.angles_deg)
            self.time_plot.getAxis("left").setTicks([ticks])
            self.time_plot.setLabel("left", "周向位置", units="°")
            self.time_plot.setTitle("实时 · 时间—周向位置—归一 RMS 振幅")
            self.time_plot.setYRange(-5, 360 + self.amplitude_scale.value(), padding=0.01)
        self.time_plot.addLine(x=float(x[-1]), pen=pg.mkPen("#94A3B8", width=1, style=Qt.PenStyle.DotLine))
        self.time_plot.setXRange(float(x[-1]) - window_s, float(x[-1]), padding=0.0)

    def _plot_cwt(self, *_args, force: bool = False) -> None:
        if not force and hasattr(self, "analysis_tabs") and self.analysis_tabs.currentIndex() != 1:
            return
        now = time.monotonic()
        if not force and now - self._last_cwt_wall_time < 0.2:
            return
        self._last_cwt_wall_time = now
        if self.dataset is None or self.live_filtered is None or self.live_count < 64:
            self.cwt_sample_count = 0
            if hasattr(self, "cwt_plot"):
                self.cwt_plot.clear()
                self.cwt_plot.setTitle("小波分析（等待当前分析窗）")
            return
        probe = max(0, self.probe_selector.currentIndex())
        params = self._machine_params()
        retained_windows = self.cwt_window_count.value()
        requested = params["window_size"] + (retained_windows - 1) * params["step_size"]
        count = min(self.live_count, requested)
        self.cwt_sample_count = count
        values = self.live_filtered[self.live_count - count : self.live_count, probe]
        values = values - np.nanmean(values)
        scales = np.arange(1, 65)
        coefficients, frequencies = pywt.cwt(values, scales, "cmor1.5-1.0", sampling_period=self.dataset.settings.output_interval_s)
        power = np.abs(coefficients) ** 2
        self.cwt_plot.clear()
        order = np.argsort(frequencies)
        sorted_frequencies = np.asarray(frequencies)[order]
        image = pg.ImageItem(power[order].T)
        time_values = self.live_times[self.live_count - count : self.live_count]
        duration = max(float(time_values[-1] - time_values[0]), self.dataset.settings.output_interval_s)
        frequency_span = max(float(sorted_frequencies[-1] - sorted_frequencies[0]), 1.0)
        image.setRect(QRectF(float(time_values[0]), float(sorted_frequencies[0]), duration, frequency_span))
        self.cwt_plot.addItem(image)
        window_text = "当前分析窗" if retained_windows == 1 else f"最近{retained_windows}个相邻窗"
        self.cwt_plot.setTitle(f"小波分析 · {self.dataset.probe_names[probe]} · {window_text}")
        self.cwt_frequencies_hz = sorted_frequencies

    def _set_state(self, state: str, text: str) -> None:
        color = "#EF4444" if state == "ALARM" else "#22C55E" if state == "MONITORING" else "#94A3B8"
        self.alarm_light.setStyleSheet(f"font-size:72px;color:{color}")
        self.state_text.setText(text)

    def export_results(self) -> None:
        if self.dataset is None or self.live_count == 0:
            QMessageBox.information(self, "导出", "当前没有可导出的在线数据。")
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出监听结果", "stall_warning_result.csv", "CSV 数据 (*.csv)")
        if not path:
            return
        output: dict[str, Any] = {"step": self.live_steps[: self.live_count], "time_s": self.live_times[: self.live_count]}
        for probe, name in enumerate(self.dataset.probe_names):
            output[f"{name}_pressure"] = self.live_values[: self.live_count, probe]
            output[f"{name}_filtered"] = self.live_filtered[: self.live_count, probe]
            output[f"{name}_alarm"] = self.live_alarm_samples[: self.live_count, probe].astype(int)
        pd.DataFrame(output).to_csv(path, index=False, encoding="utf-8-sig")
        context_path = Path(path).with_suffix(".json")
        actuation_path: Path | None = None
        if self.actuator_controller.history:
            actuation_rows = [
                command.as_dict()
                for frame in self.actuator_controller.history
                for command in frame.commands
            ]
            actuation_path = Path(path).with_name(Path(path).stem + "_actuation.csv")
            pd.DataFrame(actuation_rows).to_csv(actuation_path, index=False, encoding="utf-8-sig")
        context_path.write_text(
            json.dumps(
                {
                    "data_source": asdict(self.dataset.settings),
                    "baseline": self.online_baseline_profile,
                    "active_control": {
                        "channel_count": self.actuator_controller.channel_count,
                        "channel_angles_deg": self.actuator_controller.channel_angles_deg.tolist(),
                        "actuation_log": None if actuation_path is None else actuation_path.name,
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        self.event_list.addItem(f"已导出：{Path(path).name}")

    def closeEvent(self, event) -> None:
        self.stop_sensor_listener(quiet=True)
        super().closeEvent(event)
