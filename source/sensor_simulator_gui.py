from __future__ import annotations

import socket
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from simulate_sensor import default_csv_path, send_message
from src.config import load_defaults, resource_path
from src.io_csv import load_cfd_csv
from src.models import DataSourceSettings


class SensorSendWorker(QThread):
    progress = Signal(int, int)
    status = Signal(str)
    failed = Signal(str)
    completed = Signal(bool)

    def __init__(
        self,
        csv_path: Path,
        settings: DataSourceSettings,
        host: str,
        port: int,
        rate: float,
        batch_size: int,
        start_step: float | None,
        end_step: float | None,
        max_rows: int | None,
    ) -> None:
        super().__init__()
        self.csv_path = csv_path
        self.settings = settings
        self.host = host
        self.port = int(port)
        self.rate = float(rate)
        self.batch_size = int(batch_size)
        self.start_step = start_step
        self.end_step = end_step
        self.max_rows = max_rows

    def run(self) -> None:
        try:
            dataset = load_cfd_csv(self.csv_path, self.settings)
            mask = np.ones(dataset.sample_count, dtype=bool)
            if self.start_step is not None:
                mask &= dataset.steps >= self.start_step
            if self.end_step is not None:
                mask &= dataset.steps <= self.end_step
            indices = np.flatnonzero(mask)
            if self.max_rows is not None:
                indices = indices[: self.max_rows]
            if indices.size == 0:
                raise ValueError("指定范围内没有可发送的数据。")
            self.status.emit(
                f"已读取 {dataset.sample_count:,} 行、{dataset.probe_count} 个探针；"
                f"连接 {self.host}:{self.port}"
            )
            with socket.create_connection((self.host, self.port), timeout=10.0) as connection:
                connection.settimeout(10.0)
                send_message(
                    connection,
                    {
                        "type": "meta",
                        "probe_names": dataset.probe_names,
                        "output_interval_s": dataset.settings.output_interval_s,
                        "total_rows": int(indices.size),
                    },
                )
                self.progress.emit(0, int(indices.size))
                for offset in range(0, indices.size, self.batch_size):
                    if self.isInterruptionRequested():
                        self.status.emit("用户停止发送。")
                        self.completed.emit(False)
                        return
                    selected = indices[offset : offset + self.batch_size]
                    send_message(
                        connection,
                        {
                            "type": "samples",
                            "steps": dataset.steps[selected].tolist(),
                            "values": dataset.values[selected].tolist(),
                        },
                    )
                    sent = offset + selected.size
                    self.progress.emit(int(sent), int(indices.size))
                    if self.rate > 0 and not self._interruptible_wait(selected.size / self.rate):
                        self.status.emit("用户停止发送。")
                        self.completed.emit(False)
                        return
            self.status.emit(f"发送完成，共 {indices.size:,} 行。")
            self.completed.emit(True)
        except Exception:
            self.failed.emit(traceback.format_exc())
            self.completed.emit(False)

    def _interruptible_wait(self, duration_s: float) -> bool:
        deadline = time.monotonic() + max(0.0, duration_s)
        while time.monotonic() < deadline:
            if self.isInterruptionRequested():
                return False
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        return True


class SensorSimulatorWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        defaults = load_defaults()["data_source"]
        self.defaults = defaults
        self.worker: SensorSendWorker | None = None
        self.setWindowTitle("StallGuard · 传感器模拟器")
        self.resize(650, 620)

        root = QWidget()
        layout = QVBoxLayout(root)
        title = QLabel("CSV 多探针数据发送器")
        title.setObjectName("ProductTitle")
        subtitle = QLabel("选择任意兼容CSV，模拟在线采集并发送到失速预警程序")
        subtitle.setObjectName("Muted")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        file_row = QHBoxLayout()
        self.csv_path = QLineEdit(str(default_csv_path()))
        self.csv_path.setToolTip("选择第一列为时间步、其余数值列为探针数据的CSV文件。")
        browse = QPushButton("选择CSV…")
        browse.clicked.connect(self._browse_csv)
        file_row.addWidget(self.csv_path, 1)
        file_row.addWidget(browse)
        layout.addLayout(file_row)

        form = QFormLayout()
        self.output_interval = QDoubleSpinBox()
        self.output_interval.setDecimals(12)
        self.output_interval.setRange(1e-12, 1000.0)
        self.output_interval.setValue(float(defaults["output_interval_s"]))
        self.output_interval.setSuffix(" s")
        self.output_interval.setToolTip("相邻累计时间步对应的物理时间；必须与预警程序初始化时填写的值一致。")
        self.host = QLineEdit("127.0.0.1")
        self.port = QSpinBox()
        self.port.setRange(1, 65535)
        self.port.setValue(8765)
        self.rate = QDoubleSpinBox()
        self.rate.setRange(0.0, 1e9)
        self.rate.setDecimals(2)
        self.rate.setValue(3000.0)
        self.rate.setSuffix(" 行/秒")
        self.rate.setToolTip("模拟器墙钟发送速度；设为0表示不限速，不改变报文中的物理采样间隔。")
        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 100000)
        self.batch_size.setValue(50)
        self.batch_size.setToolTip("一条TCP消息包含的CSV行数；批量发送可提高高采样率数据的吞吐量。")
        self.start_step = QLineEdit()
        self.start_step.setPlaceholderText("留空表示从文件首行开始")
        self.start_step.setText("88000")
        self.end_step = QLineEdit()
        self.end_step.setPlaceholderText("留空表示发送到文件末尾")
        self.max_rows = QSpinBox()
        self.max_rows.setRange(0, 2_000_000_000)
        self.max_rows.setValue(0)
        self.max_rows.setSpecialValueText("不限")
        form.addRow("响应输出间隔", self.output_interval)
        form.addRow("目标地址", self.host)
        form.addRow("目标端口", self.port)
        form.addRow("发送速度", self.rate)
        form.addRow("每批行数", self.batch_size)
        form.addRow("起始时间步", self.start_step)
        form.addRow("结束时间步", self.end_step)
        form.addRow("最多发送", self.max_rows)
        layout.addLayout(form)

        buttons = QHBoxLayout()
        self.inspect_button = QPushButton("检查文件")
        self.inspect_button.clicked.connect(self._inspect_csv)
        self.start_button = QPushButton("开始输出")
        self.start_button.setObjectName("Primary")
        self.start_button.clicked.connect(self._start)
        self.stop_button = QPushButton("停止")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._stop)
        buttons.addWidget(self.inspect_button)
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.stop_button)
        layout.addLayout(buttons)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        layout.addWidget(self.progress)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("文件检查、连接状态和发送进度将在这里显示。")
        layout.addWidget(self.log, 1)
        self.setCentralWidget(root)

    def _settings(self) -> DataSourceSettings:
        return DataSourceSettings(
            output_interval_s=self.output_interval.value(),
            step_origin=float(self.defaults["step_origin"]),
            time_origin_s=float(self.defaults["time_origin_s"]),
            first_row_zero=True,
            start_angle_deg=float(self.defaults["start_angle_deg"]),
            angle_direction=int(self.defaults["angle_direction"]),
        )

    def _browse_csv(self) -> None:
        current = Path(self.csv_path.text().strip())
        initial = str(current.parent if current.parent.is_dir() else Path.cwd())
        path, _ = QFileDialog.getOpenFileName(self, "选择探针CSV", initial, "CSV 数据 (*.csv);;所有文件 (*)")
        if path:
            self.csv_path.setText(path)
            self._inspect_csv()

    def _inspect_csv(self) -> bool:
        try:
            dataset = load_cfd_csv(self.csv_path.text().strip(), self._settings())
            self._append(
                f"文件有效：{dataset.path.name}\n"
                f"  行数：{dataset.sample_count:,}\n"
                f"  探针：{dataset.probe_count} 路\n"
                f"  时间步：{dataset.steps[0]:g} ～ {dataset.steps[-1]:g}\n"
                f"  单位：{dataset.unit}"
            )
            return True
        except Exception as exc:
            self._append(f"文件检查失败：{exc}")
            QMessageBox.critical(self, "文件检查失败", str(exc))
            return False

    @staticmethod
    def _optional_float(widget: QLineEdit, label: str) -> float | None:
        text = widget.text().strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError as exc:
            raise ValueError(f"{label}必须是数值或留空。") from exc

    def _start(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            return
        try:
            path = Path(self.csv_path.text().strip()).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"文件不存在：{path}")
            start_step = self._optional_float(self.start_step, "起始时间步")
            end_step = self._optional_float(self.end_step, "结束时间步")
            if start_step is not None and end_step is not None and end_step < start_step:
                raise ValueError("结束时间步不能小于起始时间步。")
            worker = SensorSendWorker(
                path,
                self._settings(),
                self.host.text().strip() or "127.0.0.1",
                self.port.value(),
                self.rate.value(),
                self.batch_size.value(),
                start_step,
                end_step,
                None if self.max_rows.value() == 0 else self.max_rows.value(),
            )
            worker.progress.connect(self._update_progress)
            worker.status.connect(self._append)
            worker.failed.connect(self._worker_failed)
            worker.completed.connect(self._worker_completed)
            worker.finished.connect(self._worker_finished)
            self.worker = worker
            self.start_button.setEnabled(False)
            self.inspect_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self._append(f"开始输出：{path.name}")
            worker.start()
        except Exception as exc:
            QMessageBox.critical(self, "无法开始输出", str(exc))

    def _stop(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.worker.requestInterruption()
            self._append("正在停止……")
            self.stop_button.setEnabled(False)

    def _update_progress(self, sent: int, total: int) -> None:
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(sent)
        self.progress.setFormat(f"已发送 {sent:,}/{total:,} 行")

    def _append(self, message: str) -> None:
        self.log.append(message)

    def _worker_failed(self, details: str) -> None:
        summary = details.strip().splitlines()[-1] if details.strip() else "未知错误"
        self._append("发送失败：" + summary)
        QMessageBox.critical(self, "发送失败", summary)

    def _worker_completed(self, success: bool) -> None:
        if success:
            self.progress.setValue(self.progress.maximum())

    def _worker_finished(self) -> None:
        self.start_button.setEnabled(True)
        self.inspect_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.worker = None

    def closeEvent(self, event) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.worker.requestInterruption()
            self.worker.wait(1500)
        super().closeEvent(event)


def main() -> int:
    # Keep the historical command-line sender available for scripts. A plain
    # double-click (no arguments) opens the GUI.
    if len(sys.argv) > 1 and "--smoke-test" not in sys.argv:
        from simulate_sensor import main as command_line_main

        return command_line_main()
    app = QApplication(sys.argv)
    app.setApplicationName("SensorSimulator")
    app.setOrganizationName("StallGuard")
    app.setStyle("Fusion")
    try:
        app.setStyleSheet(resource_path("assets/theme.qss").read_text(encoding="utf-8"))
    except OSError:
        pass
    window = SensorSimulatorWindow()
    if "--smoke-test" in sys.argv:
        index = sys.argv.index("--smoke-test")
        if index + 1 < len(sys.argv):
            window.csv_path.setText(sys.argv[index + 1])
        return 0 if window._inspect_csv() else 1
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
