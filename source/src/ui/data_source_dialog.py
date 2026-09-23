from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QMessageBox,
    QVBoxLayout,
)

from ..models import DataSourceSettings, ProbeDataset


class DataSourceDialog(QDialog):
    def __init__(self, defaults: dict, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("CSV 数据映射")
        self.setMinimumWidth(480)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("指定传感器/CFD 探针的响应输出间隔，并确认周向等角度映射。"))
        form = QFormLayout()
        self.interval = QDoubleSpinBox()
        self.interval.setDecimals(12)
        self.interval.setRange(1e-12, 1e6)
        self.interval.setValue(float(defaults.get("output_interval_s", 5.76928006151677e-6)))
        self.interval.setSuffix(" s")
        self.interval.setToolTip("相邻两行探针数据对应的物理时间间隔，单位为秒。")
        self.first_zero = QCheckBox("监听点首个样本时间归 0")
        self.first_zero.setChecked(bool(defaults.get("first_row_zero", True)))
        self.first_zero.setToolTip("启用后以所选数据段的首行作为 0 s，便于不同采集段对齐。")
        self.start_angle = QDoubleSpinBox()
        self.start_angle.setRange(0.0, 359.999)
        self.start_angle.setDecimals(3)
        self.start_angle.setSuffix(" °")
        self.start_angle.setValue(float(defaults.get("start_angle_deg", 0.0)))
        self.start_angle.setToolTip("第一个探针对应的周向角度；其他探针按 360°/探针数等间隔排列。")
        self.direction = QComboBox()
        self.direction.addItem("角度递增", 1)
        self.direction.addItem("角度递减", -1)
        self.direction.setToolTip("探针编号沿周向的排列方向。")
        form.addRow("响应输出间隔", self.interval)
        form.addRow("时间原点", self.first_zero)
        form.addRow("首探针角度", self.start_angle)
        form.addRow("周向方向", self.direction)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def settings(self) -> DataSourceSettings:
        return DataSourceSettings(
            output_interval_s=self.interval.value(),
            first_row_zero=self.first_zero.isChecked(),
            start_angle_deg=self.start_angle.value(),
            angle_direction=int(self.direction.currentData()),
        )


def confirm_dataset_mapping(dataset: ProbeDataset, parent=None) -> bool:
    details = (
        f"样本数：{dataset.sample_count:,}\n"
        f"探针数：{dataset.probe_count}\n"
        f"周向间隔：{360.0 / dataset.probe_count:.3f}°\n"
        f"物理时长：{dataset.time_s[-1] - dataset.time_s[0]:.6f} s\n"
        f"单位：{dataset.unit}"
    )
    answer = QMessageBox.question(
        parent,
        "确认数据映射",
        details + "\n\n是否使用该数据段进行初始化标定？",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
    )
    return answer == QMessageBox.StandardButton.Yes
