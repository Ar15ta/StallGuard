from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from PySide6.QtCore import QPointF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..actuation import ActuatorController, ActuatorFrame
from ..stall_tracking import StallLocationEstimate


class ActuatorRingWidget(QWidget):
    selection_changed = Signal(int)

    def __init__(self, channel_count: int = 16, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.channel_count = int(channel_count)
        self.channel_angles_deg = np.arange(self.channel_count, dtype=float) * 360.0 / self.channel_count
        self.enabled = np.zeros(self.channel_count, dtype=bool)
        self.nozzle_offsets_deg = np.zeros(self.channel_count, dtype=float)
        self.active_indices: set[int] = set()
        self.target_angle_deg: float | None = None
        self.selected_index = 0
        self._actuator_positions: list[QPointF] = []
        self.setMinimumSize(250, 250)
        self.setToolTip("单击环上的动作机构进行设置；1号位于0°，角度沿顺时针增加。")

    def sizeHint(self) -> QSize:
        return QSize(280, 280)

    def set_selected(self, index: int) -> None:
        self.selected_index = int(np.clip(index, 0, self.channel_count - 1))
        self.update()

    def set_enabled_states(self, enabled: Sequence[bool]) -> None:
        values = np.asarray(enabled, dtype=bool)
        if values.shape != (self.channel_count,):
            raise ValueError("动作机构使能状态维度错误。")
        self.enabled = values.copy()
        self.update()

    def set_control_state(
        self,
        target_angle_deg: float | None,
        active_indices: Sequence[int],
        nozzle_offsets_deg: Sequence[float],
    ) -> None:
        offsets = np.asarray(nozzle_offsets_deg, dtype=float)
        if offsets.shape != (self.channel_count,):
            raise ValueError("动作机构角度维度错误。")
        self.target_angle_deg = None if target_angle_deg is None else float(target_angle_deg) % 360.0
        self.active_indices = {int(index) for index in active_indices}
        self.nozzle_offsets_deg = offsets.copy()
        self.update()

    @staticmethod
    def _unit_for_angle(angle_deg: float) -> QPointF:
        # Industrial annulus convention used by this panel: 0° at top, clockwise positive.
        radians = np.deg2rad(angle_deg)
        return QPointF(float(np.sin(radians)), float(-np.cos(radians)))

    @staticmethod
    def _rotated(vector: QPointF, angle_deg: float) -> QPointF:
        radians = np.deg2rad(angle_deg)
        cosine, sine = float(np.cos(radians)), float(np.sin(radians))
        return QPointF(cosine * vector.x() - sine * vector.y(), sine * vector.x() + cosine * vector.y())

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        center = QPointF(self.width() / 2.0, self.height() / 2.0)
        radius = max(52.0, min(self.width(), self.height()) * 0.31)
        painter.setPen(QPen(QColor("#F8FAFC"), 2.0))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(center, radius, radius)

        painter.setFont(QFont("Microsoft YaHei UI", 8))
        painter.setPen(QColor("#94A3B8"))
        painter.drawText(QPointF(center.x() - 10.0, center.y() - radius - 31.0), "0°")

        if self.target_angle_deg is not None:
            target_unit = self._unit_for_angle(self.target_angle_deg)
            target = center + target_unit * (radius * 0.42)
            painter.setPen(QPen(QColor("#EF4444"), 2.0))
            painter.setBrush(QColor(239, 68, 68, 80))
            painter.drawEllipse(target, 7.0, 7.0)
            painter.drawLine(target + QPointF(-11.0, 0.0), target + QPointF(11.0, 0.0))
            painter.drawLine(target + QPointF(0.0, -11.0), target + QPointF(0.0, 11.0))

        self._actuator_positions = []
        length, width = 24.0, 9.0
        for index, angle in enumerate(self.channel_angles_deg):
            radial = self._unit_for_angle(float(angle))
            position = center + radial * radius
            self._actuator_positions.append(position)
            inward = QPointF(-radial.x(), -radial.y())
            direction = self._rotated(inward, float(self.nozzle_offsets_deg[index]))
            magnitude = max(np.hypot(direction.x(), direction.y()), 1e-12)
            direction = QPointF(direction.x() / magnitude, direction.y() / magnitude)
            normal = QPointF(-direction.y(), direction.x())
            polygon = QPolygonF(
                [
                    position - direction * (length / 2.0) - normal * (width / 2.0),
                    position + direction * (length / 2.0) - normal * (width / 2.0),
                    position + direction * (length / 2.0) + normal * (width / 2.0),
                    position - direction * (length / 2.0) + normal * (width / 2.0),
                ]
            )
            is_active = index in self.active_indices
            if not self.enabled[index]:
                fill, outline = QColor(0, 0, 0, 165), QColor("#475569")
            elif is_active:
                fill, outline = QColor("#EF4444"), QColor("#FEE2E2")
            else:
                fill, outline = QColor("#F8FAFC"), QColor("#FFFFFF")
            if index == self.selected_index:
                outline = QColor("#F59E0B")
            painter.setPen(QPen(outline, 2.0 if index == self.selected_index else 1.2))
            painter.setBrush(fill)
            painter.drawPolygon(polygon)

            if is_active:
                inner_tip = position + direction * (length / 2.0)
                jet_color = QColor("#FCA5A5")
                painter.setPen(QPen(jet_color, 2.4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                for line_index in range(3):
                    start = inner_tip + direction * (5.0 + line_index * 7.0)
                    painter.drawLine(start, start + direction * 4.0)

            label_position = center + radial * (radius + 22.0)
            painter.setPen(QColor("#E5E7EB") if self.enabled[index] else QColor("#64748B"))
            painter.drawText(
                int(label_position.x() - 13.0),
                int(label_position.y() - 8.0),
                26,
                16,
                Qt.AlignmentFlag.AlignCenter,
                str(index + 1),
            )
        painter.end()

    def mousePressEvent(self, event) -> None:
        if self._actuator_positions:
            point = event.position()
            distances = [np.hypot(point.x() - item.x(), point.y() - item.y()) for item in self._actuator_positions]
            nearest = int(np.argmin(distances))
            if distances[nearest] <= 22.0:
                self.set_selected(nearest)
                self.selection_changed.emit(nearest)
        super().mousePressEvent(event)


class ActuatorPanel(QGroupBox):
    enabled_changed = Signal(int, bool)
    angle_changed = Signal(int, float)
    all_enabled_changed = Signal(bool)

    def __init__(self, controller: ActuatorController, parent: QWidget | None = None) -> None:
        super().__init__("主动控制 · 16通道", parent)
        self.controller = controller
        layout = QVBoxLayout(self)
        self.view_tabs = QTabWidget()
        self.view_tabs.setToolTip("“控制”用于配置和演示动作机构；“动作记录”用于查看各通道累计次数。")

        control_tab = QWidget()
        control_layout = QVBoxLayout(control_tab)
        self.ring = ActuatorRingWidget(controller.channel_count)
        self.ring.set_enabled_states(controller.enabled)
        self.ring.set_control_state(None, (), controller.manual_offsets_deg)
        self.ring.selection_changed.connect(self._select_from_ring)
        control_layout.addWidget(self.ring, 1)

        form = QFormLayout()
        self.channel_selector = QComboBox()
        for index, angle in enumerate(controller.channel_angles_deg):
            self.channel_selector.addItem(f"{index + 1}号 · {angle:.1f}°", index)
        self.channel_selector.setToolTip("选择需要配置的动作机构；也可直接单击环形图中的矩形。")
        self.channel_selector.currentIndexChanged.connect(self._channel_selected)
        self.output_enabled = QCheckBox("开启该通道输出")
        self.output_enabled.setToolTip("允许该动作机构接收主动控制指令；关闭时在环形图中显示为半透明黑色。")
        self.output_enabled.toggled.connect(self._enabled_toggled)
        self.manual_angle = QDoubleSpinBox()
        self.manual_angle.setRange(-180.0, 180.0)
        self.manual_angle.setDecimals(1)
        self.manual_angle.setSuffix(" °")
        self.manual_angle.setToolTip("待机方向相对指向圆心的半径方向偏角；正值沿屏幕顺时针旋转。主动控制时自动跟踪目标。")
        self.manual_angle.valueChanged.connect(self._angle_edited)
        self.pulse_duration_ms = QSpinBox()
        self.pulse_duration_ms.setRange(10, 5000)
        self.pulse_duration_ms.setValue(100)
        self.pulse_duration_ms.setSuffix(" ms")
        self.pulse_duration_ms.setToolTip("单次喷气指令的物理持续时间；每个脉冲结束后重新定位并选择最近的两个通道。")
        form.addRow("动作机构", self.channel_selector)
        form.addRow("输出状态", self.output_enabled)
        form.addRow("待机偏角", self.manual_angle)
        form.addRow("喷气持续", self.pulse_duration_ms)
        control_layout.addLayout(form)

        buttons = QHBoxLayout()
        enable_all = QPushButton("全部开启")
        disable_all = QPushButton("全部关闭")
        enable_all.setToolTip("开启全部16个软件输出通道。")
        disable_all.setToolTip("关闭全部通道并撤销正在执行的动作。")
        enable_all.clicked.connect(lambda: self._set_all(True))
        disable_all.clicked.connect(lambda: self._set_all(False))
        buttons.addWidget(enable_all)
        buttons.addWidget(disable_all)
        control_layout.addLayout(buttons)

        self.location_label = QLabel("失速团位置：等待判定")
        self.location_label.setObjectName("Muted")
        self.location_label.setWordWrap(True)
        self.feedback_label = QLabel("未输出主动控制指令")
        self.feedback_label.setObjectName("ControlFeedback")
        self.feedback_label.setWordWrap(True)
        control_layout.addWidget(self.location_label)
        control_layout.addWidget(self.feedback_label)

        history_tab = QWidget()
        history_layout = QVBoxLayout(history_tab)

        count_title = QLabel("各通道动作次数")
        count_title.setObjectName("Muted")
        count_title.setWordWrap(True)
        count_title.setText("各通道动作次数\n切换到本页可查看历史，计数仍在后台实时更新。")
        count_title.setToolTip("通道由未动作变为动作时计1次；同一脉冲内仅改变朝向不重复计数。")
        history_layout.addWidget(count_title)
        count_grid = QGridLayout()
        count_grid.setHorizontalSpacing(4)
        count_grid.setVerticalSpacing(4)
        self.count_labels: list[QLabel] = []
        for index in range(controller.channel_count):
            label = QLabel()
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setMinimumHeight(34)
            label.setToolTip(f"{index + 1}号动作机构累计启动喷气的次数。")
            count_grid.addWidget(label, index // 4, index % 4)
            self.count_labels.append(label)
        history_layout.addLayout(count_grid)
        history_layout.addStretch(1)

        self.view_tabs.addTab(control_tab, "控制")
        self.view_tabs.addTab(history_tab, "动作记录")
        layout.addWidget(self.view_tabs)
        self._refresh_counts(())
        self._channel_selected(0)

    def refresh_configuration(self) -> None:
        self.ring.set_enabled_states(self.controller.enabled)
        active = tuple(channel - 1 for channel in self.controller.last_frame.active_channels)
        offsets = [command.nozzle_offset_deg for command in self.controller.last_frame.commands]
        self.ring.set_control_state(self.controller.last_frame.target_angle_deg, active, offsets)
        self._refresh_counts(active)
        self._channel_selected(self.channel_selector.currentIndex())

    def show_control(
        self,
        estimate: StallLocationEstimate | None,
        frame: ActuatorFrame,
        pulse_sequence: int | None = None,
    ) -> None:
        active_indices = tuple(channel - 1 for channel in frame.active_channels)
        offsets = [command.nozzle_offset_deg for command in frame.commands]
        self.ring.set_enabled_states(self.controller.enabled)
        self.ring.set_control_state(frame.target_angle_deg, active_indices, offsets)
        self._refresh_counts(active_indices)
        if estimate is None:
            self.location_label.setText("失速团位置：等待判定")
            self.feedback_label.setText("未输出主动控制指令")
            return
        speed = "速度待定" if estimate.angular_speed_deg_s is None else f"{estimate.angular_speed_deg_s:+.1f} °/s"
        self.location_label.setText(
            f"预测位置 {estimate.angle_deg:.1f}° · {speed}\n"
            f"定位方式：{estimate.source} · 置信度 {estimate.confidence:.0%}"
        )
        if len(frame.active_channels) == 2:
            first, second = frame.active_channels
            sequence = "" if pulse_sequence is None else f" · 第{pulse_sequence}次"
            self.feedback_label.setText(
                f"{first}号、{second}号动作机构反馈控制中{sequence}\n"
                f"单次喷气 {self.pulse_duration_ms.value()} ms"
            )
        else:
            self.feedback_label.setText("可用动作机构不足2个，未输出")

    def _refresh_counts(self, active_indices: Sequence[int]) -> None:
        active = {int(index) for index in active_indices}
        for index, label in enumerate(self.count_labels):
            label.setText(f"{index + 1:02d}号\n{int(self.controller.activation_counts[index])}次")
            if index in active:
                label.setStyleSheet(
                    "background:#7F1D1D;color:#FEE2E2;border:1px solid #EF4444;border-radius:3px;"
                )
            else:
                label.setStyleSheet(
                    "background:#111827;color:#CBD5E1;border:1px solid #334155;border-radius:3px;"
                )

    def _select_from_ring(self, index: int) -> None:
        self.channel_selector.setCurrentIndex(index)

    def _channel_selected(self, index: int) -> None:
        if index < 0:
            return
        self.ring.set_selected(index)
        self.output_enabled.blockSignals(True)
        self.manual_angle.blockSignals(True)
        self.output_enabled.setChecked(bool(self.controller.enabled[index]))
        self.manual_angle.setValue(float(self.controller.manual_offsets_deg[index]))
        self.output_enabled.blockSignals(False)
        self.manual_angle.blockSignals(False)

    def _enabled_toggled(self, enabled: bool) -> None:
        index = self.channel_selector.currentIndex()
        if index >= 0:
            self.enabled_changed.emit(index, enabled)

    def _angle_edited(self, angle_deg: float) -> None:
        index = self.channel_selector.currentIndex()
        if index >= 0:
            self.angle_changed.emit(index, angle_deg)

    def _set_all(self, enabled: bool) -> None:
        self.all_enabled_changed.emit(enabled)
