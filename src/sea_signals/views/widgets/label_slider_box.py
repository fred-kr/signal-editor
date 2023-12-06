from typing import Sequence

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QSpinBox, QSlider, QWidget, QLabel, QGridLayout


class LabelSliderBox(QWidget):
    def __init__(self, label: str, limits: tuple[int | float, int | float], step: int | float, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.label_text = label
        self.limits = limits
        self.step = step

        self.init_ui()

    def init_ui(self) -> None:
        layout = QGridLayout(self)

        self.label = QLabel(self.label_text, self)
        layout.addWidget(self.label, 0, 0)

        self.slider = QSlider(self)
        self.slider.setOrientation(Qt.Orientation.Horizontal)
        self.slider.setMinimum(int(self.limits[0]))
        self.slider.setMaximum(int(self.limits[1]))
        self.slider.setSingleStep(int(self.step))
        layout.addWidget(self.slider, 1, 0)

        self.spin_box = QSpinBox(self)
        self.spin_box.setMinimum(int(self.limits[0]))
        self.spin_box.setMaximum(int(self.limits[1]))
        self.spin_box.setSingleStep(int(self.step))
        layout.addWidget(self.spin_box, 1, 1)

        self.slider.valueChanged.connect(self.spin_box.setValue)
        self.spin_box.valueChanged.connect(self.slider.setValue)

        self.setLayout(layout)

    def set_value(self, value: int | float) -> None:
        self.slider.setValue(int(value))
        self.spin_box.setValue(int(value))

    def get_value(self) -> int:
        return self.spin_box.value()

    def get_slider_value(self) -> int | float:
        return self.slider.value()

    def set_limits(self, limits: Sequence[int | float]) -> None:
        self.slider.setMinimum(int(limits[0]))
        self.slider.setMaximum(int(limits[1]))
        self.spin_box.setMinimum(int(limits[0]))
        self.spin_box.setMaximum(int(limits[1]))

    def set_step(self, step: int | float) -> None:
        self.slider.setSingleStep(int(step))
        self.spin_box.setSingleStep(int(step))
