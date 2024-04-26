from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QLabel, QVBoxLayout, QWidget
from superqt import QLabeledRangeSlider


class CropAtlasView(QWidget):
    crop_z_signal = Signal(int, int)
    crop_y_signal = Signal(int, int)
    crop_x_signal = Signal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setLayout(QVBoxLayout())

        self.x_slider = QLabeledRangeSlider(Qt.Orientation.Horizontal)
        self.y_slider = QLabeledRangeSlider(Qt.Orientation.Horizontal)
        self.z_slider = QLabeledRangeSlider(Qt.Orientation.Horizontal)

        self.set_ranges(100, 100, 100)

        self.layout().addWidget(QLabel("Crop X"))
        self.layout().addWidget(self.x_slider)
        self.layout().addWidget(QLabel("Crop Y"))
        self.layout().addWidget(self.y_slider)
        self.layout().addWidget(QLabel("Crop Z"))
        self.layout().addWidget(self.z_slider)

        self.x_slider.sliderMoved.connect(self._on_crop_x)
        self.y_slider.sliderMoved.connect(self._on_crop_y)
        self.z_slider.sliderMoved.connect(self._on_crop_z)
        # Emits twice, but if not connected, editing text doesn't emit signal
        # via sliderMoved
        self.x_slider.editingFinished.connect(self._on_crop_x)
        self.y_slider.editingFinished.connect(self._on_crop_y)
        self.z_slider.editingFinished.connect(self._on_crop_z)

    def set_ranges(self, x_max, y_max, z_max):
        self.x_slider.setRange(0, x_max - 1)
        self.x_slider.setValue((0, x_max - 1))
        self.y_slider.setRange(0, y_max - 1)
        self.y_slider.setValue((0, y_max - 1))
        self.z_slider.setRange(0, z_max - 1)
        self.z_slider.setValue((0, z_max - 1))

    def _on_crop_x(self):
        self.crop_x_signal.emit(
            self.x_slider.value()[0], self.x_slider.value()[1]
        )

    def _on_crop_y(self):
        self.crop_y_signal.emit(
            self.y_slider.value()[0], self.y_slider.value()[1]
        )

    def _on_crop_z(self):
        self.crop_z_signal.emit(
            self.z_slider.value()[0], self.z_slider.value()[1]
        )
