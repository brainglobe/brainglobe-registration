from qtpy.QtCore import Qt
from qtpy.QtWidgets import QLabel, QVBoxLayout, QWidget
from superqt import QRangeSlider


class CropAtlasView(QWidget):
    def __init__(self, parent: QWidget = None):
        super().__init__(parent)

        self.setLayout(QVBoxLayout())
        self.z_slider = QRangeSlider(Qt.Orientation.Horizontal)
        self.z_slider.setValue([0, self.z_slider.maximum()])
        self.y_slider = QRangeSlider(Qt.Orientation.Horizontal)
        self.y_slider.setValue([0, self.y_slider.maximum()])
        self.x_slider = QRangeSlider(Qt.Orientation.Horizontal)
        self.x_slider.setValue([0, self.x_slider.maximum()])

        self.x_slider.valueChanged.connect(self._on_change)

        self.layout().addWidget(QLabel("Crop Z", parent=self))
        self.layout().addWidget(self.z_slider)
        self.layout().addWidget(QLabel("Crop Y", parent=self))
        self.layout().addWidget(self.y_slider)
        self.layout().addWidget(QLabel("Crop X", parent=self))
        self.layout().addWidget(self.x_slider)

    def update_slider_ranges(self, atlas_shape):
        self.z_slider.setRange(0, atlas_shape[0])
        self.z_slider.setValue([0, atlas_shape[0]])
        self.y_slider.setRange(0, atlas_shape[1])
        self.y_slider.setValue([0, atlas_shape[1]])
        self.x_slider.setRange(0, atlas_shape[2])
        self.x_slider.setValue([0, atlas_shape[2]])

    def _on_change(self):
        print(
            (
                self.z_slider.value(),
                self.y_slider.value(),
                self.x_slider.value(),
            )
        )
