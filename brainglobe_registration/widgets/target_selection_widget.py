from qtpy.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QComboBox,
    QSpinBox, QDialogButtonBox, QHBoxLayout, QLabel
)
from qtpy.QtCore import Signal


class AutoSliceDialog(QDialog):
    parameters_confirmed = Signal(dict)

    def __init__(self,
                 atlas_names=None,
                 image_names=None,
                 parent=None,
                 z_max_value=100
    ):
        super().__init__(parent)
        self.setWindowTitle("Automatic Slice Detection")
        self.setLayout(QVBoxLayout())

        form = QFormLayout()

        # Atlas layer
        self.atlas_dropdown = QComboBox()
        self.atlas_dropdown.addItems(atlas_names or [])
        form.addRow("Atlas:", self.atlas_dropdown)

        # Sample (moving image)
        self.sample_dropdown = QComboBox()
        self.sample_dropdown.addItems(image_names or [])
        form.addRow("Sample image:", self.sample_dropdown)

        # Z-range
        z_layout = QHBoxLayout()
        self.z_min = QSpinBox()
        self.z_max = QSpinBox()
        self.z_min.setRange(0, 9999)
        self.z_max.setRange(0, 9999)
        self.z_min.setValue(0)

        # Set z_max using Napari viewer
        self.z_max.setMaximum(z_max_value)
        self.z_max.setValue(min(z_max_value, 100))

        z_layout.addWidget(QLabel("Min:"))
        z_layout.addWidget(self.z_min)
        z_layout.addWidget(QLabel("Max:"))
        z_layout.addWidget(self.z_max)

        form.addRow("Z range:", z_layout)

        self.layout().addLayout(form)

        # OK/Cancel buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.layout().addWidget(buttons)

    def accept(self):
        params = {
            "atlas_layer": self.atlas_dropdown.currentText(),
            "moving_image": self.sample_dropdown.currentText(),
            "z_range": (self.z_min.value(), self.z_max.value()),
        }
        self.parameters_confirmed.emit(params)
        super().accept()