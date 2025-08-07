from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class AutoSliceDialog(QDialog):
    parameters_confirmed = Signal(dict)

    def __init__(self, parent=None, z_max_value=100):
        super().__init__(parent)
        self.setWindowTitle("Automatic Slice Detection")
        self.setLayout(QVBoxLayout())

        info_label = QLabel(
            "Please ensure that you have scaled your image (if necessary) "
            "before running automatic slice detection."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #ffffff; font-weight: bold;")
        self.layout().addWidget(info_label)

        form = QFormLayout()

        # z-range
        z_layout = QHBoxLayout()
        self.z_min = QSpinBox()
        self.z_max = QSpinBox()
        self.z_min.setRange(0, 9999)
        self.z_max.setRange(0, z_max_value)
        self.z_min.setValue(0)
        self.z_max.setValue(z_max_value)

        z_layout.addWidget(QLabel("Min:"))
        z_layout.addWidget(self.z_min)
        z_layout.addWidget(QLabel("Max:"))
        z_layout.addWidget(self.z_max)

        form.addRow("Z range:", z_layout)

        # pitch bounds
        pitch_layout = QHBoxLayout()
        self.pitch_min = QSpinBox()
        self.pitch_max = QSpinBox()
        self.pitch_min.setRange(-180, 180)
        self.pitch_max.setRange(-180, 180)
        self.pitch_min.setValue(-5)
        self.pitch_max.setValue(5)

        pitch_layout.addWidget(QLabel("Min:"))
        pitch_layout.addWidget(self.pitch_min)
        pitch_layout.addWidget(QLabel("Max:"))
        pitch_layout.addWidget(self.pitch_max)

        form.addRow("Pitch bounds (degrees):", pitch_layout)

        # yaw bounds
        yaw_layout = QHBoxLayout()
        self.yaw_min = QSpinBox()
        self.yaw_max = QSpinBox()
        self.yaw_min.setRange(-180, 180)
        self.yaw_max.setRange(-180, 180)
        self.yaw_min.setValue(-5)
        self.yaw_max.setValue(5)

        yaw_layout.addWidget(QLabel("Min:"))
        yaw_layout.addWidget(self.yaw_min)
        yaw_layout.addWidget(QLabel("Max:"))
        yaw_layout.addWidget(self.yaw_max)

        form.addRow("Yaw bounds (degrees):", yaw_layout)

        # roll bounds
        roll_layout = QHBoxLayout()
        self.roll_min = QSpinBox()
        self.roll_max = QSpinBox()
        self.roll_min.setRange(-180, 180)
        self.roll_max.setRange(-180, 180)
        self.roll_min.setValue(-5)
        self.roll_max.setValue(5)

        roll_layout.addWidget(QLabel("Min:"))
        roll_layout.addWidget(self.roll_min)
        roll_layout.addWidget(QLabel("Max:"))
        roll_layout.addWidget(self.roll_max)

        form.addRow("Roll bounds (degrees):", roll_layout)

        # init_points
        self.init_points = QSpinBox()
        self.init_points.setRange(1, 1000)
        self.init_points.setValue(5)
        form.addRow("Initial random points:", self.init_points)

        # n_iter
        self.n_iter = QSpinBox()
        self.n_iter.setRange(1, 1000)
        self.n_iter.setValue(15)
        form.addRow("Bayesian iterations:", self.n_iter)

        # similarity metric dropdown
        self.metric_dropdown = QComboBox()
        self.metric_dropdown.addItems(
            [
                "Mutual Information (recommended)",
                "Normalised Cross-Correlation",
                "Structural Similarity Index",
                "Combined",
            ]
        )
        self.metric_dropdown.currentTextChanged.connect(
            self.toggle_weights_row
        )
        form.addRow("Similarity metric:", self.metric_dropdown)

        # Combined metric weights (initially hidden)
        self.weights_layout = QHBoxLayout()
        self.mi_weight = QDoubleSpinBox()
        self.ncc_weight = QDoubleSpinBox()
        self.ssim_weight = QDoubleSpinBox()

        for spinbox, label, default in zip(
            [self.mi_weight, self.ncc_weight, self.ssim_weight],
            ["MI weight:", "NCC weight:", "SSIM weight:"],
            [0.7, 0.15, 0.15],
        ):
            spinbox.setRange(0.0, 9999)
            spinbox.setSingleStep(0.01)
            spinbox.setValue(default)
            self.weights_layout.addWidget(QLabel(label))
            self.weights_layout.addWidget(spinbox)

        self.weights_container = QWidget()
        self.weights_container.setLayout(self.weights_layout)

        self.weights_label = QLabel("Combined weights:")
        form.addRow(self.weights_label, self.weights_container)

        # Initially hide both label and widget
        self.weights_label.setVisible(False)
        self.weights_container.setVisible(False)

        self.layout().addLayout(form)

        # OK/Cancel buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.layout().addWidget(buttons)

    def toggle_weights_row(self, text):
        show = "Combined" in text
        self.weights_label.setVisible(show)
        self.weights_container.setVisible(show)

    def accept(self):
        # Map dropdown selection to internal code
        metric_map = {
            "Mutual Information (recommended)": "mi",
            "Normalised Cross-Correlation": "ncc",
            "Structural Similarity Index": "ssim",
            "Combined": "combined",
        }
        selected_metric = self.metric_dropdown.currentText()
        metric_value = metric_map.get(selected_metric, "mi")

        if metric_value == "combined":
            weights = tuple(
                round(val, 2)
                for val in (
                    self.mi_weight.value(),
                    self.ncc_weight.value(),
                    self.ssim_weight.value(),
                )
            )
        elif metric_value == "mi":
            weights = tuple((1.0, 0.0, 0.0))
        elif metric_value == "ncc":
            weights = tuple((0.0, 1.0, 0.0))
        elif metric_value == "ssim":
            weights = tuple((0.0, 0.0, 1.0))

        params = {
            "z_range": (self.z_min.value(), self.z_max.value()),
            "pitch_bounds": (self.pitch_min.value(), self.pitch_max.value()),
            "yaw_bounds": (self.yaw_min.value(), self.yaw_max.value()),
            "roll_bounds": (self.roll_min.value(), self.roll_max.value()),
            "init_points": self.init_points.value(),
            "n_iter": self.n_iter.value(),
            "metric": metric_value,
            "weights": weights,
        }

        self.parameters_confirmed.emit(params)
        super().accept()
