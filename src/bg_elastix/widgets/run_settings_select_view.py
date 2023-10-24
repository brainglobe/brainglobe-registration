from qtpy.QtCore import Signal

from qtpy.QtWidgets import (
    QGroupBox,
    QCheckBox,
    QPushButton,
    QVBoxLayout,
    QComboBox,
)


class RunSettingsSelectView(QGroupBox):
    run_signal = Signal(bool, bool, bool, bool, str)
    default_file_signal = Signal(str)
    rigid_checkbox_signal = Signal(bool)
    affine_checkbox_signal = Signal(bool)
    bspline_checkbox_signal = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.file_options = ["elastix_default", "ara_tools", "brainregister_IBL"]

        self.setLayout(QVBoxLayout())

        self.rigid_checkbox = QCheckBox("Rigid")
        self.rigid_checkbox.setChecked(True)
        self.rigid_checkbox.stateChanged.connect(
            self._on_rigid_checkbox_change
        )

        self.affine_checkbox = QCheckBox("Affine")
        self.affine_checkbox.setChecked(True)
        self.affine_checkbox.stateChanged.connect(
            self._on_affine_checkbox_change
        )

        self.bspline_checkbox = QCheckBox("B-Spline")
        self.bspline_checkbox.setChecked(True)
        self.bspline_checkbox.stateChanged.connect(
            self._on_bspline_checkbox_change
        )

        self.select_default_file = QComboBox(parent=self)
        self.select_default_file.addItems(self.file_options)
        self.select_default_file.currentTextChanged.connect(
            self._on_default_file_change
        )

        self.run_button = QPushButton()
        self.run_button.setText("Run Registration")
        self.run_button.clicked.connect(self._on_run_button_click)

        self.layout().addWidget(self.rigid_checkbox)
        self.layout().addWidget(self.affine_checkbox)
        self.layout().addWidget(self.bspline_checkbox)
        self.layout().addWidget(self.select_default_file)
        self.layout().addWidget(self.run_button)

    def _on_run_button_click(self):
        self.run_signal.emit(
            self.rigid_checkbox.isChecked(),
            self.affine_checkbox.isChecked(),
            self.bspline_checkbox.isChecked(),
            self.use_default_params_checkbox.isChecked(),
            self.select_default_file.currentText(),
        )

    def _on_rigid_checkbox_change(self, state):
        self.rigid_checkbox_signal.emit(state != 0)

    def _on_affine_checkbox_change(self, state):
        self.affine_checkbox_signal.emit(state != 0)

    def _on_bspline_checkbox_change(self, state):
        self.bspline_checkbox_signal.emit(state != 0)

    def _on_default_file_change(self, default_type):
        self.default_file_signal.emit(default_type)
