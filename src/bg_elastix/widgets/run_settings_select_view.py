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

    def __init__(self, parent=None):
        super().__init__(parent)
        self.file_options = ["ARA Tools", "BrainRegister IBL"]

        self.setLayout(QVBoxLayout())

        self.rigid_checkbox = QCheckBox("Rigid")
        self.rigid_checkbox.setChecked(True)

        self.affine_checkbox = QCheckBox("Affine")
        self.affine_checkbox.setChecked(True)

        self.bspline_checkbox = QCheckBox("B-Spline")
        self.bspline_checkbox.setChecked(True)

        self.select_file = QComboBox(parent=self)
        self.select_file.addItems(self.file_options)
        self.select_file.setHidden(True)

        self.use_default_params_checkbox = QCheckBox("Default Params")
        self.use_default_params_checkbox.stateChanged.connect(
            self._on_default_param_checkbox_change
        )
        self.use_default_params_checkbox.setChecked(True)

        self.run_button = QPushButton()
        self.run_button.setText("Run Registration")
        self.run_button.clicked.connect(self._on_run_button_click)

        self.layout().addWidget(self.rigid_checkbox)
        self.layout().addWidget(self.affine_checkbox)
        self.layout().addWidget(self.bspline_checkbox)
        self.layout().addWidget(self.use_default_params_checkbox)
        self.layout().addWidget(self.select_file)
        self.layout().addWidget(self.run_button)

    def _on_run_button_click(self):
        self.run_signal.emit(
            self.rigid_checkbox.isChecked(),
            self.affine_checkbox.isChecked(),
            self.bspline_checkbox.isChecked(),
            self.use_default_params_checkbox.isChecked(),
            self.select_file.currentText(),
        )

    def _on_default_param_checkbox_change(self, state):
        if state == 0:
            self.select_file.setHidden(False)
        elif state == 2:
            self.select_file.setHidden(True)

