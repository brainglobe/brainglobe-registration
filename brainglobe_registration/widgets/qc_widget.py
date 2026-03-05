"""Quality Control widget for registration validation."""

from qtpy.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class QCWidget(QWidget):
    """Widget containing QC options and actions."""

    def __init__(self, parent=None):
        """
        Initialize the QC widget.

        Parameters
        ----------
        parent : QWidget, optional
            The parent widget, by default None
        """
        super().__init__(parent=parent)

        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)

        # Checkerboard visualization checkbox (selection, not immediate action)
        self.checkerboard_checkbox = QCheckBox("Checkerboard")
        self.checkerboard_checkbox.setEnabled(False)
        self.checkerboard_checkbox.setChecked(False)
        self.checkerboard_checkbox.setToolTip(
            "Select to generate checkerboard visualization"
        )
        self.layout().addWidget(self.checkerboard_checkbox)

        self.intensity_map_checkbox = QCheckBox("Intensity Difference Map")
        self.intensity_map_checkbox.setEnabled(False)
        self.intensity_map_checkbox.setChecked(False)
        self.intensity_map_checkbox.setToolTip(
            "Select to generate intensity-difference visualization"
        )
        self.layout().addWidget(self.intensity_map_checkbox)

        # Checkerboard square size parameter
        square_size_layout = QHBoxLayout()
        square_size_layout.setContentsMargins(0, 0, 0, 0)
        square_size_label = QLabel("Square size:")
        self.square_size_spinbox = QSpinBox()
        self.square_size_spinbox.setRange(4, 512)
        self.square_size_spinbox.setValue(32)  # Default value
        self.square_size_spinbox.setSingleStep(4)
        self.square_size_spinbox.setEnabled(False)
        self.square_size_spinbox.setToolTip(
            "Size of each checkerboard square in pixels. "
            "Larger values create bigger squares."
        )
        square_size_layout.addWidget(square_size_label)
        square_size_layout.addWidget(self.square_size_spinbox)
        square_size_layout.addStretch()
        self.layout().addLayout(square_size_layout)

        # Plot QC button - generates all selected QC plots
        self.plot_qc_button = QPushButton("Plot QC")
        self.plot_qc_button.setEnabled(False)
        self.plot_qc_button.setToolTip(
            "Generate all selected QC visualizations"
        )
        self.layout().addWidget(self.plot_qc_button)

        # Clear QC images button
        self.clear_qc_button = QPushButton("Clear QC Images")
        self.clear_qc_button.setEnabled(False)
        self.clear_qc_button.setToolTip(
            "Remove all QC visualization layers (checkerboard, etc.)"
        )
        self.layout().addWidget(self.clear_qc_button)

    def set_enabled(self, enabled: bool):
        """Enable or disable all QC controls."""
        self.checkerboard_checkbox.setEnabled(enabled)
        self.intensity_map_checkbox.setEnabled(enabled)
        self.square_size_spinbox.setEnabled(enabled)
        self.plot_qc_button.setEnabled(enabled)
        self.clear_qc_button.setEnabled(enabled)
