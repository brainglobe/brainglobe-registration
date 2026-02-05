"""
Quality Control widget for registration validation.

This widget provides QC visualizations and controls for assessing
registration quality, including checkerboard visualization and future features
such as intensity maps and Jacobian analysis.
"""

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
    """
    A QWidget subclass that provides Quality Control (QC) functionality
    for registration validation.

    This widget contains controls and visualizations for assessing
    registration quality, such as checkerboard patterns, intensity maps,
    and other QC metrics.

    Attributes
    ----------
    checkerboard_checkbox : QCheckBox
        Checkbox to select checkerboard visualization.
    plot_qc_button : QPushButton
        Button to generate selected QC plots.
    clear_qc_button : QPushButton
        Button to clear all QC visualizations.

    Methods
    -------
    set_enabled(enabled: bool)
        Enable or disable all QC controls.
    """

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
        """
        Enable or disable all QC controls.

        Parameters
        ----------
        enabled : bool
            Whether to enable the QC controls.
        """
        self.checkerboard_checkbox.setEnabled(enabled)
        self.square_size_spinbox.setEnabled(enabled)
        self.plot_qc_button.setEnabled(enabled)
        self.clear_qc_button.setEnabled(enabled)
