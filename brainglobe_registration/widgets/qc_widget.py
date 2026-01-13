"""
Quality Control widget for registration validation.

This widget provides QC visualizations and controls for assessing
registration quality, including checkerboard visualization and future features
such as intensity maps and Jacobian analysis.
"""

from qtpy.QtWidgets import QCheckBox, QVBoxLayout, QWidget


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
        Checkbox to toggle checkerboard visualization.

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

        # Checkerboard visualization checkbox
        self.checkerboard_checkbox = QCheckBox("Show Checkerboard")
        self.checkerboard_checkbox.setChecked(False)
        self.checkerboard_checkbox.setEnabled(False)
        self.layout().addWidget(self.checkerboard_checkbox)

        # Future QC features will be added here (intensity map, etc.)

    def set_enabled(self, enabled: bool):
        """
        Enable or disable all QC controls.

        Parameters
        ----------
        enabled : bool
            Whether to enable the QC controls.
        """
        self.checkerboard_checkbox.setEnabled(enabled)
