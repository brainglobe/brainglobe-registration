"""
QC (Quality Control) widget for registration visualizations.

Provides checkboxes for QC plot types and a "Plot QC" button so the user
selects what to show, then triggers computation/display with one action.
"""

from qtpy.QtWidgets import (
    QCheckBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class QCWidget(QWidget):
    """Widget for QC visualization options and actions."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)

        self.intensity_map_checkbox = QCheckBox("Intensity Difference Map")
        self.intensity_map_checkbox.setToolTip(
            "Show pixel-wise absolute difference between atlas and "
            "registered image as a heatmap (dark=low, bright=high). "
            "Use Plot QC to generate."
        )
        self.layout().addWidget(self.intensity_map_checkbox)

        self.plot_qc_button = QPushButton("Plot QC")
        self.plot_qc_button.setToolTip(
            "Generate selected QC visualizations (e.g. intensity map)."
        )
        self.layout().addWidget(self.plot_qc_button)

        self.clear_qc_button = QPushButton("Clear QC Images")
        self.clear_qc_button.setToolTip(
            "Remove QC layers and clear selection."
        )
        self.layout().addWidget(self.clear_qc_button)

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable all QC controls."""
        self.intensity_map_checkbox.setEnabled(enabled)
        self.plot_qc_button.setEnabled(enabled)
        self.clear_qc_button.setEnabled(enabled)
