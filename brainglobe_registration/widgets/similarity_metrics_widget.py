from qt_niu.dialog import display_warning
from qtpy.QtCore import (
    Signal,  ## https://www.tutorialspoint.com/pyqt/pyqt_new_signals_with_pyqtsignal.htm
)
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
)

AVAILABLE_METRICS = ["ncc", "mi", "ssim"]  # from similarity metrics code


class SimilarityWidget(QGroupBox):
    calculate_metric_requested = Signal(object, object, str)

    def __init__(self, parent=None):
        super().__init__("Find Best Slice", parent=parent)

        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Maximum)

        ##Slice range
        self.custom_range_checkbox = QCheckBox("Use Custom Slice Range")
        self.start_slice_spinbox = QSpinBox()
        self.start_slice_spinbox.setEnabled(False)
        self.end_slice_spinbox = QSpinBox()
        self.end_slice_spinbox.setEnabled(False)

        slice_range_layout = QHBoxLayout()
        slice_range_layout.addWidget(QLabel("Start:"))
        slice_range_layout.addWidget(self.start_slice_spinbox)
        slice_range_layout.addSpacing(20)
        slice_range_layout.addWidget(QLabel("End:"))
        slice_range_layout.addWidget(self.end_slice_spinbox)
        slice_range_layout.addStretch()

        ##Similarity metric
        self.metric_combobox = QComboBox()
        for metric_name in AVAILABLE_METRICS:
            self.metric_combobox.addItem(metric_name)

        metric_layout = QHBoxLayout()
        metric_layout.addWidget(QLabel("Similarity Metric:"))
        metric_layout.addWidget(self.metric_combobox)
        metric_layout.addStretch()

        ##Find button
        self.find_button = QPushButton("Find Best Matching Slice")

        ##Layout
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.custom_range_checkbox)
        main_layout.addLayout(slice_range_layout)
        main_layout.addLayout(metric_layout)
        main_layout.addWidget(self.find_button)

        # Connect
        self.custom_range_checkbox.toggled.connect(
            self._on_custom_range_toggled
        )
        self.find_button.clicked.connect(self._on_find_button_clicked)

        # Ensure spinboxes are disabled initially (controlled by checkbox)
        self._on_custom_range_toggled(self.custom_range_checkbox.isChecked())

    def _on_custom_range_toggled(self, checked: bool):
        self.start_slice_spinbox.setEnabled(checked)
        self.end_slice_spinbox.setEnabled(checked)

    def _on_group_toggled(self, checked):
        """Enable/disable widgets when the group box is toggled."""
        # Iterate through layouts and their widgets-- maybe not the best way
        # to do this need a different approach
        layout = self.layout()
        if layout:
            for i in range(layout.count()):
                item = layout.itemAt(i)
                if item:
                    widget = item.widget()
                    # Enable/disable actual widgets within the group
                    if widget and isinstance(
                        widget,
                        (QCheckBox, QSpinBox, QLabel, QComboBox, QPushButton),
                    ):
                        widget.setEnabled(checked)
                    # sub-layouts handling
                    elif item.layout() is not None:
                        sub_layout = item.layout()
                        for j in range(sub_layout.count()):
                            sub_item = sub_layout.itemAt(j)
                            if sub_item and sub_item.widget():
                                # Check type again for widgets in sublayouts
                                if isinstance(
                                    sub_item.widget(),
                                    (
                                        QCheckBox,
                                        QSpinBox,
                                        QLabel,
                                        QComboBox,
                                        QPushButton,
                                    ),
                                ):
                                    sub_item.widget().setEnabled(checked)

        # Ensure custom range widgets are correctly enabled/disabled
        if checked:
            self._on_custom_range_toggled(
                self.custom_range_checkbox.isChecked()
            )
        else:
            self.start_slice_spinbox.setEnabled(False)
            self.end_slice_spinbox.setEnabled(False)

    def set_slice_range_limits(self, min_slice, max_slice):
        """Sets the minimum and maximum values for the slice spinboxes."""
        # Ensure max >= min you can't put value more than max slice
        if max_slice < min_slice:
            max_slice = min_slice

        self.start_slice_spinbox.setMinimum(min_slice)
        self.start_slice_spinbox.setMaximum(max_slice)
        self.end_slice_spinbox.setMinimum(min_slice)
        self.end_slice_spinbox.setMaximum(max_slice)

        # Set initial values to the full range
        self.start_slice_spinbox.setValue(min_slice)
        self.end_slice_spinbox.setValue(max_slice)

    def _on_find_button_clicked(self):
        """Emit signal with selected metric and slice range."""
        if self.custom_range_checkbox.isChecked():
            start = self.start_slice_spinbox.value()
            end = self.end_slice_spinbox.value()
        else:
            start = self.start_slice_spinbox.minimum()
            end = self.end_slice_spinbox.maximum()

        metric = self.metric_combobox.currentText()

        # Use display_warning for user-facing error messages
        if start is None or end is None:
            display_warning(
                self, "Slice Range Error", "Could not determine slice range."
            )
            return
        if start > end:
            display_warning(
                self,
                "Slice Range Error",
                "Start slice cannot be greater than end slice.",
            )
            return

        self.calculate_metric_requested.emit(start, end, metric)
