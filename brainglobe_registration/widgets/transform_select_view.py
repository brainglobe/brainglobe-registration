from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QComboBox,
    QTableWidget,
)


class TransformSelectView(QTableWidget):
    """
    A QTableWidget subclass that provides a user interface for selecting
    transform types and associated files.

    This widget displays a table of available transform types and associated
    default parameter files. The user can select a transform type and an
    associated file from dropdown menus. The widget emits signals when a
    transform type is added or removed, or when a file option is changed.

    Attributes
    ----------
    transform_type_added_signal : Signal
        Emitted when a transform type is added. The signal includes the name
        of the transform type and its index.
    transform_type_removed_signal : Signal
        Emitted when a transform type is removed. The signal includes the
        index of the removed transform type.
    file_option_changed_signal : Signal
        Emitted when a file option is changed. The signal includes the name
        of the file and its index.

    Methods
    -------
    _on_transform_type_change(index):
        Handles the event when a transform type is changed. Emits the
        transform_type_added_signal or transform_type_removed_signal.
    _on_file_change(index):
        Handles the event when the default file option is changed.
        Emits the file_option_changed_signal.
    """

    transform_type_added_signal = Signal(str, int)
    transform_type_removed_signal = Signal(int)
    file_option_changed_signal = Signal(str, int)

    def __init__(self, parent=None):
        """
        Initialize the widget.

        Parameters
        ----------
        parent : QWidget, optional
            The parent widget, by default None
        """
        super().__init__(parent=parent)

        # Define the available transform types and file options
        self.file_options = [
            "elastix_default",
            "brainglobe_registration",
            "ara_tools",
            "brainregister_IBL",
        ]
        self.transform_type_options = ["", "affine", "bspline"]

        # Initialize lists to hold the dropdown menus
        self.transform_type_selections = []
        self.file_selections = []

        # Set up the table
        self.setColumnCount(2)
        self.setRowCount(len(self.transform_type_options))
        self.setHorizontalHeaderLabels(["Transform Type", "Default File"])

        # Add dropdown menus to the table for each transform type option
        for i in range(len(self.transform_type_options) - 1):
            # Create and configure the transform type dropdown menu
            combo = QComboBox(self)
            combo.addItems(self.transform_type_options)
            combo.setCurrentIndex(i + 1)
            self.transform_type_selections.append(combo)
            combo.currentIndexChanged.connect(
                self._make_transform_type_handler(combo)
            )

            # Create and configure the file option dropdown menu
            file_combo = QComboBox(self)
            file_combo.addItems(self.file_options)
            file_combo.setCurrentIndex(1)
            self.file_selections.append(file_combo)
            file_combo.currentIndexChanged.connect(
                self._make_file_handler(file_combo)
            )

            # Add the dropdown menus to the table
            self.setCellWidget(i, 0, combo)
            self.setCellWidget(i, 1, file_combo)

        # Add an extra row to the table for adding new transform types
        last_combo = QComboBox(self)
        last_combo.addItems(self.transform_type_options)
        self.transform_type_selections.append(last_combo)
        last_combo.currentIndexChanged.connect(
            self._make_transform_type_handler(last_combo)
        )

        last_file_combo = QComboBox(self)
        last_file_combo.addItems(self.file_options)
        # Enable the last file combo from the loop (now second-to-last overall)
        self.file_selections[-1].setEnabled(True)
        last_file_combo.setEnabled(False)
        self.file_selections.append(last_file_combo)
        last_file_combo.currentIndexChanged.connect(
            self._make_file_handler(last_file_combo)
        )

        last_row = len(self.transform_type_options) - 1
        self.setCellWidget(last_row, 0, last_combo)
        self.setCellWidget(last_row, 1, last_file_combo)

        self.resizeRowsToContents()
        self.resizeColumnsToContents()

    def _make_transform_type_handler(self, combo: QComboBox):
        """Return a signal handler that resolves the row index dynamically."""

        def handler(_):
            try:
                index = self.transform_type_selections.index(combo)
            except ValueError:
                return
            self._on_transform_type_change(index)

        return handler

    def _make_file_handler(self, combo: QComboBox):
        """Return a signal handler that resolves the row index dynamically."""

        def handler(_):
            try:
                index = self.file_selections.index(combo)
            except ValueError:
                return
            self._on_file_change(index)

        return handler

    def _on_transform_type_change(self, index):
        """
        Handle the event when a transform type is changed.

        If a new transform type is selected, emits the
        transform_type_added_signal and adds a new row to the table. If the
        transform type is set to "", removes the row from the table and emits
        the transform_type_removed_signal.

        Parameters
        ----------
        index : int
            The index of the changed transform type.
        """
        if self.transform_type_selections[index].currentIndex() != 0:
            self.transform_type_added_signal.emit(
                self.transform_type_selections[index].currentText(), index
            )

            self.file_selections[index].setCurrentIndex(0)

            if index >= len(self.transform_type_selections) - 1:
                current_length = self.rowCount()
                self.setRowCount(self.rowCount() + 1)

                new_combo = QComboBox(self)
                new_combo.addItems(self.transform_type_options)
                self.transform_type_selections.append(new_combo)
                new_combo.currentIndexChanged.connect(
                    self._make_transform_type_handler(new_combo)
                )

                new_file_combo = QComboBox(self)
                new_file_combo.addItems(self.file_options)
                self.file_selections[-1].setEnabled(True)
                new_file_combo.setEnabled(False)
                self.file_selections.append(new_file_combo)
                new_file_combo.currentIndexChanged.connect(
                    self._make_file_handler(new_file_combo)
                )

                self.setCellWidget(
                    current_length,
                    0,
                    self.transform_type_selections[current_length],
                )
                self.setCellWidget(
                    current_length, 1, self.file_selections[current_length]
                )

        else:
            self.transform_type_selections.pop(index)
            self.file_selections.pop(index)

            self.removeRow(index)
            self.transform_type_removed_signal.emit(index)

    def _on_file_change(self, index):
        """
        Handle the event when a file option is changed.

        Emits the file_option_changed_signal with the name of the file and
        its index.

        Parameters
        ----------
        index : int
            The index of the changed file option.
        """
        self.file_option_changed_signal.emit(
            self.file_selections[index].currentText(), index
        )
