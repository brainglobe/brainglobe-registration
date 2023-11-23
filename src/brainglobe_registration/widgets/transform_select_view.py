from qtpy.QtCore import QSignalMapper, Signal
from qtpy.QtWidgets import (
    QComboBox,
    QTableWidget,
)


class TransformSelectView(QTableWidget):
    """
    A QTableWidget subclass that provides a user interface for selecting transform types and associated files.

    This widget displays a table of available transform types and associated default parameter files.
    The user can select a transform type and an associated file from dropdown menus. The widget emits signals
    when a transform type is added or removed, or when a file option is changed.

    Attributes
    ----------
    transform_type_added_signal : Signal
        Emitted when a transform type is added. The signal includes the name of the transform type and its index.
    transform_type_removed_signal : Signal
        Emitted when a transform type is removed. The signal includes the index of the removed transform type.
    file_option_changed_signal : Signal
        Emitted when a file option is changed. The signal includes the name of the file and its index.

    Methods
    -------
    _on_transform_type_change(index):
        Handles the event when a transform type is changed. Emits the transform_type_added_signal or
        transform_type_removed_signal as appropriate.
    _on_file_change(index):
        Handles the event when a file option is changed. Emits the file_option_changed_signal.
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
            "ara_tools",
            "brainregister_IBL",
        ]
        self.transform_type_options = ["", "rigid", "affine", "bspline"]

        # Create signal mappers for the transform type and file option dropdown menus
        self.transform_type_signaller = QSignalMapper(self)
        self.transform_type_signaller.mapped[int].connect(
            self._on_transform_type_change
        )

        self.file_signaller = QSignalMapper(self)
        self.file_signaller.mapped[int].connect(self._on_file_change)

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
            self.transform_type_selections.append(QComboBox())
            self.transform_type_selections[i].addItems(
                self.transform_type_options
            )
            self.transform_type_selections[i].setCurrentIndex(i + 1)
            self.transform_type_selections[i].currentIndexChanged.connect(
                self.transform_type_signaller.map
            )

            # Create and configure the file option dropdown menu
            self.file_selections.append(QComboBox())
            self.file_selections[i].addItems(self.file_options)
            self.file_selections[i].setCurrentIndex(0)
            self.file_selections[i].currentIndexChanged.connect(
                self.file_signaller.map
            )

            # Add the dropdown menus to the signal mappers
            self.transform_type_signaller.setMapping(
                self.transform_type_selections[i], i
            )
            self.file_signaller.setMapping(self.file_selections[i], i)

            # Add the dropdown menus to the table
            self.setCellWidget(i, 0, self.transform_type_selections[i])
            self.setCellWidget(i, 1, self.file_selections[i])

        # Add an extra row to the table for adding new transform types
        self.transform_type_selections.append(QComboBox())
        self.transform_type_selections[-1].addItems(
            self.transform_type_options
        )
        self.transform_type_selections[-1].currentIndexChanged.connect(
            self.transform_type_signaller.map
        )

        self.transform_type_signaller.setMapping(
            self.transform_type_selections[-1],
            len(self.transform_type_options) - 1,
        )

        self.file_selections.append(QComboBox())
        self.file_selections[-1].addItems(self.file_options)
        self.file_selections[-2].setEnabled(True)
        self.file_selections[-1].setEnabled(False)
        self.file_selections[-1].currentIndexChanged.connect(
            self.file_signaller.map
        )

        self.file_signaller.setMapping(
            self.file_selections[-1], len(self.transform_type_options) - 1
        )

        self.setCellWidget(
            len(self.transform_type_options) - 1,
            0,
            self.transform_type_selections[-1],
        )
        self.setCellWidget(
            len(self.transform_type_options) - 1, 1, self.file_selections[-1]
        )
        self.file_selections[-1].setEnabled(False)
        self.resizeRowsToContents()
        self.resizeColumnsToContents()

    def _on_transform_type_change(self, index):
        """
        Handle the event when a transform type is changed.

        If a new transform type is selected, emits the transform_type_added_signal and adds a new row to the table.
        If the transform type is set to "", removes the row from the table and emits the transform_type_removed_signal.

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
                curr_length = self.rowCount()
                self.setRowCount(self.rowCount() + 1)

                self.transform_type_selections.append(QComboBox())
                self.transform_type_selections[-1].addItems(
                    self.transform_type_options
                )
                self.transform_type_selections[-1].currentIndexChanged.connect(
                    self.transform_type_signaller.map
                )
                self.transform_type_signaller.setMapping(
                    self.transform_type_selections[-1], curr_length
                )

                self.file_selections.append(QComboBox())
                self.file_selections[-1].addItems(self.file_options)
                self.file_selections[-2].setEnabled(True)
                self.file_selections[-1].setEnabled(False)
                self.file_selections[-1].currentIndexChanged.connect(
                    self.file_signaller.map
                )
                self.file_signaller.setMapping(
                    self.file_selections[-1], curr_length
                )

                self.setCellWidget(
                    curr_length, 0, self.transform_type_selections[curr_length]
                )
                self.setCellWidget(
                    curr_length, 1, self.file_selections[curr_length]
                )

        else:
            self.transform_type_signaller.removeMappings(
                self.transform_type_selections[index]
            )
            self.transform_type_selections.pop(index)

            self.file_signaller.removeMappings(self.file_selections[index])
            self.file_selections.pop(index)

            # Update mappings
            for i in range(index, len(self.transform_type_selections)):
                self.transform_type_signaller.removeMappings(
                    self.transform_type_selections[i]
                )
                self.transform_type_signaller.setMapping(
                    self.transform_type_selections[i], i
                )
                self.file_signaller.removeMappings(self.file_selections[i])
                self.file_signaller.setMapping(self.file_selections[i], i)

            self.removeRow(index)
            self.transform_type_removed_signal.emit(index)

    def _on_file_change(self, index):
        """
        Handle the event when a file option is changed.

        Emits the file_option_changed_signal with the name of the file and its index.

        Parameters
        ----------
        index : int
            The index of the changed file option.
        """
        self.file_option_changed_signal.emit(
            self.file_selections[index].currentText(), index
        )
