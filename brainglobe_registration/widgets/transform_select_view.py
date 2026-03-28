from qtpy.QtCore import QSignalMapper, Signal
from qtpy.QtWidgets import (
    QComboBox,
    QPushButton,
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
    import_button_clicked_signal = Signal(int)
    export_button_clicked_signal = Signal(int)

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

        # Create signal mappers for the transform type and file option
        # dropdown menus
        self.transform_type_signaller = QSignalMapper(self)
        self.transform_type_signaller.mapped[int].connect(
            self._on_transform_type_change
        )

        self.file_signaller = QSignalMapper(self)
        self.file_signaller.mapped[int].connect(self._on_file_change)

        self.import_button_signaller = QSignalMapper(self)
        self.import_button_signaller.mapped[int].connect(
            self._on_import_button_clicked
        )

        self.export_button_signaller = QSignalMapper(self)
        self.export_button_signaller.mapped[int].connect(
            self._on_export_button_clicked
        )

        # Initialize lists to hold the dropdown menus
        self.transform_type_selections = []
        self.file_selections = []
        self.import_buttons = []
        self.export_buttons = []

        # Set up the table
        self.setColumnCount(4)
        self.setRowCount(len(self.transform_type_options))
        self.setHorizontalHeaderLabels(
            ["Transform Type", "File", "Import", "Export"]
        )

        # Add dropdown menus to the table for each transform type option
        for i in range(len(self.transform_type_options) - 1):
            # Create and configure the transform type dropdown menu
            self.transform_type_selections.append(QComboBox(self))
            self.transform_type_selections[i].addItems(
                self.transform_type_options
            )
            self.transform_type_selections[i].setCurrentIndex(i + 1)
            self.transform_type_selections[i].currentIndexChanged.connect(
                self.transform_type_signaller.map
            )

            # Create and configure the file option dropdown menu
            self.file_selections.append(QComboBox(self))
            self.file_selections[i].addItems(self.file_options)
            self.file_selections[i].setCurrentIndex(1)
            self.file_selections[i].currentIndexChanged.connect(
                self.file_signaller.map
            )

            # Add the dropdown menus to the signal mappers
            self.transform_type_signaller.setMapping(
                self.transform_type_selections[i], i
            )
            self.file_signaller.setMapping(self.file_selections[i], i)

            import_button = QPushButton("Import")
            import_button.clicked.connect(self.import_button_signaller.map)
            self.import_button_signaller.setMapping(import_button, i)
            self.import_buttons.append(import_button)

            export_button = QPushButton("Export")
            export_button.clicked.connect(self.export_button_signaller.map)
            self.export_button_signaller.setMapping(export_button, i)
            self.export_buttons.append(export_button)

            # Add the dropdown menus to the table
            self.setCellWidget(i, 0, self.transform_type_selections[i])
            self.setCellWidget(i, 1, self.file_selections[i])
            self.setCellWidget(i, 2, self.import_buttons[i])
            self.setCellWidget(i, 3, self.export_buttons[i])

        # Add an extra row to the table for adding new transform types
        self.transform_type_selections.append(QComboBox(self))
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

        self.file_selections.append(QComboBox(self))
        self.file_selections[-1].addItems(self.file_options)
        self.file_selections[-2].setEnabled(True)
        self.file_selections[-1].setEnabled(False)
        self.file_selections[-1].currentIndexChanged.connect(
            self.file_signaller.map
        )

        # Subtract 1 to account for the empty transform type option
        self.file_signaller.setMapping(
            self.file_selections[-1], len(self.transform_type_options) - 1
        )

        import_button = QPushButton("Import")
        import_button.setEnabled(False)
        import_button.clicked.connect(self.import_button_signaller.map)
        self.import_button_signaller.setMapping(
            import_button, len(self.transform_type_options) - 1
        )
        self.import_buttons.append(import_button)

        export_button = QPushButton("Export")
        export_button.setEnabled(False)
        export_button.clicked.connect(self.export_button_signaller.map)
        self.export_button_signaller.setMapping(
            export_button, len(self.transform_type_options) - 1
        )
        self.export_buttons.append(export_button)

        self.setCellWidget(
            len(self.transform_type_options) - 1,
            0,
            self.transform_type_selections[-1],
        )
        self.setCellWidget(
            len(self.transform_type_options) - 1, 1, self.file_selections[-1]
        )
        self.setCellWidget(
            len(self.transform_type_options) - 1, 2, self.import_buttons[-1]
        )
        self.setCellWidget(
            len(self.transform_type_options) - 1, 3, self.export_buttons[-1]
        )

        self.resizeRowsToContents()
        self.resizeColumnsToContents()

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
            self.file_selections[index].setEnabled(True)
            self.import_buttons[index].setEnabled(True)
            self.export_buttons[index].setEnabled(True)

            if index >= len(self.transform_type_selections) - 1:
                current_length = self.rowCount()
                self.setRowCount(self.rowCount() + 1)

                self.transform_type_selections.append(QComboBox(self))
                self.transform_type_selections[-1].addItems(
                    self.transform_type_options
                )
                self.transform_type_selections[-1].currentIndexChanged.connect(
                    self.transform_type_signaller.map
                )
                self.transform_type_signaller.setMapping(
                    self.transform_type_selections[-1], current_length
                )

                self.file_selections.append(QComboBox(self))
                self.file_selections[-1].addItems(self.file_options)
                self.file_selections[-2].setEnabled(True)
                self.file_selections[-1].setEnabled(False)
                self.file_selections[-1].currentIndexChanged.connect(
                    self.file_signaller.map
                )
                self.file_signaller.setMapping(
                    self.file_selections[-1], current_length
                )

                import_button = QPushButton("Import")
                self.import_buttons[-1].setEnabled(True)
                import_button.setEnabled(False)
                import_button.clicked.connect(self.import_button_signaller.map)
                self.import_button_signaller.setMapping(
                    import_button, current_length
                )
                self.import_buttons.append(import_button)

                export_button = QPushButton("Export")
                self.export_buttons[-1].setEnabled(True)
                export_button.setEnabled(False)
                export_button.clicked.connect(self.export_button_signaller.map)
                self.export_button_signaller.setMapping(
                    export_button, current_length
                )
                self.export_buttons.append(export_button)

                self.setCellWidget(
                    current_length,
                    0,
                    self.transform_type_selections[current_length],
                )
                self.setCellWidget(
                    current_length, 1, self.file_selections[current_length]
                )
                self.setCellWidget(
                    current_length, 2, self.import_buttons[current_length]
                )
                self.setCellWidget(
                    current_length, 3, self.export_buttons[current_length]
                )

        else:
            self.transform_type_signaller.removeMappings(
                self.transform_type_selections[index]
            )
            self.transform_type_selections.pop(index)

            self.file_signaller.removeMappings(self.file_selections[index])
            self.file_selections.pop(index)

            self.import_button_signaller.removeMappings(
                self.import_buttons[index]
            )
            self.import_buttons.pop(index)

            self.export_button_signaller.removeMappings(
                self.export_buttons[index]
            )
            self.export_buttons.pop(index)

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
                self.import_button_signaller.removeMappings(
                    self.import_buttons[i]
                )
                self.import_button_signaller.setMapping(
                    self.import_buttons[i], i
                )
                self.export_button_signaller.removeMappings(
                    self.export_buttons[i]
                )
                self.export_button_signaller.setMapping(
                    self.export_buttons[i], i
                )

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

    def _on_import_button_clicked(self, index: int) -> None:
        self.import_button_clicked_signal.emit(index)

    def _on_export_button_clicked(self, index: int) -> None:
        self.export_button_clicked_signal.emit(index)

    def set_transform_type_selection(
        self, index: int, transform_type: str
    ) -> None:
        if index < 0 or index >= len(self.transform_type_selections):
            raise IndexError("Transform selection out of order")

        transform_index = self.transform_type_selections[index].findText(
            transform_type
        )
        if transform_index == -1:
            return

        self.transform_type_selections[index].setCurrentIndex(transform_index)

    def set_custom_file_path(self, index: int, file_path: str) -> None:
        from pathlib import Path

        if index < 0 or index >= len(self.file_selections):
            raise IndexError("Transform file selection out of order")

        # Clear any existing custom files first to avoid duplicates
        self.clear_custom_file(index)

        filename = Path(file_path).name
        display_text = f"(Custom) {filename}"

        self.file_selections[index].addItem(display_text)
        option_index = self.file_selections[index].findText(display_text)

        self.file_selections[index].blockSignals(True)
        self.file_selections[index].setCurrentIndex(option_index)
        self.file_selections[index].blockSignals(False)

        self.file_selections[index].setToolTip(file_path)

    def clear_custom_file(self, index: int) -> None:
        if index < 0 or index >= len(self.file_selections):
            raise IndexError("Transform file selection out of order")

        for i in range(self.file_selections[index].count() - 1, -1, -1):
            if self.file_selections[index].itemText(i).startswith("(Custom)"):
                self.file_selections[index].removeItem(i)

        self.file_selections[index].setToolTip("")
