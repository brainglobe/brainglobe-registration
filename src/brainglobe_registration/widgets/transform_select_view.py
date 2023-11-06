from qtpy.QtCore import QSignalMapper, Signal
from qtpy.QtWidgets import (
    QComboBox,
    QTableWidget,
)


class TransformSelectView(QTableWidget):
    transform_type_added_signal = Signal(str, int)
    transform_type_removed_signal = Signal(int)
    file_signal = Signal(str, int)

    def __init__(self, parent=None):
        super().__init__(parent=parent)

        self.file_options = [
            "elastix_default",
            "ara_tools",
            "brainregister_IBL",
        ]
        self.transform_type_options = ["", "rigid", "affine", "bspline"]

        self.transform_type_signaller = QSignalMapper(self)
        self.transform_type_signaller.mapped[int].connect(
            self._on_transform_type_change
        )

        self.file_signaller = QSignalMapper(self)
        self.file_signaller.mapped[int].connect(self._on_file_change)

        self.transform_type_selections = []
        self.file_selections = []

        self.setColumnCount(2)
        self.setRowCount(len(self.transform_type_options))
        self.setHorizontalHeaderLabels(["Transform Type", "Default File"])

        # -1 accounts for the "empty" first selection of transform_type_options
        for i in range(len(self.transform_type_options) - 1):
            self.transform_type_selections.append(QComboBox())
            self.transform_type_selections[i].addItems(
                self.transform_type_options
            )
            self.transform_type_selections[i].setCurrentIndex(i + 1)
            self.transform_type_selections[i].currentIndexChanged.connect(
                self.transform_type_signaller.map
            )

            self.file_selections.append(QComboBox())
            self.file_selections[i].addItems(self.file_options)
            self.file_selections[i].setCurrentIndex(0)
            self.file_selections[i].currentIndexChanged.connect(
                self.file_signaller.map
            )

            self.transform_type_signaller.setMapping(
                self.transform_type_selections[i], i
            )
            self.file_signaller.setMapping(self.file_selections[i], i)

            self.setCellWidget(i, 0, self.transform_type_selections[i])
            self.setCellWidget(i, 1, self.file_selections[i])

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
        #
        # elif index == 0:
        #     for i in range(len(self.transform_type_selections) - 1, 0, -1):
        #         self.transform_type_signaller.removeMappings(
        #             self.transform_type_selections[i]
        #         )
        #         self.transform_type_selections.pop(i)
        #         self.file_signaller.removeMappings(self.file_selections[i])
        #         self.file_selections.pop(i)
        #         self.removeRow(i)

    def _on_file_change(self, index):
        self.file_signal.emit(self.file_selections[index].currentText(), index)
