from qtpy.QtWidgets import (
    QTableWidget,
    QTableWidgetItem,
)


class RegistrationParameterListView(QTableWidget):
    def __init__(self, param_dict: dict, transform_type: str, parent=None):
        super().__init__(parent)
        self.param_dict = {}
        self.transform_type = transform_type
        self.setColumnCount(2)
        self.setHorizontalHeaderLabels(["Parameter", "Value"])

        self.set_data(param_dict)

        self.cellChanged.connect(self._on_cell_change)

    def set_data(self, param_dict):
        self.setRowCount(len(param_dict) + 1)
        for i, k in enumerate(param_dict):
            new_param = QTableWidgetItem(k)
            new_value = QTableWidgetItem(param_dict[k])

            self.setItem(i, 0, new_param)
            self.setItem(i, 1, new_value)

        self.resizeColumnsToContents()
        self.resizeRowsToContents()
        self.param_dict = param_dict

    def _on_cell_change(self, row, column):
        if column == 1:
            parameter = self.item(row, 0).text()
            value = self.item(row, 1).text()
            self.param_dict[parameter] = value

            if row == self.rowCount() - 1:
                self.setRowCount(self.rowCount() + 1)
        # TODO - add a way to remove rows if they are empty removing
        #  them from the param dictionary (might have to save the parameter when
        #  it is selected)