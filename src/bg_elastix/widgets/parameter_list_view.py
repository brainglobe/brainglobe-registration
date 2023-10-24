from qtpy.QtCore import Signal

from qtpy.QtWidgets import (
    QTableWidget,
    QTableWidgetItem,
    QFormLayout,
    QSpinBox,
    QDoubleSpinBox,
    QPushButton,
    QHBoxLayout,
)


class RegistrationParameterListView(QTableWidget):

    def __init__(self, param_dict: dict, transform_type: str, parent=None):
        super().__init__(parent)

        self.transform_type = transform_type
        self.setColumnCount(2)
        self.setHorizontalHeaderLabels(["Parameter", "Value"])

        self.set_data(param_dict)

    def set_data(self, param_dict):
        self.setRowCount(len(param_dict)+2)
        for i, k in enumerate(param_dict):
            new_param = QTableWidgetItem(k)
            new_value = QTableWidgetItem(param_dict[k])

            self.setItem(i, 0, new_param)
            self.setItem(i, 1, new_value)

        self.resizeColumnsToContents()
        self.resizeRowsToContents()
