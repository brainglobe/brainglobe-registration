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

    def __init__(self, parent=None):
        super().__init__(parent)

        self.default_params = {}
        self.setColumnCount(2)
        self.setHorizontalHeaderLabels(["Parameter", "Value"])

        with open("./parameters/rigid.txt", "r") as f:
            for line in f.readlines():
                if line[0] == "(":
                    split_line = line[1:-2].split()
                    self.default_params[split_line[0]] = split_line[1].strip("\"")

        self.setRowCount(len(self.default_params)+2)

        for i, k in enumerate(self.default_params):
            new_param = QTableWidgetItem(k)
            new_value = QTableWidgetItem(self.default_params[k])

            self.setItem(i, 0, new_param)
            self.setItem(i, 1, new_value)
