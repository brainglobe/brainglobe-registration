from qtpy.QtWidgets import (
    QTableWidget,
    QTableWidgetItem,
)


class RegistrationParameterListView(QTableWidget):
    """
    A QTableWidget subclass that displays and manages registration parameters.

    This widget displays a table of registration parameters and their values.
    The parameters can be edited directly in the table. When a parameter value is
    changed, the parameter dictionary is updated.

    Attributes
    ----------
    param_dict : dict
        The dictionary of registration parameters.
    transform_type : str
        The transform type.

    Methods
    -------
    set_data(param_dict):
        Sets the data in the table from the parameter dictionary.
    _on_cell_change(row, column):
        Updates the parameter dictionary when a cell in the table is changed.
    """

    def __init__(self, param_dict: dict, transform_type: str, parent=None):
        """
        Initialize the widget.

        Parameters
        ----------
        param_dict : dict
            The dictionary of registration parameters.
        transform_type : str
            The type of transform being used.
        parent : QWidget, optional
            The parent widget, by default None
        """
        super().__init__(parent)
        self.param_dict = {}
        self.transform_type = transform_type
        self.setColumnCount(2)

        self.set_data(param_dict)
        self.setHorizontalHeaderItem(0, QTableWidgetItem("Parameter"))
        self.setHorizontalHeaderItem(1, QTableWidgetItem("Values"))

        self.cellChanged.connect(self._on_cell_change)

    def set_data(self, param_dict):
        """
        Sets the data in the table from the parameter dictionary.

        Parameters
        ----------
        param_dict : dict
            The dictionary of registration parameters.
        """
        self.clear()
        self.setRowCount(len(param_dict) + 1)
        for i, k in enumerate(param_dict):
            new_param = QTableWidgetItem(k)
            new_value = QTableWidgetItem(", ".join(param_dict[k]))

            self.setItem(i, 0, new_param)
            self.setItem(i, 1, new_value)

        self.resizeColumnsToContents()
        self.resizeRowsToContents()
        self.param_dict = param_dict

    def _on_cell_change(self, row, column):
        """
        Updates the parameter dictionary when a cell in the table is changed.

        Parameters
        ----------
        row : int
            The row of the changed cell.
        column : int
            The column of the changed cell.
        """
        if column == 1 and self.item(row, 0):
            parameter = self.item(row, 0).text()
            value = self.item(row, 1).text().split(", ")
            self.param_dict[parameter] = value

            if row == self.rowCount() - 1:
                self.setRowCount(self.rowCount() + 1)
        # TODO - add a way to remove rows if they are empty removing
        #  them from the param dictionary (might have to save the parameter when
        #  it is selected)
