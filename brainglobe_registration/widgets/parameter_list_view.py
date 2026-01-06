from typing import Dict

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QTableWidget,
    QTableWidgetItem,
)


class RegistrationParameterListView(QTableWidget):
    """
    A QTableWidget subclass that displays and manages registration parameters.

    This widget displays a table of registration parameters and their values.
    The parameters can be edited directly in the table. When a parameter
    value is changed, the parameter dictionary is updated.

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
        self.param_dict = param_dict
        self.transform_type = transform_type
        self.setColumnCount(2)

        # Track old parameter names for row deletion
        # Maps row index to parameter name before editing
        self._old_param_names: Dict[int, str] = {}

        self.set_data(param_dict)
        self.setHorizontalHeaderItem(0, QTableWidgetItem("Parameter"))
        self.setHorizontalHeaderItem(1, QTableWidgetItem("Values"))

        # Connect signals for tracking parameter name changes
        # Use itemChanged to track changes, cellChanged for processing
        self.itemChanged.connect(self._on_item_changed)
        self.cellChanged.connect(self._on_cell_change)

        # Track the previous text before editing starts
        # This helps us capture the old parameter name
        self._editing_item = None

    def set_data(self, param_dict):
        """
        Sets the data in the table from the parameter dictionary.

        Parameters
        ----------
        param_dict : dict
            The dictionary of registration parameters.
        """
        self.clear()
        self._old_param_names.clear()
        self.setRowCount(len(param_dict) + 1)
        for i, k in enumerate(param_dict):
            new_param = QTableWidgetItem(k)
            new_value = QTableWidgetItem(", ".join(param_dict[k]))

            # Store the parameter name in the item's data for tracking
            new_param.setData(Qt.ItemDataRole.UserRole, k)

            self.setItem(i, 0, new_param)
            self.setItem(i, 1, new_value)
            # Track the parameter name for this row
            self._old_param_names[i] = k

        self.resizeColumnsToContents()
        self.resizeRowsToContents()
        self.param_dict = param_dict

    def _on_item_changed(self, item):
        """
        Track parameter name changes to enable row deletion.

        This is called when an item changes. We use this to ensure
        the old parameter name is tracked before cellChanged fires.

        Parameters
        ----------
        item : QTableWidgetItem
            The item that was changed.
        """
        # Skip if signals are blocked (during row removal)
        if self.signalsBlocked():
            return

        row = item.row()
        column = item.column()

        # If parameter name column (0) is being edited,
        # ensure we track the old name
        if column == 0:
            # Get the old name from our tracking dict or item data
            if row not in self._old_param_names:
                # Try to get from item's UserRole data first
                old_name = item.data(Qt.ItemDataRole.UserRole)
                if old_name is None or old_name == "":
                    # Fallback: use current text (might already be new value)
                    old_name = item.text()
                # Store in tracking dict
                self._old_param_names[row] = old_name

    def _on_cell_change(self, row, column):
        """
        Updates the parameter dictionary when a cell in the table is changed.

        Handles:
        - Updating parameter values when column 1 (values) is changed
        - Removing parameters when parameter name (column 0) is cleared
        - Removing rows when both parameter name and values are empty

        Parameters
        ----------
        row : int
            The row of the changed cell.
        column : int
            The column of the changed cell.
        """
        # Skip if signals are blocked (during row removal)
        if self.signalsBlocked():
            return

        param_item = self.item(row, 0)
        value_item = self.item(row, 1)

        # Handle parameter name deletion (column 0)
        if column == 0:
            old_param_name = self._old_param_names.get(row)
            new_param_name = param_item.text().strip() if param_item else ""

            # If parameter name was cleared or is empty
            if not new_param_name and old_param_name:
                # Remove the old parameter from dictionary
                if old_param_name in self.param_dict:
                    del self.param_dict[old_param_name]

                # Check if value is also empty - if so, remove the row
                value_text = value_item.text().strip() if value_item else ""
                if not value_text:
                    # Block signals temporarily to avoid recursion
                    self.blockSignals(True)

                    try:
                        # Remove the row
                        self.removeRow(row)

                        # Update tracking dictionary for remaining rows
                        self._old_param_names = {}
                        for i in range(self.rowCount()):
                            item = self.item(i, 0)
                            if item is not None:
                                stored_name = item.data(
                                    Qt.ItemDataRole.UserRole
                                )
                                if stored_name:
                                    self._old_param_names[i] = stored_name
                                else:
                                    self._old_param_names[i] = item.text()
                    finally:
                        # Always unblock signals, even if there's an error
                        self.blockSignals(False)
                    return
                else:
                    # Parameter name cleared but value exists
                    # Clear the value item and update tracking
                    self.blockSignals(True)
                    try:
                        if value_item:
                            value_item.setText("")
                    finally:
                        self.blockSignals(False)
                    # Update stored name to empty
                    self._old_param_names[row] = ""

            # If parameter name was changed (not cleared), update tracking
            elif new_param_name and new_param_name != old_param_name:
                # Remove old parameter from dict if it existed
                if old_param_name and old_param_name in self.param_dict:
                    old_value = self.param_dict.pop(old_param_name)
                    # Add new parameter with old value
                    self.param_dict[new_param_name] = old_value

                # Update stored name
                self._old_param_names[row] = new_param_name
                # Update item data
                if param_item:
                    param_item.setData(
                        Qt.ItemDataRole.UserRole, new_param_name
                    )

        # Handle value changes (column 1)
        elif column == 1 and param_item:
            parameter = param_item.text().strip()
            if parameter:
                value_text = value_item.text() if value_item else ""
                value = value_text.split(", ") if value_text else [""]
                # Filter out empty strings from split
                value = [v.strip() for v in value if v.strip()]
                if value:
                    self.param_dict[parameter] = value
                elif parameter in self.param_dict:
                    # Value cleared but parameter exists - remove from dict
                    del self.param_dict[parameter]

                # Add new row if we're at the last row
                if row == self.rowCount() - 1:
                    self.setRowCount(self.rowCount() + 1)
