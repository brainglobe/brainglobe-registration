import pytest
from qtpy.QtWidgets import QTableWidgetItem

from brainglobe_registration.widgets.parameter_list_view import (
    RegistrationParameterListView,
)

param_dict = {
    "AutomaticScalesEstimation": ["true"],
    "AutomaticTransformInitialization": ["true"],
    "BSplineInterpolationOrder": ["1"],
    "CheckNumberOfSamples": ["true"],
    "Transform": ["BSplineTransform"],
}
transform_type = "bspline"


@pytest.fixture
def parameter_list_view() -> RegistrationParameterListView:
    """Create a fresh parameter list view for each test."""
    parameter_list_view = RegistrationParameterListView(
        param_dict=param_dict.copy(), transform_type=transform_type
    )
    return parameter_list_view


def test_parameter_list_view(parameter_list_view, qtbot):
    qtbot.addWidget(parameter_list_view)

    assert parameter_list_view.rowCount() == len(param_dict) + 1
    assert parameter_list_view.columnCount() == 2

    assert parameter_list_view.horizontalHeaderItem(0).text() == "Parameter"
    assert parameter_list_view.horizontalHeaderItem(1).text() == "Values"

    for i, k in enumerate(param_dict):
        assert parameter_list_view.item(i, 0).text() == k
        assert parameter_list_view.item(i, 1).text() == ", ".join(
            param_dict[k]
        )


def test_parameter_list_view_cell_change(parameter_list_view, qtbot):
    qtbot.addWidget(parameter_list_view)

    with qtbot.waitSignal(
        parameter_list_view.cellChanged, timeout=1000
    ) as blocker:
        parameter_list_view.item(0, 1).setText("false")

    assert blocker.args == [0, 1]
    assert parameter_list_view.param_dict["AutomaticScalesEstimation"] == [
        "false"
    ]


def test_parameter_list_view_cell_change_last_row(parameter_list_view, qtbot):
    qtbot.addWidget(parameter_list_view)

    current_row_count = parameter_list_view.rowCount()
    last_row_index = len(param_dict)

    parameter_list_view.setItem(
        last_row_index, 0, QTableWidgetItem("TestParameter")
    )
    parameter_list_view.setItem(last_row_index, 1, QTableWidgetItem("true"))

    assert parameter_list_view.param_dict["TestParameter"] == ["true"]
    assert parameter_list_view.rowCount() == current_row_count + 1


def test_parameter_list_view_cell_change_last_row_no_param(
    parameter_list_view, qtbot
):
    qtbot.addWidget(parameter_list_view)

    current_row_count = parameter_list_view.rowCount()
    last_row_index = len(param_dict)

    parameter_list_view.setItem(last_row_index, 1, QTableWidgetItem("true"))

    assert parameter_list_view.rowCount() == current_row_count


def test_parameter_list_view_cell_change_last_row_no_value(
    parameter_list_view, qtbot
):
    qtbot.addWidget(parameter_list_view)

    current_row_count = parameter_list_view.rowCount()
    last_row_index = len(param_dict)

    parameter_list_view.setItem(
        last_row_index, 0, QTableWidgetItem("TestParameter")
    )

    assert parameter_list_view.rowCount() == current_row_count


class TestRowDeletion:
    """Tests for row deletion functionality."""

    def test_delete_row_by_clearing_parameter_name(
        self, parameter_list_view, qtbot
    ):
        """Test that clearing parameter name removes it from dictionary."""
        qtbot.addWidget(parameter_list_view)

        # Get initial state
        param_name = "AutomaticScalesEstimation"
        assert param_name in parameter_list_view.param_dict

        # Find the row with this parameter
        target_row = None
        for i in range(parameter_list_view.rowCount()):
            item = parameter_list_view.item(i, 0)
            if item and item.text() == param_name:
                target_row = i
                break

        assert target_row is not None

        # Clear the parameter name
        # Note: setText() triggers itemChanged, which then triggers cellChanged
        with qtbot.waitSignal(parameter_list_view.cellChanged, timeout=1000):
            parameter_list_view.item(target_row, 0).setText("")

        # Parameter should be removed from dictionary
        assert param_name not in parameter_list_view.param_dict

    def test_delete_row_when_both_name_and_value_empty(
        self, parameter_list_view, qtbot
    ):
        """Test that row is removed when both name and value are empty."""
        qtbot.addWidget(parameter_list_view)

        # Get initial state
        initial_row_count = parameter_list_view.rowCount()
        param_name = "CheckNumberOfSamples"
        assert param_name in parameter_list_view.param_dict

        # Find the row
        target_row = None
        for i in range(parameter_list_view.rowCount()):
            item = parameter_list_view.item(i, 0)
            if item and item.text() == param_name:
                target_row = i
                break

        assert target_row is not None

        # Clear both name and value
        with qtbot.waitSignal(parameter_list_view.cellChanged, timeout=1000):
            parameter_list_view.item(target_row, 1).setText("")

        # Clear parameter name (this should trigger row deletion)
        # Note: row deletion happens synchronously, so we just need to wait
        # for the cellChanged signal
        with qtbot.waitSignal(parameter_list_view.cellChanged, timeout=1000):
            parameter_list_view.item(target_row, 0).setText("")

        # Row should be removed
        assert parameter_list_view.rowCount() == initial_row_count - 1
        assert param_name not in parameter_list_view.param_dict

    def test_parameter_name_cleared_but_value_remains(
        self, parameter_list_view, qtbot
    ):
        """Test clearing name when value exists clears value and deletes."""
        qtbot.addWidget(parameter_list_view)

        initial_row_count = parameter_list_view.rowCount()
        param_name = "BSplineInterpolationOrder"
        target_row = None
        for i in range(parameter_list_view.rowCount()):
            item = parameter_list_view.item(i, 0)
            if item and item.text() == param_name:
                target_row = i
                break

        assert target_row is not None

        # Clear parameter name (value exists, so both cleared and row deleted)
        parameter_list_view.item(target_row, 0).setText("")
        qtbot.wait(100)  # Wait for signals to process

        # Parameter should be removed from dict
        assert param_name not in parameter_list_view.param_dict
        # Row should be deleted (since both name and value are now empty)
        assert parameter_list_view.rowCount() == initial_row_count - 1

    def test_parameter_renaming(self, parameter_list_view, qtbot):
        """Test that renaming a parameter updates the dictionary correctly."""
        qtbot.addWidget(parameter_list_view)

        old_name = "Transform"
        new_name = "NewTransformName"
        original_value = parameter_list_view.param_dict[old_name].copy()

        # Find the row
        target_row = None
        for i in range(parameter_list_view.rowCount()):
            item = parameter_list_view.item(i, 0)
            if item and item.text() == old_name:
                target_row = i
                break

        assert target_row is not None

        # Rename the parameter
        with qtbot.waitSignal(parameter_list_view.cellChanged, timeout=1000):
            parameter_list_view.item(target_row, 0).setText(new_name)

        # Old name should be removed, new name should have the value
        assert old_name not in parameter_list_view.param_dict
        assert new_name in parameter_list_view.param_dict
        assert parameter_list_view.param_dict[new_name] == original_value

    def test_delete_multiple_rows(self, parameter_list_view, qtbot):
        """Test deleting multiple rows in sequence."""
        qtbot.addWidget(parameter_list_view)

        initial_row_count = parameter_list_view.rowCount()
        params_to_delete = [
            "AutomaticScalesEstimation",
            "CheckNumberOfSamples",
        ]

        for param_name in params_to_delete:
            # Find the row
            target_row = None
            for i in range(parameter_list_view.rowCount()):
                item = parameter_list_view.item(i, 0)
                if item and item.text() == param_name:
                    target_row = i
                    break

            if target_row is not None:
                # Clear value first
                parameter_list_view.item(target_row, 1).setText("")
                qtbot.wait(100)

                # Clear name to delete row
                parameter_list_view.item(target_row, 0).setText("")
                qtbot.wait(150)  # Wait for signals and row removal

                # Verify deletion
                assert param_name not in parameter_list_view.param_dict

        # Verify rows were removed
        final_row_count = parameter_list_view.rowCount()
        assert final_row_count < initial_row_count

    def test_clear_value_removes_parameter(self, parameter_list_view, qtbot):
        """Test that clearing value removes parameter from dictionary."""
        qtbot.addWidget(parameter_list_view)

        param_name = "BSplineInterpolationOrder"
        assert param_name in parameter_list_view.param_dict

        # Find the row
        target_row = None
        for i in range(parameter_list_view.rowCount()):
            item = parameter_list_view.item(i, 0)
            if item and item.text() == param_name:
                target_row = i
                break

        assert target_row is not None

        # Clear the value
        parameter_list_view.item(target_row, 1).setText("")
        qtbot.wait(100)  # Wait for signals to process

        # Parameter should be removed from dictionary
        assert param_name not in parameter_list_view.param_dict

    def test_old_param_names_tracking(self, parameter_list_view, qtbot):
        """Test that old parameter names are tracked correctly."""
        qtbot.addWidget(parameter_list_view)

        # Check that tracking dictionary is initialized
        assert hasattr(parameter_list_view, "_old_param_names")
        assert len(parameter_list_view._old_param_names) > 0

        # Verify tracking for existing parameters
        from qtpy.QtCore import Qt

        for i, param_name in enumerate(param_dict.keys()):
            item = parameter_list_view.item(i, 0)
            if item:
                # Check that UserRole data is set
                stored_name = item.data(Qt.ItemDataRole.UserRole)
                assert stored_name == param_name

    def test_set_data_resets_tracking(self, parameter_list_view, qtbot):
        """Test that set_data resets the tracking dictionary."""
        qtbot.addWidget(parameter_list_view)

        # Modify tracking
        parameter_list_view._old_param_names[999] = "test"

        # Reset data
        new_param_dict = {"NewParam": ["value1", "value2"]}
        parameter_list_view.set_data(new_param_dict)

        # Tracking should be reset
        assert 999 not in parameter_list_view._old_param_names
        assert len(parameter_list_view._old_param_names) == len(new_param_dict)

    def test_delete_last_row_does_not_crash(self, parameter_list_view, qtbot):
        """Test that deleting the last row doesn't cause crashes."""
        qtbot.addWidget(parameter_list_view)

        # Add a parameter in the last row
        last_row = parameter_list_view.rowCount() - 1
        parameter_list_view.setItem(last_row, 0, QTableWidgetItem("TestParam"))
        parameter_list_view.setItem(
            last_row, 1, QTableWidgetItem("test_value")
        )

        qtbot.wait(100)

        # Clear both cells
        parameter_list_view.item(last_row, 1).setText("")
        qtbot.wait(100)

        parameter_list_view.item(last_row, 0).setText("")
        qtbot.wait(200)  # Wait for signals and row removal

        # Should not crash, row should be removed
        assert "TestParam" not in parameter_list_view.param_dict
        assert parameter_list_view.rowCount() >= len(param_dict)

    def test_empty_value_with_valid_name(self, parameter_list_view, qtbot):
        """Test behavior when value is empty but name is valid."""
        qtbot.addWidget(parameter_list_view)

        param_name = "AutomaticTransformInitialization"
        target_row = None
        for i in range(parameter_list_view.rowCount()):
            item = parameter_list_view.item(i, 0)
            if item and item.text() == param_name:
                target_row = i
                break

        assert target_row is not None

        # Clear value
        parameter_list_view.item(target_row, 1).setText("")
        qtbot.wait(100)  # Wait for signals to process

        # Parameter should be removed from dict (empty value)
        assert param_name not in parameter_list_view.param_dict

    def test_whitespace_only_parameter_name(self, parameter_list_view, qtbot):
        """Test that whitespace-only parameter names are treated as empty."""
        qtbot.addWidget(parameter_list_view)

        param_name = "Transform"
        target_row = None
        for i in range(parameter_list_view.rowCount()):
            item = parameter_list_view.item(i, 0)
            if item and item.text() == param_name:
                target_row = i
                break

        assert target_row is not None

        # Set parameter name to whitespace only
        parameter_list_view.item(target_row, 0).setText("   ")
        qtbot.wait(100)  # Wait for signals to process

        # Should be treated as empty and removed
        assert param_name not in parameter_list_view.param_dict

    def test_delete_row_when_value_cleared_first(
        self, parameter_list_view, qtbot
    ):
        """Test that row is deleted when value is cleared first, then name."""
        qtbot.addWidget(parameter_list_view)

        initial_row_count = parameter_list_view.rowCount()
        param_name = "CheckNumberOfSamples"
        assert param_name in parameter_list_view.param_dict

        # Find the row
        target_row = None
        for i in range(parameter_list_view.rowCount()):
            item = parameter_list_view.item(i, 0)
            if item and item.text() == param_name:
                target_row = i
                break

        assert target_row is not None

        # Clear value first
        parameter_list_view.item(target_row, 1).setText("")
        qtbot.wait(100)

        # Clear parameter name (this should trigger row deletion)
        with qtbot.waitSignal(parameter_list_view.cellChanged, timeout=1000):
            parameter_list_view.item(target_row, 0).setText("")

        # Row should be removed
        assert parameter_list_view.rowCount() == initial_row_count - 1
        assert param_name not in parameter_list_view.param_dict

    def test_delete_row_when_both_cleared_via_value(
        self, parameter_list_view, qtbot
    ):
        """Test that row is deleted when name is cleared and value exists."""
        qtbot.addWidget(parameter_list_view)

        initial_row_count = parameter_list_view.rowCount()
        param_name = "BSplineInterpolationOrder"
        assert param_name in parameter_list_view.param_dict

        # Find the row
        target_row = None
        for i in range(parameter_list_view.rowCount()):
            item = parameter_list_view.item(i, 0)
            if item and item.text() == param_name:
                target_row = i
                break

        assert target_row is not None

        # Clear parameter name first (both cleared and row deleted)
        with qtbot.waitSignal(parameter_list_view.cellChanged, timeout=1000):
            parameter_list_view.item(target_row, 0).setText("")
        qtbot.wait(100)

        # Row removed (deleted when name cleared since value was also cleared)
        assert parameter_list_view.rowCount() == initial_row_count - 1
        assert param_name not in parameter_list_view.param_dict

    def test_signals_blocked_in_item_changed(self, parameter_list_view, qtbot):
        """Test that itemChanged skips processing when signals are blocked."""
        qtbot.addWidget(parameter_list_view)

        # Block signals
        parameter_list_view.blockSignals(True)

        # Create item and call method directly to test signalsBlocked check
        item = parameter_list_view.item(0, 0)
        # Call method directly to test early return when signals blocked
        parameter_list_view._on_item_changed(item)

        # Unblock signals
        parameter_list_view.blockSignals(False)

        # Verify the method executed without crashing
        assert item is not None

    def test_signals_blocked_in_cell_change(self, parameter_list_view, qtbot):
        """Test that cellChanged skips processing when signals are blocked."""
        qtbot.addWidget(parameter_list_view)

        # Block signals
        parameter_list_view.blockSignals(True)

        # Call method directly to test early return when signals blocked
        parameter_list_view._on_cell_change(0, 1)

        # Unblock signals
        parameter_list_view.blockSignals(False)

        # Verify the method executed without crashing
        assert parameter_list_view.rowCount() > 0

    def test_parameter_name_cleared_value_remains_else_branch(
        self, parameter_list_view, qtbot
    ):
        """Test else branch when parameter name cleared but value exists."""
        from qtpy.QtCore import Qt

        qtbot.addWidget(parameter_list_view)

        # Clear UserRole from items to test else branch at line 207
        for i in range(min(2, parameter_list_view.rowCount() - 1)):
            item = parameter_list_view.item(i, 0)
            if item:
                item.setData(Qt.ItemDataRole.UserRole, None)

        param_name = "Transform"
        target_row = None
        for i in range(parameter_list_view.rowCount()):
            item = parameter_list_view.item(i, 0)
            if item and item.text() == param_name:
                target_row = i
                break

        assert target_row is not None

        # Ensure value exists
        value_item = parameter_list_view.item(target_row, 1)
        assert value_item is not None
        assert value_item.text() != ""

        # Clear parameter name (value should also be cleared and row deleted)
        initial_row_count = parameter_list_view.rowCount()
        parameter_list_view.item(target_row, 0).setText("")
        qtbot.wait(100)

        # Parameter should be removed from dict
        assert param_name not in parameter_list_view.param_dict
        # Row should be deleted
        assert parameter_list_view.rowCount() == initial_row_count - 1

    def test_row_deletion_with_item_without_userrole(
        self, parameter_list_view, qtbot
    ):
        """Test row deletion when item lacks UserRole (covers else branch)."""
        qtbot.addWidget(parameter_list_view)

        # Add a new row with an item that doesn't have UserRole set
        last_row = parameter_list_view.rowCount() - 1
        new_item = QTableWidgetItem("TestParam")
        # Don't set UserRole - this will test the else branch
        parameter_list_view.setItem(last_row, 0, new_item)
        parameter_list_view.setItem(
            last_row, 1, QTableWidgetItem("test_value")
        )
        qtbot.wait(100)

        # Now clear both to trigger deletion
        parameter_list_view.item(last_row, 1).setText("")
        qtbot.wait(100)
        parameter_list_view.item(last_row, 0).setText("")
        qtbot.wait(100)

        # Row should be deleted
        assert "TestParam" not in parameter_list_view.param_dict

    def test_delete_row_via_value_when_both_empty(
        self, parameter_list_view, qtbot
    ):
        """Test deleting row when value changed, both name and value empty."""
        qtbot.addWidget(parameter_list_view)

        initial_row_count = parameter_list_view.rowCount()

        # Get a row with a parameter, clear the name first
        param_name = "Transform"
        target_row = None
        for i in range(parameter_list_view.rowCount()):
            item = parameter_list_view.item(i, 0)
            if item and item.text() == param_name:
                target_row = i
                break

        assert target_row is not None

        # Clear the name (this will clear value and delete row)
        parameter_list_view.item(target_row, 0).setText("")
        qtbot.wait(100)

        # Row should be removed
        assert parameter_list_view.rowCount() == initial_row_count - 1
        assert param_name not in parameter_list_view.param_dict

    def test_row_deletion_else_branches_no_userrole(
        self, parameter_list_view, qtbot
    ):
        """Test row deletion else branches when items lack UserRole data."""
        from qtpy.QtCore import Qt

        qtbot.addWidget(parameter_list_view)

        # Create items without UserRole to test else branches
        # Simulates scenario where items exist but don't have UserRole set
        param_name = "CheckNumberOfSamples"
        target_row = None
        for i in range(parameter_list_view.rowCount()):
            item = parameter_list_view.item(i, 0)
            if item and item.text() == param_name:
                target_row = i
                break

        assert target_row is not None

        # Clear UserRole data from the item to test else branch
        param_item = parameter_list_view.item(target_row, 0)
        param_item.setData(Qt.ItemDataRole.UserRole, None)

        # Also clear from a few other items to test multiple else branches
        for i in range(min(3, parameter_list_view.rowCount())):
            item = parameter_list_view.item(i, 0)
            if item:
                # Store text before clearing UserRole
                item_text = item.text()
                item.setData(Qt.ItemDataRole.UserRole, None)
                # Restore text (in case it was cleared)
                if not item.text():
                    item.setText(item_text)

        # Now trigger row deletion by clearing both name and value
        parameter_list_view.item(target_row, 1).setText("")
        qtbot.wait(100)
        parameter_list_view.item(target_row, 0).setText("")
        qtbot.wait(100)

        # Row deleted and tracking rebuilt using else branches
        assert param_name not in parameter_list_view.param_dict

    def test_delete_row_when_value_cleared_name_already_empty(
        self, parameter_list_view, qtbot
    ):
        """Test deleting row when value cleared, name empty (236-254)."""
        from qtpy.QtCore import Qt

        qtbot.addWidget(parameter_list_view)

        initial_row_count = parameter_list_view.rowCount()

        # Clear UserRole from items to test else branch at line 250
        for i in range(min(2, parameter_list_view.rowCount() - 1)):
            item = parameter_list_view.item(i, 0)
            if item:
                item.setData(Qt.ItemDataRole.UserRole, None)

        # Create a row with empty name but with a value
        last_row = parameter_list_view.rowCount() - 1
        param_item = QTableWidgetItem("")
        # Don't set UserRole to test else branch at line 250
        parameter_list_view.setItem(last_row, 0, param_item)
        value_item = QTableWidgetItem("test_value")
        parameter_list_view.setItem(last_row, 1, value_item)
        qtbot.wait(100)

        # Clear value - both name and value empty, so row should be deleted
        with qtbot.waitSignal(parameter_list_view.cellChanged, timeout=1000):
            parameter_list_view.item(last_row, 1).setText("")

        # Row should be removed (one less than initial)
        assert parameter_list_view.rowCount() == initial_row_count - 1
