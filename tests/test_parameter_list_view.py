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

    def test_delete_row_by_clearing_parameter_name(self, parameter_list_view, qtbot):
        """Test that clearing parameter name removes it from dictionary."""
        qtbot.addWidget(parameter_list_view)

        # Get initial state
        initial_row_count = parameter_list_view.rowCount()
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
        parameter_list_view.item(target_row, 0).setText("")
        qtbot.wait(100)  # Wait for signals to process

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
        parameter_list_view.item(target_row, 1).setText("")
        qtbot.wait(100)  # Small delay to ensure value is cleared

        # Clear parameter name (this should trigger row deletion)
        parameter_list_view.item(target_row, 0).setText("")
        qtbot.wait(200)  # Wait for signals and row removal to process

        # Row should be removed
        assert parameter_list_view.rowCount() == initial_row_count - 1
        assert param_name not in parameter_list_view.param_dict

    def test_parameter_name_cleared_but_value_remains(
        self, parameter_list_view, qtbot
    ):
        """Test that clearing name but keeping value clears value too."""
        qtbot.addWidget(parameter_list_view)

        param_name = "BSplineInterpolationOrder"
        target_row = None
        for i in range(parameter_list_view.rowCount()):
            item = parameter_list_view.item(i, 0)
            if item and item.text() == param_name:
                target_row = i
                break

        assert target_row is not None
        original_value = parameter_list_view.item(target_row, 1).text()

        # Clear parameter name (value should also be cleared)
        parameter_list_view.item(target_row, 0).setText("")
        qtbot.wait(100)  # Wait for signals to process

        # Parameter should be removed from dict
        assert param_name not in parameter_list_view.param_dict
        # Value should be cleared
        value_item = parameter_list_view.item(target_row, 1)
        assert value_item is None or value_item.text() == ""

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
        parameter_list_view.item(target_row, 0).setText(new_name)
        qtbot.wait(100)  # Wait for signals to process

        # Old name should be removed, new name should have the value
        assert old_name not in parameter_list_view.param_dict
        assert new_name in parameter_list_view.param_dict
        assert parameter_list_view.param_dict[new_name] == original_value

    def test_delete_multiple_rows(self, parameter_list_view, qtbot):
        """Test deleting multiple rows in sequence."""
        qtbot.addWidget(parameter_list_view)

        initial_row_count = parameter_list_view.rowCount()
        params_to_delete = ["AutomaticScalesEstimation", "CheckNumberOfSamples"]

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
        parameter_list_view.setItem(last_row, 1, QTableWidgetItem("test_value"))

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
