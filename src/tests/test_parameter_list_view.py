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


@pytest.fixture(scope="class")
def parameter_list_view() -> RegistrationParameterListView:
    parameter_list_view = RegistrationParameterListView(
        param_dict=param_dict, transform_type=transform_type
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

    curr_row_count = parameter_list_view.rowCount()
    last_row_index = len(param_dict)

    parameter_list_view.setItem(
        last_row_index, 0, QTableWidgetItem("TestParameter")
    )
    parameter_list_view.setItem(last_row_index, 1, QTableWidgetItem("true"))

    assert parameter_list_view.param_dict["TestParameter"] == ["true"]
    assert parameter_list_view.rowCount() == curr_row_count + 1


def test_parameter_list_view_cell_change_last_row_no_param(
    parameter_list_view, qtbot
):
    qtbot.addWidget(parameter_list_view)

    curr_row_count = parameter_list_view.rowCount()
    last_row_index = len(param_dict)

    parameter_list_view.setItem(last_row_index, 1, QTableWidgetItem("true"))

    assert parameter_list_view.rowCount() == curr_row_count


def test_parameter_list_view_cell_change_last_row_no_value(
    parameter_list_view, qtbot
):
    qtbot.addWidget(parameter_list_view)

    curr_row_count = parameter_list_view.rowCount()
    last_row_index = len(param_dict)

    parameter_list_view.setItem(
        last_row_index, 0, QTableWidgetItem("TestParameter")
    )

    assert parameter_list_view.rowCount() == curr_row_count
