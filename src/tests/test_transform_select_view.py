import pytest

from brainglobe_registration.widgets.transform_select_view import (
    TransformSelectView,
)


@pytest.fixture(scope="class")
def transform_select_view() -> TransformSelectView:
    transform_select_view = TransformSelectView()
    yield transform_select_view


def test_transform_select_view(transform_select_view, qtbot):
    qtbot.addWidget(transform_select_view)

    assert (
        transform_select_view.horizontalHeaderItem(0).text()
        == "Transform Type"
    )
    assert (
        transform_select_view.horizontalHeaderItem(1).text() == "Default File"
    )

    assert transform_select_view.rowCount() == len(
        transform_select_view.transform_type_options
    )
    assert transform_select_view.columnCount() == 2

    for i in range(len(transform_select_view.transform_type_options) - 1):
        assert (
            transform_select_view.cellWidget(i, 0).currentText()
            == transform_select_view.transform_type_options[i + 1]
        )
        assert (
            transform_select_view.cellWidget(i, 1).currentText()
            == transform_select_view.file_options[0]
        )

    last_row = len(transform_select_view.transform_type_options) - 1

    assert (
        transform_select_view.cellWidget(last_row, 0).currentText()
        == transform_select_view.transform_type_options[0]
    )
    assert (
        transform_select_view.cellWidget(last_row, 1).currentText()
        == transform_select_view.file_options[0]
    )
    assert not transform_select_view.cellWidget(last_row, 1).isEnabled()


def test_transform_type_added_signal(transform_select_view, qtbot):
    qtbot.addWidget(transform_select_view)

    last_index = len(transform_select_view.transform_type_options) - 1
    row_count = transform_select_view.rowCount()

    with qtbot.waitSignal(
        transform_select_view.transform_type_added_signal, timeout=1000
    ) as blocker:
        transform_select_view.cellWidget(0, 0).setCurrentIndex(last_index)

    assert blocker.args == [
        transform_select_view.transform_type_options[last_index],
        0,
    ]
    assert transform_select_view.rowCount() == row_count


def test_file_option_changed_signal(transform_select_view, qtbot):
    qtbot.addWidget(transform_select_view)

    last_index = len(transform_select_view.file_options) - 1
    row_count = transform_select_view.rowCount()

    with qtbot.waitSignal(
        transform_select_view.file_option_changed_signal, timeout=1000
    ) as blocker:
        transform_select_view.cellWidget(0, 1).setCurrentIndex(last_index)

    assert blocker.args == [transform_select_view.file_options[last_index], 0]
    assert transform_select_view.rowCount() == row_count


def test_transform_type_added_signal_last_row(transform_select_view, qtbot):
    qtbot.addWidget(transform_select_view)

    last_index = len(transform_select_view.transform_type_options) - 1
    row_count = transform_select_view.rowCount()

    with qtbot.waitSignal(
        transform_select_view.transform_type_added_signal, timeout=1000
    ) as blocker:
        transform_select_view.cellWidget(row_count - 1, 0).setCurrentIndex(
            last_index
        )

    assert blocker.args == [
        transform_select_view.transform_type_options[last_index],
        row_count - 1,
    ]
    assert transform_select_view.rowCount() == row_count + 1
    assert (
        transform_select_view.cellWidget(row_count, 0).currentText()
        == transform_select_view.transform_type_options[0]
    )
    assert (
        transform_select_view.cellWidget(row_count, 1).currentText()
        == transform_select_view.file_options[0]
    )
    assert transform_select_view.cellWidget(row_count - 1, 1).isEnabled()
    assert not transform_select_view.cellWidget(row_count, 1).isEnabled()

    with qtbot.waitSignal(
        transform_select_view.transform_type_added_signal, timeout=1000
    ) as blocker:
        transform_select_view.cellWidget(row_count, 0).setCurrentIndex(
            last_index
        )

    assert blocker.args == [
        transform_select_view.transform_type_options[last_index],
        row_count,
    ]
