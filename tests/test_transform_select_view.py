import pytest

from brainglobe_registration.widgets.transform_select_view import (
    TransformSelectView,
)


@pytest.fixture(scope="class")
def transform_select_view() -> TransformSelectView:
    transform_select_view = TransformSelectView()
    return transform_select_view


def test_transform_select_view(transform_select_view, qtbot):
    qtbot.addWidget(transform_select_view)

    assert (
        transform_select_view.horizontalHeaderItem(0).text()
        == "Transform Type"
    )
    assert (
        transform_select_view.horizontalHeaderItem(1).text() == "File"
    )
    assert transform_select_view.horizontalHeaderItem(2).text() == "Import"
    assert transform_select_view.horizontalHeaderItem(3).text() == "Export"

    # At initialisation only affine and bspline transforms are shown
    assert transform_select_view.rowCount() == len(
        transform_select_view.transform_type_options
    )
    assert transform_select_view.columnCount() == 4

    # At initialisation affine and bspline transforms are shown
    # with ara_tools as the default file
    for i in range(len(transform_select_view.transform_type_options) - 1):
        assert (
            transform_select_view.cellWidget(i, 0).currentText()
            == transform_select_view.transform_type_options[i + 1]
        )
        assert (
            transform_select_view.cellWidget(i, 1).currentText()
            == transform_select_view.file_options[1]
        )
        assert transform_select_view.cellWidget(i, 2).isEnabled()
        assert transform_select_view.cellWidget(i, 3).isEnabled()

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
    assert not transform_select_view.cellWidget(last_row, 2).isEnabled()
    assert not transform_select_view.cellWidget(last_row, 3).isEnabled()


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
    assert transform_select_view.cellWidget(row_count - 1, 2).isEnabled()
    assert transform_select_view.cellWidget(row_count - 1, 3).isEnabled()
    assert not transform_select_view.cellWidget(row_count, 2).isEnabled()
    assert not transform_select_view.cellWidget(row_count, 3).isEnabled()

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


def test_transform_type_removed(transform_select_view, qtbot):
    qtbot.addWidget(transform_select_view)

    row_count = transform_select_view.rowCount()

    with qtbot.waitSignal(
        transform_select_view.transform_type_removed_signal, timeout=1000
    ) as blocker:
        transform_select_view.cellWidget(0, 0).setCurrentIndex(0)

    assert blocker.args == [0]
    assert transform_select_view.rowCount() == row_count - 1

    with qtbot.waitSignal(
        transform_select_view.transform_type_removed_signal, timeout=1000
    ) as blocker:
        transform_select_view.cellWidget(0, 0).setCurrentIndex(0)

    assert blocker.args == [0]
    assert transform_select_view.rowCount() == row_count - 2


def test_file_option_default_on_transform_change(transform_select_view, qtbot):
    """
    When the transform type is changed, the file option should be set to the
    default
    """
    qtbot.addWidget(transform_select_view)
    file_option_index = 2

    transform_select_view.cellWidget(0, 1).setCurrentIndex(file_option_index)
    assert (
        transform_select_view.cellWidget(0, 1).currentText()
        == transform_select_view.file_options[file_option_index]
    )

    # Change the transform type from affine to bspline
    transform_select_view.cellWidget(0, 0).setCurrentIndex(2)

    assert (
        transform_select_view.cellWidget(0, 1).currentText()
        == transform_select_view.file_options[0]
    )


def test_import_button_clicked_signal(transform_select_view, qtbot):
    qtbot.addWidget(transform_select_view)

    with qtbot.waitSignal(
        transform_select_view.import_button_clicked_signal, timeout=1000
    ) as blocker:
        transform_select_view.cellWidget(0, 2).click()

    assert blocker.args == [0]


def test_export_button_clicked_signal(transform_select_view, qtbot):
    qtbot.addWidget(transform_select_view)

    with qtbot.waitSignal(
        transform_select_view.export_button_clicked_signal, timeout=1000
    ) as blocker:
        transform_select_view.cellWidget(0, 3).click()

    assert blocker.args == [0]


def test_set_transform_type_selection(transform_select_view, qtbot):
    qtbot.addWidget(transform_select_view)

    transform_select_view.set_transform_type_selection(0, "bspline")

    assert transform_select_view.cellWidget(0, 0).currentText() == "bspline"


def test_set_custom_file_path(transform_select_view, qtbot):
    qtbot.addWidget(transform_select_view)

    custom_file_path = "C:/tmp/custom_affine.txt"
    transform_select_view.set_custom_file_path(0, custom_file_path)

    assert (
        transform_select_view.cellWidget(0, 1).currentText()
        == "(Custom) custom_affine.txt"
    )
    assert (
        transform_select_view.cellWidget(0, 1).toolTip()
        == custom_file_path
    )


def test_clear_custom_file(transform_select_view, qtbot):
    qtbot.addWidget(transform_select_view)

    custom_file_path = "C:/tmp/custom_affine.txt"
    transform_select_view.set_custom_file_path(0, custom_file_path)

    assert (
        transform_select_view.cellWidget(0, 1).currentText()
        == "(Custom) custom_affine.txt"
    )

    transform_select_view.clear_custom_file(0)

    # Ensure custom option is removed
    combo_box = transform_select_view.cellWidget(0, 1)
    for i in range(combo_box.count()):
        assert not combo_box.itemText(i).startswith("(Custom)")
    
    assert combo_box.toolTip() == ""
