from unittest.mock import MagicMock, patch

import pytest
from qtpy.QtWidgets import QApplication

from brainglobe_registration.widgets.similarity_metrics_widget import (
    AVAILABLE_METRICS,
    SimilarityWidget,
)


# Working with this: pytest qtbot https://pytest-qt.readthedocs.io/en/latest/
@pytest.fixture
def similarity_widget(qtbot) -> SimilarityWidget:
    """Fixture to create a similarity widget for testing."""
    widget = SimilarityWidget()
    qtbot.addWidget(widget)
    return widget


def test_initialization(similarity_widget: SimilarityWidget):
    """Test the initialization of the similarity widget."""
    assert similarity_widget.title() == "Find Best Slice"
    assert not similarity_widget.custom_range_checkbox.isChecked()
    assert not similarity_widget.start_slice_spinbox.isEnabled()
    assert not similarity_widget.end_slice_spinbox.isEnabled()
    assert (
        similarity_widget.metric_combobox.currentText() == AVAILABLE_METRICS[0]
    )
    assert similarity_widget.find_button.isEnabled()
    assert similarity_widget.metric_combobox.count() == len(AVAILABLE_METRICS)


def test_custom_range_checkbox(similarity_widget: SimilarityWidget):
    """Test the custom range checkbox toggle"""
    # enable the range
    similarity_widget.custom_range_checkbox.setChecked(True)
    assert similarity_widget.start_slice_spinbox.isEnabled()
    assert similarity_widget.end_slice_spinbox.isEnabled()

    # disable the range
    similarity_widget.custom_range_checkbox.setChecked(False)
    assert not similarity_widget.start_slice_spinbox.isEnabled()
    assert not similarity_widget.end_slice_spinbox.isEnabled()


@pytest.mark.parametrize(
    "min_val, max_val, expected_min, expected_max, expected_value",
    [
        (10, 200, 10, 200, 10),  # normal case
        (50, 40, 50, 50, 50),  # max < min case
    ],
)
def test_set_slice_range_limits_start_spinbox(
    similarity_widget: SimilarityWidget,
    min_val,
    max_val,
    expected_min,
    expected_max,
    expected_value,
):
    similarity_widget.set_slice_range_limits(min_val, max_val)
    assert similarity_widget.start_slice_spinbox.minimum() == expected_min
    assert similarity_widget.start_slice_spinbox.maximum() == expected_max
    assert similarity_widget.start_slice_spinbox.value() == expected_value


@pytest.mark.parametrize(
    "min_val, max_val, expected_min, expected_max, expected_value",
    [
        (10, 200, 10, 200, 200),  # normal case: value should be max
        (50, 40, 50, 50, 50),  # max < min case: value should be min
    ],
)
def test_set_slice_range_limits_end_spinbox(
    similarity_widget: SimilarityWidget,
    min_val,
    max_val,
    expected_min,
    expected_max,
    expected_value,
):
    similarity_widget.set_slice_range_limits(min_val, max_val)
    assert similarity_widget.end_slice_spinbox.minimum() == expected_min
    assert similarity_widget.end_slice_spinbox.maximum() == expected_max
    assert similarity_widget.end_slice_spinbox.value() == expected_value


def test_find_button_default_range(qtbot, similarity_widget: SimilarityWidget):
    """Test the find button click with the default range"""
    min_slice, max_slice = 0, 100
    similarity_widget.set_slice_range_limits(min_slice, max_slice)
    similarity_widget.custom_range_checkbox.setChecked(False)
    selected_metric = AVAILABLE_METRICS[0]  ## default is : ncc

    with qtbot.waitSignal(
        similarity_widget.calculate_metric_requested, timeout=5000
    ) as blocker:
        similarity_widget.find_button.click()
    assert blocker.args == [min_slice, max_slice, selected_metric]


def test_find_button_custom_range(qtbot, similarity_widget: SimilarityWidget):
    """Test the find button click with a custom range"""
    similarity_widget.set_slice_range_limits(0, 100)  # full range
    similarity_widget.custom_range_checkbox.setChecked(True)

    custom_start, custom_end = 20, 80
    similarity_widget.start_slice_spinbox.setValue(custom_start)
    similarity_widget.end_slice_spinbox.setValue(custom_end)
    selected_metric = AVAILABLE_METRICS[1]  # eg mi
    similarity_widget.metric_combobox.setCurrentText(selected_metric)

    with qtbot.waitSignal(
        similarity_widget.calculate_metric_requested, timeout=5000
    ) as blocker:
        similarity_widget.find_button.click()

    assert blocker.args == [custom_start, custom_end, selected_metric]


def test_find_button_invalid_range(qtbot, similarity_widget: SimilarityWidget):
    """Test the find button with an invalid range"""
    similarity_widget.set_slice_range_limits(0, 100)
    similarity_widget.custom_range_checkbox.setChecked(True)

    custom_start, custom_end = 100, 20  # invalid range
    similarity_widget.start_slice_spinbox.setValue(custom_start)
    similarity_widget.end_slice_spinbox.setValue(custom_end)

    # create a mock slot
    mock_slot = MagicMock()

    # connect the mock slot to the signal
    similarity_widget.calculate_metric_requested.connect(mock_slot)

    # Patch display_warning to check it is called
    with patch(
        "brainglobe_registration.widgets.similarity_metrics_widget.display_warning"
    ) as mock_warning:
        similarity_widget.find_button.click()
        QApplication.processEvents()
        mock_warning.assert_called_once_with(
            similarity_widget,
            "Slice Range Error",
            "Start slice cannot be greater than end slice.",
        )

    # disconnect the mock slot
    similarity_widget.calculate_metric_requested.disconnect(mock_slot)

    # assert the mock slot was not called
    mock_slot.assert_not_called()


@pytest.mark.parametrize("metric_to_test", AVAILABLE_METRICS)
def test_metric_selection_emitted(
    qtbot, similarity_widget: SimilarityWidget, metric_to_test: str
):
    """Test that selected metric is emitted when find button is clicked"""
    min_slice, max_slice = 5, 15
    similarity_widget.set_slice_range_limits(min_slice, max_slice)
    similarity_widget.custom_range_checkbox.setChecked(
        False
    )  # use default range
    similarity_widget.metric_combobox.setCurrentText(metric_to_test)

    with qtbot.waitSignal(
        similarity_widget.calculate_metric_requested, timeout=5000
    ) as blocker:
        similarity_widget.find_button.click()

    assert blocker.args == [min_slice, max_slice, metric_to_test]


def test_on_group_toggled_disables_children(
    similarity_widget: SimilarityWidget,
):
    """Test that _on_group_toggled(False) disables child widgets."""
    similarity_widget.custom_range_checkbox.setChecked(
        True
    )  # Enables spinboxes

    similarity_widget._on_group_toggled(False)

    assert not similarity_widget.custom_range_checkbox.isEnabled()
    assert not similarity_widget.start_slice_spinbox.isEnabled()
    assert not similarity_widget.end_slice_spinbox.isEnabled()
    assert not similarity_widget.metric_combobox.isEnabled()
    assert not similarity_widget.find_button.isEnabled()
    # Check a label too
    assert (
        not similarity_widget.layout()
        .itemAt(1)
        .layout()
        .itemAt(0)
        .widget()
        .isEnabled()
    )  # Label "Start:"


def test_on_group_toggled_enables_custom_range(
    similarity_widget: SimilarityWidget,
):
    """Test _on_group_toggled(True) enables custom_range_checkbox."""
    # custom_range_checkbox is unchecked when group is re-enabled
    similarity_widget.custom_range_checkbox.setChecked(False)
    similarity_widget._on_group_toggled(False)
    similarity_widget._on_group_toggled(True)

    assert similarity_widget.custom_range_checkbox.isEnabled()
    assert (
        not similarity_widget.start_slice_spinbox.isEnabled()
    )  # Should be False
    assert (
        not similarity_widget.end_slice_spinbox.isEnabled()
    )  # Should be False
    assert similarity_widget.metric_combobox.isEnabled()
    assert similarity_widget.find_button.isEnabled()

    # custom_range_checkbox is checked when group is re-enabled
    similarity_widget.custom_range_checkbox.setChecked(True)
    similarity_widget._on_group_toggled(False)  # Disable all
    # The state of custom_range_checkbox is True before
    # _on_group_toggled(True) is called
    similarity_widget._on_group_toggled(True)  # Re-enable

    assert similarity_widget.custom_range_checkbox.isEnabled()
    assert similarity_widget.start_slice_spinbox.isEnabled()  # Should be True
    assert similarity_widget.end_slice_spinbox.isEnabled()  # Should be True
    assert similarity_widget.metric_combobox.isEnabled()
    assert similarity_widget.find_button.isEnabled()
