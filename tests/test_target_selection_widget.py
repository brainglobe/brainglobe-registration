import pytest

from brainglobe_registration.widgets.target_selection_widget import (
    AutoSliceDialog,
)


@pytest.fixture
def dialog(qtbot):
    dlg = AutoSliceDialog(z_max_value=200)
    qtbot.addWidget(dlg)
    return dlg


def test_default_values(dialog):
    assert dialog.z_min.value() == 0
    assert dialog.z_max.value() == 200
    assert dialog.pitch_min.value() == -5
    assert dialog.pitch_max.value() == 5
    assert dialog.yaw_min.value() == -5
    assert dialog.yaw_max.value() == 5
    assert dialog.roll_min.value() == -5
    assert dialog.roll_max.value() == 5
    assert dialog.init_points.value() == 5
    assert dialog.n_iter.value() == 15
    assert (
        dialog.metric_dropdown.currentText()
        == "Mutual Information (recommended)"
    )


def test_emit_parameters(qtbot, dialog):
    dialog.z_min.setValue(10)
    dialog.z_max.setValue(20)
    dialog.pitch_min.setValue(-10)
    dialog.pitch_max.setValue(10)
    dialog.yaw_min.setValue(-15)
    dialog.yaw_max.setValue(15)
    dialog.roll_min.setValue(-20)
    dialog.roll_max.setValue(20)
    dialog.init_points.setValue(7)
    dialog.n_iter.setValue(25)
    dialog.metric_dropdown.setCurrentText("Combined")

    with qtbot.waitSignal(
        dialog.parameters_confirmed, timeout=1000
    ) as blocker:
        dialog.accept()

    params = blocker.args[0]
    assert params["z_range"] == (10, 20)
    assert params["pitch_bounds"] == (-10, 10)
    assert params["yaw_bounds"] == (-15, 15)
    assert params["roll_bounds"] == (-20, 20)
    assert params["init_points"] == 7
    assert params["n_iter"] == 25
    assert params["metric"] == "combined"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Mutual Information (recommended)", "mi"),
        ("Normalised Cross-Correlation", "ncc"),
        ("Structural Similarity Index", "ssim"),
        ("Combined", "combined"),
    ],
)
def test_metric_mapping(qtbot, dialog, text, expected):
    dialog.metric_dropdown.setCurrentText(text)

    with qtbot.waitSignal(
        dialog.parameters_confirmed, timeout=1000
    ) as blocker:
        dialog.accept()

    params = blocker.args[0]
    assert params["metric"] == expected
