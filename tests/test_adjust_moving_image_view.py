import pytest

from brainglobe_registration.widgets.adjust_moving_image_view import (
    AdjustMovingImageView,
)

max_translate = 2000
max_rotate = 360


@pytest.fixture(scope="class")
def adjust_moving_image_view() -> AdjustMovingImageView:
    adjust_moving_image_view = AdjustMovingImageView()
    return adjust_moving_image_view


@pytest.mark.parametrize(
    "x_value, expected",
    [
        (-100, -100),
        (100, 100),
        (max_translate + 1, max_translate),
        (-1 * (max_translate + 1), -1 * max_translate),
    ],
)
def test_x_position_changed(
    qtbot, adjust_moving_image_view, x_value, expected
):
    qtbot.addWidget(adjust_moving_image_view)

    with qtbot.waitSignal(
        adjust_moving_image_view.adjust_image_signal, timeout=1000
    ) as blocker:
        adjust_moving_image_view.adjust_moving_image_x.setValue(x_value)

    assert blocker.args == [expected, 0, 0]


@pytest.mark.parametrize(
    "y_value, expected",
    [
        (-100, -100),
        (100, 100),
        (max_translate + 1, max_translate),
        (-1 * (max_translate + 1), -1 * max_translate),
    ],
)
def test_y_position_changed(
    qtbot, adjust_moving_image_view, y_value, expected
):
    qtbot.addWidget(adjust_moving_image_view)

    with qtbot.waitSignal(
        adjust_moving_image_view.adjust_image_signal, timeout=1000
    ) as blocker:
        adjust_moving_image_view.adjust_moving_image_y.setValue(y_value)

    assert blocker.args == [0, expected, 0]


@pytest.mark.parametrize(
    "rotate_value, expected",
    [
        (-100, -100),
        (100, 100),
        (10.5, 10.5),
        (max_rotate + 1, max_rotate),
        (-1 * (max_rotate + 1), -1 * max_rotate),
    ],
)
def test_rotation_position_changed(
    qtbot, adjust_moving_image_view, rotate_value, expected
):
    qtbot.addWidget(adjust_moving_image_view)

    with qtbot.waitSignal(
        adjust_moving_image_view.adjust_image_signal, timeout=1000
    ) as blocker:
        adjust_moving_image_view.adjust_moving_image_rotate.setValue(
            rotate_value
        )

    assert blocker.args == [0, 0, expected]


def test_reset_image_button_click(qtbot, adjust_moving_image_view):
    qtbot.addWidget(adjust_moving_image_view)

    with qtbot.waitSignal(
        adjust_moving_image_view.reset_image_signal, timeout=1000
    ):
        adjust_moving_image_view.adjust_moving_image_reset_button.click()

    assert adjust_moving_image_view.adjust_moving_image_x.value() == 0
    assert adjust_moving_image_view.adjust_moving_image_y.value() == 0
    assert adjust_moving_image_view.adjust_moving_image_rotate.value() == 0
