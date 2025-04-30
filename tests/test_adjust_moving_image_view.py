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


def test_init(qtbot, adjust_moving_image_view):
    qtbot.addWidget(adjust_moving_image_view)

    assert adjust_moving_image_view.layout().rowCount() == 11


@pytest.mark.parametrize(
    "x_scale, y_scale, z_scale",
    [(2.5, 2.5, 5.0), (10, 20, 30), (10.2212, 10.2289, 10.2259)],
)
def test_scale_image_button_xy_click(
    qtbot, adjust_moving_image_view, x_scale, y_scale, z_scale
):
    qtbot.addWidget(adjust_moving_image_view)

    with qtbot.waitSignal(
        adjust_moving_image_view.scale_image_signal, timeout=1000
    ) as blocker:
        adjust_moving_image_view.adjust_moving_image_pixel_size_x.setValue(
            x_scale
        )
        adjust_moving_image_view.adjust_moving_image_pixel_size_y.setValue(
            y_scale
        )
        adjust_moving_image_view.adjust_moving_image_pixel_size_z.setValue(
            z_scale
        )
        adjust_moving_image_view.scale_moving_image_button.click()

    assert blocker.args == [
        round(x_scale, 3),
        round(y_scale, 3),
        round(z_scale, 3),
    ]


def test_atlas_rotation_changed(
    qtbot,
    adjust_moving_image_view,
):
    qtbot.addWidget(adjust_moving_image_view)

    with qtbot.waitSignal(
        adjust_moving_image_view.atlas_rotation_signal, timeout=1000
    ) as blocker:
        adjust_moving_image_view.adjust_atlas_pitch.setValue(10)
        adjust_moving_image_view.adjust_atlas_yaw.setValue(20)
        adjust_moving_image_view.adjust_atlas_roll.setValue(30)

        adjust_moving_image_view.adjust_atlas_rotation.click()

    assert blocker.args == [10, 20, 30]


def test_atlas_reset_button_click(
    qtbot,
    adjust_moving_image_view,
):
    qtbot.addWidget(adjust_moving_image_view)

    with qtbot.waitSignal(
        adjust_moving_image_view.reset_atlas_signal, timeout=1000
    ):
        adjust_moving_image_view.reset_atlas_button.click()

    assert (
        adjust_moving_image_view.adjust_atlas_pitch.value() == 0
        and adjust_moving_image_view.adjust_atlas_yaw.value() == 0
        and adjust_moving_image_view.adjust_atlas_roll.value() == 0
    )
