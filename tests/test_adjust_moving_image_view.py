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

    assert adjust_moving_image_view.layout().rowCount() == 15


@pytest.mark.parametrize(
    "x_scale, y_scale, z_scale, orientation",
    [
        (2.5, 2.5, 5.0, "prs"),
        (10, 20, 30, "asr"),
        (10.2212, 10.2289, 10.2259, "las"),
    ],
)
def test_scale_image_button_xy_click(
    qtbot, adjust_moving_image_view, x_scale, y_scale, z_scale, orientation
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
        adjust_moving_image_view.data_orientation_field.setText(orientation)
        adjust_moving_image_view.scale_moving_image_button.click()

    assert blocker.args == [
        round(x_scale, 3),
        round(y_scale, 3),
        round(z_scale, 3),
        orientation,
    ]


def test_atlas_rotation_changed(
    qtbot,
    adjust_moving_image_view,
):
    qtbot.addWidget(adjust_moving_image_view)

    # Block signals while setting all values to avoid partial emissions
    adjust_moving_image_view.adjust_atlas_pitch.blockSignals(True)
    adjust_moving_image_view.adjust_atlas_yaw.blockSignals(True)
    adjust_moving_image_view.adjust_atlas_roll.blockSignals(True)

    # Set slider values (in tenths of degrees)
    adjust_moving_image_view.adjust_atlas_pitch.setValue(100)  # 10 deg
    adjust_moving_image_view.adjust_atlas_yaw.setValue(200)  # 20 deg
    adjust_moving_image_view.adjust_atlas_roll.setValue(300)  # 30 deg

    adjust_moving_image_view.adjust_atlas_pitch.blockSignals(False)
    adjust_moving_image_view.adjust_atlas_yaw.blockSignals(False)
    adjust_moving_image_view.adjust_atlas_roll.blockSignals(False)

    # Trigger slider changed to emit signal with all values set
    with qtbot.waitSignal(
        adjust_moving_image_view.atlas_rotation_signal, timeout=1000
    ) as blocker:
        adjust_moving_image_view._on_slider_changed()
        qtbot.wait(10)

    # Signal emits degrees (slider value / 10)
    assert blocker.args == [10.0, 20.0, 30.0]


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


def test_moving_image_reset_button_click(qtbot, adjust_moving_image_view):
    qtbot.addWidget(adjust_moving_image_view)

    with qtbot.waitSignal(
        adjust_moving_image_view.reset_moving_image_signal, timeout=1000
    ):
        adjust_moving_image_view.reset_moving_image_button.click()


@pytest.mark.parametrize("is_3d", [True, False])
def test_set_is_3d_visibility(qtbot, is_3d):
    widget = AdjustMovingImageView()
    qtbot.addWidget(widget)
    widget.show()

    widget.set_is_3d(is_3d)

    assert widget.adjust_moving_image_pixel_size_z.isVisible() == is_3d
    assert widget.z_row_label.isVisible() == is_3d
    assert widget.data_orientation_field.isVisible() == is_3d
    assert widget.orientation_row_label.isVisible() == is_3d


def test_set_rotation_values(qtbot, adjust_moving_image_view):
    """Test programmatic setting of rotation slider values."""
    qtbot.addWidget(adjust_moving_image_view)

    adjust_moving_image_view.set_rotation_values(15.5, 30.0, -45.0)

    # Values are stored as tenths of degrees
    assert adjust_moving_image_view.adjust_atlas_pitch.value() == 155
    assert adjust_moving_image_view.adjust_atlas_yaw.value() == 300
    assert adjust_moving_image_view.adjust_atlas_roll.value() == -450


def test_slider_throttle(qtbot, adjust_moving_image_view):
    """Test that slider changes are throttled (fires immediately, then rate-limited)."""
    qtbot.addWidget(adjust_moving_image_view)

    # First change should fire immediately (throttle is ready)
    with qtbot.waitSignal(
        adjust_moving_image_view.atlas_rotation_signal, timeout=100
    ) as blocker:
        adjust_moving_image_view.adjust_atlas_pitch.setValue(100)

    assert blocker.args[0] == 10.0  # 100 / 10

    # Throttle timer should now be in cooldown
    assert adjust_moving_image_view._rotation_throttle_timer.is_active()

    # Rapid changes during cooldown get queued
    adjust_moving_image_view.adjust_atlas_pitch.setValue(200)
    adjust_moving_image_view.adjust_atlas_pitch.setValue(300)

    # Wait for cooldown to end and final value to fire
    with qtbot.waitSignal(
        adjust_moving_image_view.atlas_rotation_signal, timeout=100
    ) as blocker:
        qtbot.wait(10)

    # Final value should be emitted after cooldown
    assert blocker.args[0] == 30.0  # 300 / 10


def test_interpolation_order_changed(qtbot, adjust_moving_image_view):
    """Test interpolation order dropdown signal."""
    qtbot.addWidget(adjust_moving_image_view)

    with qtbot.waitSignal(
        adjust_moving_image_view.interpolation_order_changed, timeout=1000
    ) as blocker:
        adjust_moving_image_view.interpolation_order_dropdown.setCurrentIndex(0)

    assert blocker.args == [0]


def test_get_interpolation_order(qtbot, adjust_moving_image_view):
    """Test getting current interpolation order."""
    qtbot.addWidget(adjust_moving_image_view)

    adjust_moving_image_view.interpolation_order_dropdown.setCurrentIndex(0)
    assert adjust_moving_image_view.get_interpolation_order() == 0

    adjust_moving_image_view.interpolation_order_dropdown.setCurrentIndex(1)
    assert adjust_moving_image_view.get_interpolation_order() == 1
