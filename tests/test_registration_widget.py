import pytest

from brainglobe_registration.registration_widget import RegistrationWidget


@pytest.fixture()
def registration_widget(make_napari_viewer_with_images):
    viewer = make_napari_viewer_with_images

    widget = RegistrationWidget(viewer)

    return widget


@pytest.fixture()
def registration_widget_with_example_atlas(make_napari_viewer_with_images):
    """
    Create an initialised RegistrationWidget with the "example_mouse_100um"
    loaded.

    Parameters
    ------------
    make_napari_viewer_with_images for testing
        Fixture that creates a napari viewer
    """
    viewer = make_napari_viewer_with_images

    widget = RegistrationWidget(viewer)

    # Based on the downloaded atlases by the fixture in conftest.py,
    # the example atlas will be in the third position.
    example_atlas_index = -1
    for i, atlas in enumerate(widget._available_atlases):
        if atlas == "example_mouse_100um":
            example_atlas_index = i
            break

    widget._on_atlas_dropdown_index_changed(example_atlas_index)

    return widget


def test_registration_widget(make_napari_viewer_with_images):
    widget = RegistrationWidget(make_napari_viewer_with_images)

    assert widget is not None


def test_atlas_dropdown_index_changed_with_valid_index(
    make_napari_viewer_with_images, registration_widget
):
    registration_widget._on_atlas_dropdown_index_changed(2)

    assert (
        registration_widget._atlas.atlas_name
        == registration_widget._available_atlases[2]
    )
    assert registration_widget.run_button.isEnabled()
    assert registration_widget._viewer.grid.enabled


def test_atlas_dropdown_index_changed_with_zero_index(
    make_napari_viewer_with_images, registration_widget
):
    registration_widget._on_atlas_dropdown_index_changed(0)

    assert registration_widget._atlas is None
    assert not registration_widget.run_button.isEnabled()
    assert not registration_widget._viewer.grid.enabled


def test_atlas_dropdown_index_changed_with_existing_atlas(
    make_napari_viewer_with_images, registration_widget
):
    registration_widget._on_atlas_dropdown_index_changed(2)

    registration_widget._on_atlas_dropdown_index_changed(1)

    assert (
        registration_widget._atlas.atlas_name
        == registration_widget._available_atlases[1]
    )
    assert registration_widget.run_button.isEnabled()
    assert registration_widget._viewer.grid.enabled


def test_sample_dropdown_index_changed_with_valid_index(
    make_napari_viewer_with_images, registration_widget
):
    registration_widget._on_sample_dropdown_index_changed(1)

    assert (
        registration_widget._moving_image.name
        == registration_widget._sample_images[1]
    )


def test_scale_moving_image_no_atlas(
    make_napari_viewer_with_images, registration_widget, mocker
):
    mocked_show_error = mocker.patch(
        "brainglobe_registration.registration_widget.show_error"
    )
    registration_widget._atlas = None
    registration_widget.adjust_moving_image_widget.scale_image_signal.emit(
        10, 10
    )
    mocked_show_error.assert_called_once_with(
        "Sample image or atlas not selected. "
        "Please select a sample image and atlas before scaling"
    )


def test_scale_moving_image_no_sample_image(
    make_napari_viewer_with_images, registration_widget, mocker
):
    mocked_show_error = mocker.patch(
        "brainglobe_registration.registration_widget.show_error"
    )
    registration_widget._moving_image = None
    registration_widget.adjust_moving_image_widget.scale_image_signal.emit(
        10, 10
    )
    mocked_show_error.assert_called_once_with(
        "Sample image or atlas not selected. "
        "Please select a sample image and atlas before scaling"
    )


@pytest.mark.parametrize(
    "x_scale_factor, y_scale_factor",
    [
        (0.5, 0.5),
        (1.0, 1.0),
        (2.0, 2.0),
        (0.5, 1.0),
        (1.0, 0.5),
    ],
)
def test_scale_moving_image(
    make_napari_viewer_with_images,
    registration_widget,
    mocker,
    x_scale_factor,
    y_scale_factor,
):
    mock_atlas = mocker.patch(
        "brainglobe_registration.registration_widget.BrainGlobeAtlas"
    )
    mock_atlas.resolution = [20, 20, 20]
    registration_widget._atlas = mock_atlas

    current_size = registration_widget._moving_image.data.shape
    registration_widget.adjust_moving_image_widget.scale_image_signal.emit(
        mock_atlas.resolution[1] * x_scale_factor,
        mock_atlas.resolution[2] * y_scale_factor,
    )

    assert registration_widget._moving_image.data.shape == (
        current_size[0] * y_scale_factor,
        current_size[1] * x_scale_factor,
    )


@pytest.mark.parametrize(
    "pitch, yaw, roll, expected_shape",
    [
        (0, 0, 0, (132, 80, 114)),
        (45, 0, 0, (150, 150, 114)),
        (0, 45, 0, (174, 80, 174)),
        (0, 0, 45, (132, 137, 137)),
        (0, 90, 90, (80, 114, 132)),
    ],
)
def test_on_adjust_atlas_rotation(
    registration_widget_with_example_atlas,
    pitch,
    yaw,
    roll,
    expected_shape,
):
    reg_widget = registration_widget_with_example_atlas
    atlas_shape = reg_widget._atlas.reference.shape

    reg_widget._on_adjust_atlas_rotation(pitch, yaw, roll)

    assert reg_widget._atlas_data_layer.data.shape == expected_shape
    assert reg_widget._atlas_annotations_layer.data.shape == expected_shape
    assert reg_widget._atlas.reference.shape == atlas_shape


def test_on_adjust_atlas_rotation_no_atlas(registration_widget, mocker):
    mocked_show_error = mocker.patch(
        "brainglobe_registration.registration_widget.show_error"
    )
    registration_widget._on_adjust_atlas_rotation(10, 10, 10)
    mocked_show_error.assert_called_once_with(
        "No atlas selected. Please select an atlas before rotating"
    )


def test_on_atlas_reset(registration_widget_with_example_atlas):
    reg_widget = registration_widget_with_example_atlas
    atlas_shape = reg_widget._atlas.reference.shape
    reg_widget._on_adjust_atlas_rotation(10, 10, 10)

    reg_widget._on_atlas_reset()

    assert reg_widget._atlas_data_layer.data.shape == atlas_shape
    assert reg_widget._atlas.reference.shape == atlas_shape
    assert reg_widget._atlas_annotations_layer.data.shape == atlas_shape


def test_on_atlas_reset_no_atlas(registration_widget, mocker):
    mocked_show_error = mocker.patch(
        "brainglobe_registration.registration_widget.show_error"
    )

    registration_widget._on_atlas_reset()
    mocked_show_error.assert_called_once_with(
        "No atlas selected. Please select an atlas before resetting"
    )
