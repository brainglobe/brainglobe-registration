import logging
from pathlib import Path

import numpy as np
import pytest
from brainglobe_atlasapi.descriptors import ANNOTATION_DTYPE, REFERENCE_DTYPE
from brainglobe_space import AnatomicalSpace
from tifffile import imread

from brainglobe_registration.registration_widget import RegistrationWidget
from brainglobe_registration.utils.logging import (
    StripANSIColorFilter,
)


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
    assert registration_widget._atlas_data_layer is not None
    assert registration_widget._atlas_annotations_layer is not None
    assert registration_widget._atlas_data_layer.data.dtype == REFERENCE_DTYPE
    assert (
        registration_widget._atlas_annotations_layer.data.dtype
        == ANNOTATION_DTYPE
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
    atlas_backup = registration_widget._atlas
    registration_widget._atlas = None
    registration_widget.adjust_moving_image_widget.scale_image_signal.emit(
        10, 10, 10, "prs"
    )
    mocked_show_error.assert_called_once_with(
        "Sample image or atlas not selected. "
        "Please select a sample image and atlas before scaling"
    )
    registration_widget._atlas = atlas_backup


def test_scale_moving_image_no_sample_image(
    make_napari_viewer_with_images, registration_widget, mocker
):
    mocked_show_error = mocker.patch(
        "brainglobe_registration.registration_widget.show_error"
    )
    image_backup = registration_widget._moving_image
    registration_widget._moving_image = None
    registration_widget.adjust_moving_image_widget.scale_image_signal.emit(
        10, 10, 10, "prs"
    )
    mocked_show_error.assert_called_once_with(
        "Sample image or atlas not selected. "
        "Please select a sample image and atlas before scaling"
    )
    registration_widget._moving_image = image_backup


@pytest.mark.parametrize(
    "x_res, y_res",
    [
        (0, 10),
        (10, 0),
    ],
)
def test_scale_moving_image_wrong_scale(
    make_napari_viewer_with_images, registration_widget, mocker, x_res, y_res
):
    mocked_show_error = mocker.patch(
        "brainglobe_registration.registration_widget.show_error"
    )
    mock_atlas = mocker.patch(
        "brainglobe_registration.registration_widget.BrainGlobeAtlas"
    )
    mock_atlas.resolution = [20, 20, 20]
    registration_widget._atlas = mock_atlas

    registration_widget.adjust_moving_image_widget.adjust_moving_image_pixel_size_x.setValue(
        x_res
    )
    registration_widget.adjust_moving_image_widget.adjust_moving_image_pixel_size_y.setValue(
        y_res
    )

    registration_widget.adjust_moving_image_widget._on_scale_image_button_click()

    mocked_show_error.assert_called_once_with(
        "Pixel sizes must be greater than 0"
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
def test_scale_moving_image_2d(
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
        mock_atlas.resolution[2] * x_scale_factor,
        mock_atlas.resolution[1] * y_scale_factor,
        0.001,
        # Empty orientation string
        "",
    )

    assert registration_widget._moving_image.data.shape == (
        current_size[0] * y_scale_factor,
        current_size[1] * x_scale_factor,
    )


@pytest.mark.parametrize(
    "x_scale_factor, y_scale_factor, z_scale_factor",
    [
        (0.5, 0.5, 0.5),
        (1.0, 1.0, 1.0),
        (2.0, 2.0, 2.0),
        (0.5, 1.0, 1.0),
        (1.0, 0.5, 1.0),
        (1.0, 1.0, 0.5),
    ],
)
def test_scale_moving_image_3d(
    make_napari_viewer_with_images,
    registration_widget,
    mocker,
    x_scale_factor,
    y_scale_factor,
    z_scale_factor,
):
    mock_atlas = mocker.patch(
        "brainglobe_registration.registration_widget.BrainGlobeAtlas"
    )
    mock_atlas.resolution = [100, 100, 100]
    mock_atlas.space = AnatomicalSpace(
        origin="asr", resolution=mock_atlas.resolution
    )
    registration_widget._atlas = mock_atlas

    moving_image_3d_index = registration_widget._sample_images.index(
        "moving_image_3d"
    )
    registration_widget._on_sample_dropdown_index_changed(
        moving_image_3d_index
    )

    current_size = registration_widget._moving_image.data.shape
    registration_widget.adjust_moving_image_widget.scale_image_signal.emit(
        mock_atlas.resolution[2] * x_scale_factor,
        mock_atlas.resolution[1] * y_scale_factor,
        mock_atlas.resolution[0] * z_scale_factor,
        "asr",
    )

    expected_size = np.round(
        (
            current_size[0] * z_scale_factor,
            current_size[1] * y_scale_factor,
            current_size[2] * x_scale_factor,
        )
    )

    assert np.all(
        registration_widget._moving_image.data.shape == expected_size
    )


def test_invalid_sample_orientation(
    make_napari_viewer_with_images, registration_widget, mocker
):
    mocked_show_error = mocker.patch(
        "brainglobe_registration.registration_widget.show_error"
    )
    mock_atlas = mocker.patch(
        "brainglobe_registration.registration_widget.BrainGlobeAtlas"
    )
    mock_atlas.resolution = [20, 20, 20]
    registration_widget._atlas = mock_atlas

    moving_image_3d_index = registration_widget._sample_images.index(
        "moving_image_3d"
    )
    registration_widget._on_sample_dropdown_index_changed(
        moving_image_3d_index
    )

    registration_widget.adjust_moving_image_widget.adjust_moving_image_pixel_size_x.setValue(
        mock_atlas.resolution[2] * 0.5
    )
    registration_widget.adjust_moving_image_widget.adjust_moving_image_pixel_size_y.setValue(
        mock_atlas.resolution[1] * 0.5
    )
    registration_widget.adjust_moving_image_widget.adjust_moving_image_pixel_size_z.setValue(
        mock_atlas.resolution[0] * 0.5
    )
    registration_widget.adjust_moving_image_widget.data_orientation_field.setText(
        "abc"
    )

    registration_widget.adjust_moving_image_widget._on_scale_image_button_click()

    mocked_show_error.assert_called_once_with(
        "Invalid orientation. "
        "Please use the BrainGlobe convention (e.g. 'psl')"
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
    assert (
        reg_widget._atlas_data_layer.data.dtype
        == reg_widget._atlas.reference.dtype
    )
    assert (
        reg_widget._atlas_annotations_layer.data.dtype
        == reg_widget._atlas.annotation.dtype
    )


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
    assert (
        reg_widget._atlas_data_layer.data.dtype
        == reg_widget._atlas.reference.dtype
    )
    assert (
        reg_widget._atlas_annotations_layer.data.dtype
        == reg_widget._atlas.annotation.dtype
    )


def test_on_atlas_reset_no_atlas(registration_widget, mocker):
    mocked_show_error = mocker.patch(
        "brainglobe_registration.registration_widget.show_error"
    )

    registration_widget._on_atlas_reset()
    mocked_show_error.assert_called_once_with(
        "No atlas selected. Please select an atlas before resetting"
    )


def test_on_output_directory_text_edited(registration_widget):
    registration_widget.output_directory_text_field.setText(str(Path.home()))

    registration_widget._on_output_directory_text_edited()

    assert registration_widget.output_directory == Path.home()


def test_on_open_file_dialog_clicked(registration_widget, mocker):
    mocked_open_dialog = mocker.patch(
        "brainglobe_registration.registration_widget.QFileDialog.getExistingDirectory"
    )
    mocked_open_dialog.return_value = str(Path.home())

    registration_widget.open_file_dialog.click()

    assert registration_widget.output_directory == Path.home()
    mocked_open_dialog.assert_called_once()


def test_on_open_file_dialog_cancelled(registration_widget, mocker):
    expected_dir = Path.home() / "mock_directory"
    registration_widget.output_directory = expected_dir
    mocked_open_dialog = mocker.patch(
        "brainglobe_registration.registration_widget.QFileDialog.getExistingDirectory"
    )
    mocked_open_dialog.return_value = ""

    registration_widget.open_file_dialog.click()

    assert registration_widget.output_directory == expected_dir
    mocked_open_dialog.assert_called_once()


def test_on_run_button_clicked_no_atlas(registration_widget, mocker):
    mocked_display_info = mocker.patch(
        "brainglobe_registration.registration_widget.display_info"
    )
    registration_widget.run_button.setEnabled(True)
    registration_widget._atlas = None
    registration_widget._atlas_data_layer = None
    registration_widget.run_button.click()
    mocked_display_info.assert_called_once_with(
        widget=registration_widget,
        title="Warning",
        message="Please select an atlas before clicking 'Run'.",
    )


def test_on_run_button_clicked_no_sample_image(
    registration_widget_with_example_atlas, mocker
):
    mocked_display_info = mocker.patch(
        "brainglobe_registration.registration_widget.display_info"
    )
    registration_widget_with_example_atlas.run_button.setEnabled(True)
    registration_widget_with_example_atlas._moving_image = None
    registration_widget_with_example_atlas.run_button.click()
    mocked_display_info.assert_called_once_with(
        widget=registration_widget_with_example_atlas,
        title="Warning",
        message="Please select a moving image before clicking 'Run'.",
    )


def test_on_run_button_clicked_no_output_directory(
    registration_widget_with_example_atlas, mocker
):
    mocked_display_info = mocker.patch(
        "brainglobe_registration.registration_widget.display_info"
    )
    registration_widget_with_example_atlas.run_button.setEnabled(True)
    registration_widget_with_example_atlas._moving_image = True
    registration_widget_with_example_atlas.output_directory = None
    registration_widget_with_example_atlas.run_button.click()
    mocked_display_info.assert_called_once_with(
        widget=registration_widget_with_example_atlas,
        title="Warning",
        message="Please select an output directory before clicking 'Run'.",
    )


def test_on_run_button_clicked_moving_equal_atlas(
    registration_widget_with_example_atlas, mocker
):
    mocked_display_info = mocker.patch(
        "brainglobe_registration.registration_widget.display_info"
    )
    registration_widget_with_example_atlas.run_button.setEnabled(True)
    registration_widget_with_example_atlas._moving_image = (
        registration_widget_with_example_atlas._atlas_data_layer
    )
    registration_widget_with_example_atlas.run_button.click()
    mocked_display_info.assert_called_once_with(
        widget=registration_widget_with_example_atlas,
        title="Warning",
        message="Your moving image cannot be an atlas.",
    )


def test_on_run_button_click_2d(registration_widget, tmp_path):
    allen_25_index = registration_widget._available_atlases.index(
        "allen_mouse_25um"
    )
    registration_widget._on_atlas_dropdown_index_changed(allen_25_index)

    registration_widget._viewer.dims.set_current_step(0, 293)
    moving_image = imread(
        Path(__file__).parent / "test_images/sample_hipp.tif"
    ).astype(np.float32)
    moving_image_name = "sample_hipp"
    registration_widget._moving_image = registration_widget._viewer.add_image(
        moving_image, name=moving_image_name
    )
    registration_widget.output_directory = tmp_path

    registration_widget.run_button.click()

    assert (tmp_path / "TransformParameters.0.txt").exists()
    assert (tmp_path / "TransformParameters.1.txt").exists()
    assert (tmp_path / "InverseTransformParameters.0.txt").exists()
    assert (tmp_path / "InverseTransformParameters.1.txt").exists()
    assert (
        tmp_path / f"downsampled_standard_{moving_image_name}.tiff"
    ).exists()
    assert (tmp_path / "registered_atlas.tiff").exists()
    assert (tmp_path / "registered_hemisphere.tiff").exists()
    assert (tmp_path / "areas.csv").exists()
    assert (tmp_path / "boundaries.tiff").exists()
    assert (tmp_path / "deformation_field_0.tiff").exists()
    assert (tmp_path / "deformation_field_1.tiff").exists()
    assert (tmp_path / "downsampled.tiff").exists()
    assert (tmp_path / "brainglobe-registration.json").exists()


def test_open_auto_slice_dialog_no_atlas(registration_widget, mocker):
    mocked_display_info = mocker.patch(
        "brainglobe_registration.registration_widget.display_info"
    )
    registration_widget._atlas = None
    registration_widget._atlas_data_layer = None
    registration_widget._moving_image = True

    registration_widget._open_auto_slice_dialog()

    mocked_display_info.assert_called_once_with(
        widget=registration_widget,
        title="Warning",
        message="Please select an atlas before "
        "clicking 'Automatic Slice Detection'.",
    )


def test_open_auto_slice_dialog_no_moving_image(
    registration_widget_with_example_atlas, mocker
):
    mocked_display_info = mocker.patch(
        "brainglobe_registration.registration_widget.display_info"
    )
    registration_widget_with_example_atlas._moving_image = None

    registration_widget_with_example_atlas._open_auto_slice_dialog()

    mocked_display_info.assert_called_once_with(
        widget=registration_widget_with_example_atlas,
        title="Warning",
        message="Please select a moving image before "
        "clicking 'Automatic Slice Detection'.",
    )


def test_open_auto_slice_dialog_moving_equals_atlas(
    registration_widget_with_example_atlas, mocker
):
    mocked_display_info = mocker.patch(
        "brainglobe_registration.registration_widget.display_info"
    )
    registration_widget_with_example_atlas._moving_image = (
        registration_widget_with_example_atlas._atlas_data_layer
    )

    registration_widget_with_example_atlas._open_auto_slice_dialog()

    mocked_display_info.assert_called_once_with(
        widget=registration_widget_with_example_atlas,
        title="Warning",
        message="Your moving image cannot be an atlas.",
    )


def test_open_auto_slice_dialog_valid(
    registration_widget_with_example_atlas, mocker
):
    mocked_dialog_class = mocker.patch(
        "brainglobe_registration.registration_widget.AutoSliceDialog"
    )
    mocked_dialog = mocked_dialog_class.return_value
    # Register that dialog box was called
    mocked_dialog.exec_ = mocker.Mock()

    registration_widget_with_example_atlas._moving_image = mocker.Mock()
    (
        registration_widget_with_example_atlas._moving_image
        != registration_widget_with_example_atlas._atlas_data_layer
    )

    registration_widget_with_example_atlas._open_auto_slice_dialog()

    mocked_dialog_class.assert_called_once()
    mocked_dialog.exec_.assert_called_once()


def test__on_auto_slice_parameters_confirmed_starts_worker(
    registration_widget, mocker
):
    params = {
        "init_points": 3,
        "n_iter": 5,
    }
    total = 2 * (params["init_points"] + params["n_iter"])

    # Mock progress bar
    progress_bar_mock = mocker.Mock()
    registration_widget.adjust_moving_image_widget.progress_bar = (
        progress_bar_mock
    )

    mocked_worker = mocker.Mock()
    mocked_create_worker = mocker.patch(
        "brainglobe_registration.registration_widget.create_worker",
        return_value=mocked_worker,
    )

    # Call method
    registration_widget._on_auto_slice_parameters_confirmed(params)

    # Assert progress bar is set correctly
    progress_bar_mock.setVisible.assert_called_once_with(True)
    progress_bar_mock.setValue.assert_called_once_with(0)
    progress_bar_mock.setRange.assert_called_once_with(0, total)

    # Assert worker created and connected correctly
    mocked_create_worker.assert_called_once()
    mocked_worker.yielded.connect.assert_called_once_with(
        registration_widget.handle_auto_slice_progress
    )
    mocked_worker.returned.connect.assert_called_once_with(
        registration_widget.set_optimal_rotation_params
    )
    mocked_worker.start.assert_called_once()


def test_strip_ansi_color_filter_removes_escape_codes():
    record = logging.LogRecord(
        "name",
        logging.INFO,
        "pathname",
        0,
        "\x1b[31mRed Text\x1b[0m",
        (),
        None,
    )
    filt = StripANSIColorFilter()
    filt.filter(record)

    assert record.msg == "Red Text"


def test_run_auto_slice_thread_logs_and_yields_results(
    mocker, registration_widget
):
    atlas_image = np.ones((10, 10), dtype=np.uint8)
    moving_image = (np.ones((10, 10)) * 2).astype(np.int16)

    mocker.patch(
        "brainglobe_registration.registration_widget."
        "get_data_from_napari_layer",
        side_effect=[atlas_image, moving_image],
    )

    logging_dir = Path("/tmp/fake_logging_dir")
    mocker.patch(
        "brainglobe_registration.registration_widget.get_brainglobe_dir",
        return_value=logging_dir,
    )

    mock_args_namedtuple = mocker.Mock()
    mock_get_args = mocker.patch(
        "brainglobe_registration.registration_widget."
        "get_auto_slice_logging_args",
        return_value=mock_args_namedtuple,
    )

    mock_start_logging = mocker.patch(
        "brainglobe_registration.registration_widget.fancylog.start_logging"
    )

    # Patch logger handlers
    handler_1 = mocker.Mock()
    handler_2 = mocker.Mock()
    handler_1.level = logging.NOTSET
    handler_2.level = logging.NOTSET
    logger = logging.getLogger()
    original_handlers = logger.handlers
    logger.handlers = [handler_1, handler_2]

    # Spy on logging.info
    info_spy = mocker.spy(logging, "info")

    final_result = {
        "best_pitch": 1,
        "best_yaw": 2,
        "best_roll": 3,
        "best_z_slice": 4,
    }

    def mock_generator(*args, **kwargs):
        yield  # 1st progress
        yield  # 2nd progress
        return final_result

    mocker.patch(
        "brainglobe_registration.registration_widget.run_bayesian_generator",
        side_effect=mock_generator,
    )

    # Define params
    params = {
        "z_range": (0, 10),
        "pitch_bounds": (-10, 10),
        "yaw_bounds": (-10, 10),
        "roll_bounds": (-10, 10),
        "init_points": 1,
        "n_iter": 1,
        "metric": "mi",
        "weights": [1.0, 0.0, 0.0],
    }

    # Call the function and collect outputs
    gen = registration_widget.run_auto_slice_thread(params)

    progress = []
    try:
        while True:
            progress.append(next(gen))
    except StopIteration as stop:
        final_result = stop.value

    expected_output_dir = str(logging_dir / "brainglobe_registration_logs")

    mock_start_logging.assert_called_once_with(
        output_dir=expected_output_dir,
        package=mocker.ANY,
        filename="auto_slice_log",
        variables=mock_args_namedtuple,
        log_header="AUTO SLICE DETECTION LOG",
        verbose=True,
        write_git=False,
    )

    mock_get_args.assert_called_once_with(params)

    # Check StripANSIColorFilter added
    handler_1.addFilter.assert_called_once()
    handler_2.addFilter.assert_called_once()
    assert isinstance(
        handler_1.addFilter.call_args.args[0], StripANSIColorFilter
    )

    # Check logging.info was called
    info_spy.assert_any_call("Starting Bayesian slice detection...")

    # Check progress yields
    assert progress == [{"progress": 1}, {"progress": 2}]

    assert final_result == {
        "done": True,
        "best_pitch": 1,
        "best_yaw": 2,
        "best_roll": 3,
        "best_z_slice": 4,
    }

    # Restore logger handlers
    logger.handlers = original_handlers


def test_set_optimal_rotation_params_sets_gui_values(
    registration_widget, mocker
):
    result = {
        "done": True,
        "best_pitch": 5,
        "best_yaw": 10,
        "best_roll": -2,
        "best_z_slice": 42,
    }

    # Create mock layers
    mock_layer = mocker.Mock()
    mock_layer.name = "mock_layer"

    # Create viewer mock with iterable .layers
    viewer_mock = mocker.Mock()
    viewer_mock.layers = [mock_layer]
    viewer_mock.dims.set_point = mocker.Mock()

    # Mock adjust widget
    adjust_widget_mock = mocker.Mock()
    adjust_widget_mock.adjust_atlas_pitch.setValue = mocker.Mock()
    adjust_widget_mock.adjust_atlas_yaw.setValue = mocker.Mock()
    adjust_widget_mock.adjust_atlas_roll.setValue = mocker.Mock()
    adjust_widget_mock.progress_bar.reset = mocker.Mock()
    adjust_widget_mock.progress_bar.setVisible = mocker.Mock()

    # Assign mocks to widget
    registration_widget._viewer = viewer_mock
    registration_widget.adjust_moving_image_widget = adjust_widget_mock
    registration_widget._on_adjust_atlas_rotation = mocker.Mock()

    # Call method
    registration_widget.set_optimal_rotation_params(result)

    # Assertions
    registration_widget._on_adjust_atlas_rotation.assert_called_once_with(
        5, 10, -2
    )
    viewer_mock.dims.set_point.assert_called_once_with(0, 42)
    adjust_widget_mock.adjust_atlas_pitch.setValue.assert_called_once_with(5)
    adjust_widget_mock.adjust_atlas_yaw.setValue.assert_called_once_with(10)
    adjust_widget_mock.adjust_atlas_roll.setValue.assert_called_once_with(-2)
    adjust_widget_mock.progress_bar.reset.assert_called_once()
    adjust_widget_mock.progress_bar.setVisible.assert_called_once_with(False)
