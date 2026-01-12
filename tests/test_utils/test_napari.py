from unittest.mock import MagicMock, Mock

import dask.array as da
import numpy as np
import pytest
from pytransform3d.rotations import active_matrix_from_angle

from brainglobe_registration.utils.napari import (
    adjust_napari_image_layer,
    check_atlas_installed,
    find_layer_index,
    get_data_from_napari_layer,
    get_image_layer_names,
)


def test_adjust_napari_image_layer_no_translation_no_rotation():
    image_layer = Mock()
    image_layer.data.shape = (100, 100)
    adjust_napari_image_layer(image_layer, 0, 0, 0)
    assert image_layer.translate == (0, 0)
    assert np.array_equal(image_layer.affine, np.eye(3))


def test_adjust_napari_image_layer_with_translation_no_rotation():
    image_layer = Mock()
    image_layer.data.shape = (100, 100)
    adjust_napari_image_layer(image_layer, 10, 20, 0)
    assert image_layer.translate == (20, 10)
    assert np.array_equal(image_layer.affine, np.eye(3))


def test_adjust_napari_image_layer_no_translation_with_rotation():
    image_layer = Mock()
    image_layer.data.shape = (100, 100)
    adjust_napari_image_layer(image_layer, 0, 0, 45)
    rotation_matrix = active_matrix_from_angle(2, np.deg2rad(45))
    translate_matrix = np.eye(3)
    origin = np.asarray(image_layer.data.shape) // 2
    translate_matrix[:2, -1] = origin
    expected_transform_matrix = (
        translate_matrix @ rotation_matrix @ np.linalg.inv(translate_matrix)
    )
    assert np.array_equal(image_layer.affine, expected_transform_matrix)


def test_adjust_napari_image_layer_with_translation_and_rotation():
    """Test adjust_napari_image_layer with both translation and rotation."""
    image_layer = Mock()
    image_layer.data.shape = (100, 100)
    adjust_napari_image_layer(image_layer, 10, 20, 45)
    assert image_layer.translate == (20, 10)
    # Verify that the affine transform was set (not identity)
    assert not np.array_equal(image_layer.affine, np.eye(3))


@pytest.mark.parametrize(
    "name, index",
    [
        ("moving_image_2d", 0),
        ("moving_image_3d", 1),
    ],
)
def test_find_layer_index(make_napari_viewer_with_images, name, index):
    viewer = make_napari_viewer_with_images

    assert find_layer_index(viewer, name) == index


def test_find_layer_index_not_found(make_napari_viewer_with_images):
    """Test find_layer_index returns -1 when layer is not found."""
    viewer = make_napari_viewer_with_images
    assert find_layer_index(viewer, "non_existent_layer") == -1


def test_get_image_layer_names(make_napari_viewer_with_images):
    viewer = make_napari_viewer_with_images

    layer_names = get_image_layer_names(viewer)

    assert len(layer_names) == 2
    assert layer_names == ["moving_image_2d", "moving_image_3d"]


@pytest.mark.parametrize(
    "layer_data, selection",
    [
        (np.arange(1000).reshape((10, 10, 10)), None),
        (np.arange(1000).reshape((10, 10, 10)), (slice(0, 5),)),
        (da.arange(1000).reshape((10, 10, 10)), None),
        (da.arange(1000).reshape((10, 10, 10)), (slice(0, 5),)),
    ],
)
def test_get_data_from_napari_layer(layer_data, selection):
    layer = Mock()
    layer.data = layer_data

    result = get_data_from_napari_layer(layer, selection)

    assert isinstance(result, np.ndarray)

    if selection is None:
        assert np.array_equal(result, layer_data)
    else:
        assert np.array_equal(result, layer_data[selection])


@pytest.mark.parametrize(
    "layer_data, selection",
    [
        (np.arange(1000).reshape((10, 10, 10)), (slice(0, 1),)),
        (da.arange(1000).reshape((10, 10, 10)), (slice(0, 1),)),
    ],
)
def test_get_data_from_napari_layer_squeeze(layer_data, selection):
    layer = Mock()
    layer.data = layer_data

    result = get_data_from_napari_layer(layer, selection)

    assert isinstance(result, np.ndarray)
    assert result.ndim == layer_data.ndim - 1
    assert np.array_equal(result, layer_data[selection].squeeze())


def test_check_atlas_installed_no_atlases(mocker):
    """Test check_atlas_installed when no atlases are available."""
    # Mock get_downloaded_atlases to return empty list
    mock_get_atlases = mocker.patch(
        "brainglobe_registration.utils.napari.get_downloaded_atlases",
        return_value=[],
    )
    # Mock display_info
    mock_display_info = mocker.patch(
        "brainglobe_registration.utils.napari.display_info"
    )

    parent_widget = MagicMock()
    check_atlas_installed(parent_widget)

    mock_get_atlases.assert_called_once()
    mock_display_info.assert_called_once()
    call_args = mock_display_info.call_args
    assert call_args[1]["widget"] == parent_widget
    assert call_args[1]["title"] == "Information"


def test_check_atlas_installed_with_atlases(mocker):
    """Test check_atlas_installed when atlases are available."""
    # Mock get_downloaded_atlases to return non-empty list
    mock_get_atlases = mocker.patch(
        "brainglobe_registration.utils.napari.get_downloaded_atlases",
        return_value=["atlas1", "atlas2"],
    )
    # Mock display_info
    mock_display_info = mocker.patch(
        "brainglobe_registration.utils.napari.display_info"
    )

    parent_widget = MagicMock()
    check_atlas_installed(parent_widget)

    mock_get_atlases.assert_called_once()
    # display_info should not be called when atlases exist
    mock_display_info.assert_not_called()
