from pathlib import Path
from unittest.mock import MagicMock, Mock

import dask.array as da
import numpy as np
import pytest
from brainglobe_atlasapi import BrainGlobeAtlas
from pytransform3d.rotations import active_matrix_from_angle

from brainglobe_registration.utils.atlas import (
    calculate_region_size,
    convert_atlas_labels,
    generate_mask_from_atlas_annotations,
    mask_atlas,
    mask_atlas_with_annotations,
    restore_atlas_labels,
)
from brainglobe_registration.utils.file import (
    open_parameter_file,
    serialize_registration_widget,
)
from brainglobe_registration.utils.napari import (
    adjust_napari_image_layer,
    check_atlas_installed,
    find_layer_index,
    get_data_from_napari_layer,
    get_image_layer_names,
)
from brainglobe_registration.utils.transforms import (
    calculate_rotated_bounding_box,
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


def test_open_parameter_file_with_valid_content():
    file_path = Path("test_file.txt")
    with open(file_path, "w") as f:
        f.write("(key1 value1 value2)\n(key2 value3 value4)")
    result = open_parameter_file(file_path)
    assert result == {
        "key1": ["value1", "value2"],
        "key2": ["value3", "value4"],
    }
    file_path.unlink()


def test_open_parameter_file_with_invalid_content():
    file_path = Path("test_file.txt")
    with open(file_path, "w") as f:
        f.write("invalid content")
    result = open_parameter_file(file_path)
    assert result == {}
    file_path.unlink()


def test_open_parameter_file_with_empty_content():
    file_path = Path("test_file.txt")
    with open(file_path, "w") as f:
        f.write("")
    result = open_parameter_file(file_path)
    assert result == {}
    file_path.unlink()


def test_open_parameter_file_with_comment_and_paren():
    """Test open_parameter_file handles entries with ) and / comments."""
    file_path = Path("test_file.txt")
    with open(file_path, "w") as f:
        f.write(
            "(key1 value1 value2 )\n(key2 value3 / comment)\n(key3 value4)"
        )
    result = open_parameter_file(file_path)
    assert result == {
        "key1": ["value1", "value2"],
        "key2": ["value3"],
        "key3": ["value4"],
    }
    file_path.unlink()


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
    "basis, rotation, expected_bounds",
    [
        (0, 0, (50, 100, 200)),
        (0, 90, (50, 200, 100)),
        (0, 180, (50, 100, 200)),
        (0, 45, (50, 212, 212)),
        (1, 0, (50, 100, 200)),
        (1, 90, (200, 100, 50)),
        (1, 180, (50, 100, 200)),
        (1, 45, (177, 100, 177)),
        (2, 0, (50, 100, 200)),
        (2, 90, (100, 50, 200)),
        (2, 180, (50, 100, 200)),
        (2, 45, (106, 106, 200)),
    ],
)
def test_calculate_rotated_bounding_box(basis, rotation, expected_bounds):
    image_shape = (50, 100, 200)
    rotation_matrix = np.eye(4)
    rotation_matrix[:-1, :-1] = active_matrix_from_angle(
        basis, np.deg2rad(rotation)
    )

    result_shape = calculate_rotated_bounding_box(image_shape, rotation_matrix)

    assert result_shape == expected_bounds


def test_convert_atlas_labels_no_change():
    mock_annotations = np.arange(1024).reshape((32, 32))

    result, mapping = convert_atlas_labels(mock_annotations)

    assert np.array_equal(result, mock_annotations)
    assert len(mapping) == 0


def test_convert_atlas_labels_high_labels():
    mock_annotations = np.arange(2**15, 2**15 + 1024).reshape((32, 32))

    result, mapping = convert_atlas_labels(mock_annotations)

    # Since the labels are consecutive, starting at the max label, there
    # should be no change to the array and the mapping should be empty
    assert np.array_equal(result, mock_annotations)
    assert len(mapping) == 0


def test_convert_atlas_labels():
    rng = np.random.default_rng(42)

    mock_annotations = rng.integers(
        2**32 - 1, size=(128, 128), dtype=np.uint32
    )

    result, mapping = convert_atlas_labels(mock_annotations)

    max_value = 2**15
    unique_values = np.unique(mock_annotations)
    expected_mapping_count = unique_values[unique_values >= max_value].size

    assert len(mapping) == expected_mapping_count

    restored_image = restore_atlas_labels(result, mapping)

    assert np.array_equal(restored_image, mock_annotations)


def test_calculate_areas(tmp_path):
    atlas = BrainGlobeAtlas("allen_mouse_100um")

    mid_point = atlas.annotation.shape[0] // 2
    mock_annotations = atlas.annotation[mid_point, :, :]
    hemispheres = atlas.hemispheres[mid_point, :, :]

    output_path = tmp_path / "areas.csv"

    out_df = calculate_region_size(
        atlas, mock_annotations, hemispheres, output_path
    )

    assert output_path.exists()
    assert out_df.columns.size == 4

    # Based on regression testing, the following values are expected
    assert out_df.loc[672, "structure_name"] == "Caudoputamen"
    assert out_df.loc[672, "left_area_mm2"] == 1.98
    assert out_df.loc[672, "right_area_mm2"] == 2.0
    assert out_df.loc[672, "total_area_mm2"] == 3.98

    assert out_df.loc[961, "structure_name"] == "Piriform area"
    assert out_df.loc[961, "left_area_mm2"] == 1.27
    assert out_df.loc[961, "right_area_mm2"] == 1.28
    assert out_df.loc[961, "total_area_mm2"] == 2.55


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


@pytest.fixture
def dummy_atlas(mocker):
    # Patch the BrainGlobeAtlas class
    mock_atlas_class = mocker.patch(
        "brainglobe_registration.utils.atlas.BrainGlobeAtlas"
    )

    # Create a mock instance
    mock_instance = mock_atlas_class.return_value

    # Assign test data
    mock_instance.reference = np.array(
        [[100, 150, 200], [50, 0, 75], [25, 25, 25]], dtype=np.uint16
    )

    mock_instance.annotation = np.array(
        [[1, 0, 2], [0, 0, 3], [4, 0, 0]], dtype=np.uint16
    )

    return mock_instance


def test_generate_mask_from_atlas_annotations(dummy_atlas):
    mask = generate_mask_from_atlas_annotations(dummy_atlas)
    expected_mask = np.array([[1, 0, 1], [0, 0, 1], [1, 0, 0]], dtype=np.uint8)

    assert np.array_equal(mask, expected_mask)


def test_mask_atlas(dummy_atlas):
    # Create a mask to apply manually
    mask = np.array([[1, 0, 1], [0, 0, 1], [1, 0, 0]], dtype=np.uint8)

    masked_image = mask_atlas(dummy_atlas, mask)
    expected = np.array(
        [[100, 0, 200], [0, 0, 75], [25, 0, 0]], dtype=np.uint16
    )

    assert np.array_equal(masked_image, expected)


def test_mask_atlas_with_annotations(dummy_atlas):
    masked_image = mask_atlas_with_annotations(dummy_atlas)
    expected = np.array(
        [[100, 0, 200], [0, 0, 75], [25, 0, 0]], dtype=np.uint16
    )

    assert np.array_equal(masked_image, expected)


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


def test_serialize_registration_widget(mocker):
    """Test serialize_registration_widget for different object types."""
    from pathlib import PurePath

    import napari

    # Test with napari layer
    mock_layer = mocker.MagicMock(spec=napari.layers.Layer)
    mock_layer.name = "test_layer"
    assert serialize_registration_widget(mock_layer) == "test_layer"

    # Test with napari viewer
    mock_viewer = mocker.MagicMock(spec=napari.Viewer)
    mock_viewer.__str__ = mocker.MagicMock(return_value="test_viewer")
    assert serialize_registration_widget(mock_viewer) == "test_viewer"

    # Test with PurePath
    test_path = PurePath("test/path")
    assert serialize_registration_widget(test_path) == str(test_path)

    # Test with BrainGlobeAtlas
    mock_atlas = mocker.MagicMock(spec=BrainGlobeAtlas)
    mock_atlas.atlas_name = "test_atlas"
    assert serialize_registration_widget(mock_atlas) == "test_atlas"

    # Test with numpy array
    test_array = np.array([[1, 2], [3, 4]])
    result = serialize_registration_widget(test_array)
    assert "<class 'numpy.ndarray'>" in result
    assert "(2, 2)" in result
    assert test_array.dtype.name in result or str(test_array.dtype) in result

    # Test with other object that has __dict__() method
    # (like RegistrationWidget)
    class TestObj:
        def __dict__(self):
            return {"key": "value"}

    test_obj = TestObj()
    assert serialize_registration_widget(test_obj) == {"key": "value"}
