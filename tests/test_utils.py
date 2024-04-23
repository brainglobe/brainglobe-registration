from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from pytransform3d.rotations import active_matrix_from_angle

from brainglobe_registration.utils.utils import (
    adjust_napari_image_layer,
    find_layer_index,
    get_image_layer_names,
    open_parameter_file,
    calculate_rotated_bounding_box,
)


def adjust_napari_image_layer_no_translation_no_rotation():
    image_layer = Mock()
    image_layer.data.shape = (100, 100)
    adjust_napari_image_layer(image_layer, 0, 0, 0)
    assert image_layer.translate == (0, 0)
    assert np.array_equal(image_layer.affine, np.eye(3))


def adjust_napari_image_layer_with_translation_no_rotation():
    image_layer = Mock()
    image_layer.data.shape = (100, 100)
    adjust_napari_image_layer(image_layer, 10, 20, 0)
    assert image_layer.translate == (20, 10)
    assert np.array_equal(image_layer.affine, np.eye(3))


def adjust_napari_image_layer_no_translation_with_rotation():
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


def open_parameter_file_with_valid_content():
    file_path = Path("test_file.txt")
    with open(file_path, "w") as f:
        f.write("(key1 value1 value2)\n(key2 value3 value4)")
    result = open_parameter_file(file_path)
    assert result == {
        "key1": ["value1", "value2"],
        "key2": ["value3", "value4"],
    }
    file_path.unlink()


def open_parameter_file_with_invalid_content():
    file_path = Path("test_file.txt")
    with open(file_path, "w") as f:
        f.write("invalid content")
    result = open_parameter_file(file_path)
    assert result == {}
    file_path.unlink()


def open_parameter_file_with_empty_content():
    file_path = Path("test_file.txt")
    with open(file_path, "w") as f:
        f.write("")
    result = open_parameter_file(file_path)
    assert result == {}
    file_path.unlink()


@pytest.mark.parametrize(
    "name, index",
    [
        ("moving_image", 0),
        ("atlas_image", 1),
    ],
)
def test_find_layer_index(make_napari_viewer_with_images, name, index):
    viewer = make_napari_viewer_with_images

    assert find_layer_index(viewer, name) == index


def test_get_image_layer_names(make_napari_viewer_with_images):
    viewer = make_napari_viewer_with_images

    layer_names = get_image_layer_names(viewer)

    assert len(layer_names) == 2
    assert layer_names == ["moving_image", "atlas_image"]


@pytest.mark.parametrize(
    "basis, rotation, expected_bounds",
    [
        (0, 0, (50, 100, 200)),
        (0, 90, (50, 200, 100)),
        (0, 180,  (50, 100, 200)),
        (0, 45, (50, 212, 212)),
        (1, 0, (50, 100, 200)),
        (1, 90, (200, 100, 50)),
        (1, 180, (50, 100, 200)),
        (1, 45, (177, 100, 177)),
        (2, 0, (50, 100, 200)),
        (2, 90, (100, 50, 200)),
        (2, 180, (50, 100, 200)),
        (2, 45, (106, 106, 200)),
    ]
)
def test_calculate_rotated_bounding_box(basis, rotation, expected_bounds):
    image_shape = (50, 100, 200)
    rotation_matrix = np.eye(4)
    rotation_matrix[:-1, :-1] = active_matrix_from_angle(basis, np.deg2rad(rotation))

    result_shape = calculate_rotated_bounding_box(image_shape, rotation_matrix)

    assert result_shape == expected_bounds
