import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from brainglobe_registration.utils.transforms import (
    create_rotation_matrix,
    rotate_volume,
    scale_moving_image,
)


@pytest.fixture
def dummy_volume():
    """Create a dummy volume with arbitrary shape and values."""
    return np.random.rand(20, 40, 60).astype(np.float32)


def test_create_rotation_matrix_identity(dummy_volume):
    """
    Test that identity rotation gives a transform close to identity matrix.
    """
    shape = dummy_volume.shape
    transform, bbox = create_rotation_matrix(0, 0, 0, shape)

    expected = np.eye(4)
    assert_array_almost_equal(transform, expected, decimal=3)
    assert bbox == shape


@pytest.mark.parametrize(
    "angles, expected_rot",
    [
        # Rotation around X-axis (clockwise)
        (
            (30, 0, 0),
            np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(np.deg2rad(30)), np.sin(np.deg2rad(30))],
                    [0, -np.sin(np.deg2rad(30)), np.cos(np.deg2rad(30))],
                ]
            ),
        ),
        # Rotation around Y-axis (clockwise)
        (
            (0, 30, 0),
            np.array(
                [
                    [np.cos(np.deg2rad(30)), 0, -np.sin(np.deg2rad(30))],
                    [0, 1, 0],
                    [np.sin(np.deg2rad(30)), 0, np.cos(np.deg2rad(30))],
                ]
            ),
        ),
        # Rotation around Z-axis (clockwise)
        (
            (0, 0, 30),
            np.array(
                [
                    [np.cos(np.deg2rad(30)), np.sin(np.deg2rad(30)), 0],
                    [-np.sin(np.deg2rad(30)), np.cos(np.deg2rad(30)), 0],
                    [0, 0, 1],
                ]
            ),
        ),
    ],
)
def test_create_rotation_matrix_single_axis(
    angles, expected_rot, dummy_volume
):
    """
    Check that create_rotation_matrix returns expected affine rotation
    matrix (ignoring translation) for 30° rotation about one axis.
    """
    shape = dummy_volume.shape
    transform, _ = create_rotation_matrix(*angles, shape)

    # Extract rotation part
    rot_actual = transform[:3, :3]

    np.testing.assert_allclose(rot_actual, expected_rot, rtol=1e-5, atol=1e-5)


def test_create_rotation_matrix_nonzero(dummy_volume):
    """Test non-zero rotations yield valid affine matrix and bounding box."""
    shape = dummy_volume.shape
    transform, bbox = create_rotation_matrix(15, -10, 30, shape)

    assert transform.shape == (4, 4), "Expected 4x4 affine matrix"
    assert all(
        isinstance(b, int) and b > 0 for b in bbox
    ), "Bounding box must be positive ints"
    assert not np.allclose(
        transform, np.eye(4)
    ), "Rotation matrix should differ from identity"


def test_rotate_volume_output_shape_and_dtype(dummy_volume):
    """
    Check that rotate_volume returns correct shape, dtype,
    and values under identity transform.
    """
    shape = dummy_volume.shape
    transform, bbox = create_rotation_matrix(0, 0, 0, shape)

    rotated = rotate_volume(dummy_volume, shape, transform, bbox).compute()
    assert rotated.shape == bbox, "Output shape mismatch with bounding box"
    assert rotated.dtype == dummy_volume.dtype, "Dtype mismatch after rotation"
    # Accept approximate similarity due to interpolation
    correlation = np.corrcoef(dummy_volume.flatten(), rotated.flatten())[0, 1]
    assert (
        correlation > 0.90
    ), "Rotated volume differs significantly under identity transform"


def test_rotate_volume_nontrivial_transform(dummy_volume):
    """
    Ensure volume shape changes with non-zero rotation.
    """
    shape = dummy_volume.shape
    transform, bbox = create_rotation_matrix(20, 10, 5, shape)
    rotated = rotate_volume(dummy_volume, shape, transform, bbox).compute()

    assert (
        rotated.shape == bbox
    ), "Rotated volume shape must match bounding box"
    assert (
        rotated.shape != shape
    ), "Rotated shape should differ from input for non-zero rotation"


def test_scale_moving_image_correct_shape_2d():
    """
    Test that a 2D moving image is correctly scaled to match atlas resolution.
    """
    img = np.random.rand(100, 200)  # Shape in (y, x)
    moving_res = (1.0, 2.0, 4.0)  # z, y, x (ignored z for 2D)
    atlas_res = (1.0, 1.0, 1.0)  # Target resolution

    # Expected output shape:
    expected_y = int(img.shape[0] * (moving_res[1] / atlas_res[1]))
    expected_x = int(img.shape[1] * (moving_res[2] / atlas_res[2]))

    scaled = scale_moving_image(img, atlas_res, moving_res)

    assert scaled.shape == (
        expected_y,
        expected_x,
    ), f"Expected {(expected_y, expected_x)}, got {scaled.shape}"
    assert isinstance(scaled, np.ndarray)


def test_scale_moving_image_invalid_scale():
    """
    Test that scaling with invalid resolution raises ValueError.
    """
    img = np.random.rand(50, 50)
    with pytest.raises(ValueError, match="Pixel sizes must be greater than 0"):
        _ = scale_moving_image(
            img, atlas_res=(25.0, 25.0, 25.0), moving_res=(0, 0, 0)
        )
