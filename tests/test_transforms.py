import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from brainglobe_registration.utils.transforms import (
    create_rotation_matrix,
    rotate_volume,
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
    assert_array_almost_equal(transform, expected, decimal=5)
    assert bbox == shape


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
    """Check that rotate_volume returns correct shape and dtype."""
    shape = dummy_volume.shape
    transform, bbox = create_rotation_matrix(0, 0, 0, shape)

    rotated = rotate_volume(dummy_volume, shape, transform, bbox)
    assert rotated.shape == bbox, "Output shape mismatch with bounding box"
    assert rotated.dtype == dummy_volume.dtype, "Dtype mismatch after rotation"


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
