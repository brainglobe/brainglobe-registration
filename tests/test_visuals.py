"""
Tests for visualization utilities in brainglobe_registration.utils.visuals.
"""

import numpy as np
import pytest

from brainglobe_registration.utils.visuals import generate_checkerboard


def test_generate_checkerboard_2d_basic():
    """Test basic 2D checkerboard generation."""
    image1 = np.ones((100, 100), dtype=np.uint8) * 255
    image2 = np.zeros((100, 100), dtype=np.uint8)

    checkerboard = generate_checkerboard(image1, image2, square_size=32)

    # Check shape
    assert checkerboard.shape == (100, 100)

    # Check that values are normalized (between 0 and 1)
    assert checkerboard.min() >= 0.0
    assert checkerboard.max() <= 1.0

    # Check that it's not all zeros or all ones (should have pattern)
    assert checkerboard.min() < checkerboard.max()


def test_generate_checkerboard_2d_square_pattern():
    """Test that checkerboard creates alternating pattern."""
    # Create images with distinct values
    image1 = np.ones((64, 64), dtype=np.uint8) * 255
    image2 = np.ones((64, 64), dtype=np.uint8) * 100

    checkerboard = generate_checkerboard(
        image1, image2, square_size=16, normalize=True
    )

    # Check that there's variation (pattern exists)
    assert np.std(checkerboard) > 0


def test_generate_checkerboard_3d():
    """Test 3D checkerboard generation."""
    image1 = np.ones((10, 100, 100), dtype=np.uint8) * 255
    image2 = np.zeros((10, 100, 100), dtype=np.uint8)

    checkerboard = generate_checkerboard(image1, image2, square_size=32)

    # Check shape
    assert checkerboard.shape == (10, 100, 100)

    # Check that values are normalized
    assert checkerboard.min() >= 0.0
    assert checkerboard.max() <= 1.0


def test_generate_checkerboard_shape_mismatch():
    """Test that checkerboard raises error for mismatched shapes."""
    image1 = np.ones((100, 100), dtype=np.uint8)
    image2 = np.ones((100, 101), dtype=np.uint8)  # Different width

    with pytest.raises(ValueError, match="Images must have the same shape"):
        generate_checkerboard(image1, image2)


def test_generate_checkerboard_unsupported_dimension():
    """Test that checkerboard raises error for unsupported dimensions."""
    image1 = np.ones((10, 10, 10, 10), dtype=np.uint8)  # 4D
    image2 = np.ones((10, 10, 10, 10), dtype=np.uint8)

    with pytest.raises(
        ValueError, match="Unsupported image dimensionality"
    ):
        generate_checkerboard(image1, image2)


def test_generate_checkerboard_normalize_false():
    """Test checkerboard generation without normalization."""
    image1 = np.ones((100, 100), dtype=np.uint16) * 1000
    image2 = np.ones((100, 100), dtype=np.uint16) * 500

    checkerboard = generate_checkerboard(
        image1, image2, square_size=32, normalize=False
    )

    # Should preserve original value range
    assert checkerboard.dtype == np.float64  # Converted to float64 internally
    # Values should be from the input images
    assert checkerboard.min() >= 0
    assert checkerboard.max() <= 1000


def test_generate_checkerboard_all_same_values():
    """Test checkerboard with identical images (edge case)."""
    image1 = np.ones((100, 100), dtype=np.uint8) * 128
    image2 = np.ones((100, 100), dtype=np.uint8) * 128

    checkerboard = generate_checkerboard(image1, image2, square_size=32)

    # When all values are the same, normalized output should be 0.5
    assert np.allclose(checkerboard, 0.5)


def test_generate_checkerboard_different_dtypes():
    """Test checkerboard with different input dtypes."""
    image1 = np.ones((64, 64), dtype=np.float32) * 1.0
    image2 = np.ones((64, 64), dtype=np.float32) * 0.5

    checkerboard = generate_checkerboard(image1, image2, square_size=16)

    assert checkerboard.shape == (64, 64)
    assert checkerboard.min() >= 0.0
    assert checkerboard.max() <= 1.0


def test_generate_checkerboard_square_size_adaptation():
    """Test checkerboard with different square sizes."""
    image1 = np.random.rand(200, 200).astype(np.float32)
    image2 = np.random.rand(200, 200).astype(np.float32)

    for square_size in [8, 16, 32, 64]:
        checkerboard = generate_checkerboard(
            image1, image2, square_size=square_size
        )
        assert checkerboard.shape == (200, 200)


def test_generate_checkerboard_3d_multiple_slices():
    """Test that 3D checkerboard applies same pattern to all slices."""
    image1 = np.ones((5, 64, 64), dtype=np.uint8) * 255
    image2 = np.zeros((5, 64, 64), dtype=np.uint8)

    checkerboard = generate_checkerboard(image1, image2, square_size=16)

    # All slices should have the same pattern
    slice_0 = checkerboard[0, :, :]
    slice_2 = checkerboard[2, :, :]
    slice_4 = checkerboard[4, :, :]

    np.testing.assert_array_equal(slice_0, slice_2)
    np.testing.assert_array_equal(slice_2, slice_4)
