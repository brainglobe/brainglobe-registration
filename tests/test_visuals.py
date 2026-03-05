"""Tests for visualization utilities in brainglobe_registration.utils.visuals."""

import numpy as np
import pytest

from brainglobe_registration.utils.visuals import (
    generate_checkerboard,
    generate_intensity_difference_map,
)


def test_generate_checkerboard_2d_basic():
    """Test basic 2D checkerboard generation."""
    image1 = np.ones((100, 100), dtype=np.uint8) * 255
    image2 = np.zeros((100, 100), dtype=np.uint8)
    image1[0:50, :] = 200

    checkerboard = generate_checkerboard(image1, image2, square_size=32)

    assert checkerboard.shape == (100, 100)
    assert checkerboard.min() >= 0
    assert checkerboard.max() <= 65535
    assert checkerboard.dtype == np.uint16
    assert checkerboard.min() < checkerboard.max()


def test_generate_checkerboard_2d_square_pattern():
    """Test that checkerboard creates alternating pattern."""
    image1 = np.ones((64, 64), dtype=np.uint8) * 255
    image2 = np.ones((64, 64), dtype=np.uint8) * 100
    image2[0:32, 0:32] = 200

    checkerboard = generate_checkerboard(
        image1, image2, square_size=16, normalize=True
    )

    assert np.std(checkerboard) > 0


def test_generate_checkerboard_3d():
    """Test 3D checkerboard generation."""
    image1 = np.ones((10, 100, 100), dtype=np.uint8) * 255
    image2 = np.zeros((10, 100, 100), dtype=np.uint8)

    checkerboard = generate_checkerboard(image1, image2, square_size=32)

    assert checkerboard.shape == (10, 100, 100)
    assert checkerboard.min() >= 0
    assert checkerboard.max() <= 65535
    assert checkerboard.dtype == np.uint16


def test_generate_checkerboard_shape_mismatch():
    """Test that checkerboard crops to minimum shape for mismatched shapes."""
    image1 = np.ones((100, 100), dtype=np.uint8)
    image2 = np.ones((100, 101), dtype=np.uint8)

    checkerboard = generate_checkerboard(image1, image2)

    assert checkerboard.shape == (100, 100)


def test_generate_checkerboard_unsupported_dimension():
    """Test that checkerboard raises error for unsupported dimensions."""
    image1 = np.ones((10, 10, 10, 10), dtype=np.uint8)
    image2 = np.ones((10, 10, 10, 10), dtype=np.uint8)

    with pytest.raises(ValueError, match="Unsupported image dimensionality"):
        generate_checkerboard(image1, image2)


def test_generate_checkerboard_normalize_false():
    """Test checkerboard generation without normalization."""
    image1 = np.ones((100, 100), dtype=np.uint16) * 1000
    image2 = np.ones((100, 100), dtype=np.uint16) * 500

    checkerboard = generate_checkerboard(
        image1, image2, square_size=32, normalize=False
    )

    assert checkerboard.dtype == np.uint16
    assert checkerboard.min() >= 0
    assert checkerboard.max() <= 1000


def test_generate_checkerboard_all_same_values():
    """Test checkerboard with identical images (edge case)."""
    image1 = np.ones((100, 100), dtype=np.uint8) * 128
    image2 = np.ones((100, 100), dtype=np.uint8) * 128

    checkerboard = generate_checkerboard(image1, image2, square_size=32)

    assert checkerboard.dtype == np.uint16
    assert np.all(checkerboard == checkerboard.flat[0])


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

    slice_0 = checkerboard[0, :, :]
    slice_2 = checkerboard[2, :, :]
    slice_4 = checkerboard[4, :, :]

    np.testing.assert_array_equal(slice_0, slice_2)
    np.testing.assert_array_equal(slice_2, slice_4)


def test_generate_intensity_difference_map_2d_basic():
    """Same shape, normalized: output in [0,1], float32."""
    a = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    b = np.array([[0.0, 2.0], [1.0, 3.0]], dtype=np.float32)
    out = generate_intensity_difference_map(a, b, normalize=True)
    assert out.shape == (2, 2)
    assert out.dtype == np.float32
    assert 0 <= out.min() and out.max() <= 1.0
    np.testing.assert_allclose(out[0, 1], 1.0 / 3.0, rtol=1e-5)
    np.testing.assert_allclose(out[1, 0], 1.0 / 3.0, rtol=1e-5)


def test_generate_intensity_difference_map_2d_shape_mismatch():
    """Different shapes: crop to overlap."""
    a = np.ones((10, 10), dtype=np.uint8)
    b = np.ones((10, 12), dtype=np.uint8)
    out = generate_intensity_difference_map(a, b, normalize=True)
    assert out.shape == (10, 10)


def test_generate_intensity_difference_map_3d():
    """3D same shape."""
    a = np.random.rand(4, 5, 6).astype(np.float32)
    b = np.random.rand(4, 5, 6).astype(np.float32)
    out = generate_intensity_difference_map(a, b, normalize=True)
    assert out.shape == (4, 5, 6)
    assert out.dtype == np.float32
    assert 0 <= out.min() and out.max() <= 1.0


def test_generate_intensity_difference_map_normalize_false():
    """Without normalization, output reflects raw abs difference."""
    a = np.array([[10, 20]], dtype=np.int32)
    b = np.array([[12, 18]], dtype=np.int32)
    out = generate_intensity_difference_map(a, b, normalize=False)
    assert out.dtype == np.float32
    np.testing.assert_array_almost_equal(out, [[2, 2]])


def test_generate_intensity_difference_map_ndim_mismatch():
    """2D vs 3D raises ValueError."""
    a = np.ones((5, 5))
    b = np.ones((2, 5, 5))
    with pytest.raises(ValueError, match="same number of dimensions"):
        generate_intensity_difference_map(a, b)


def test_generate_intensity_difference_map_unsupported_ndim():
    """1D raises ValueError."""
    a = np.ones(10)
    b = np.ones(10)
    with pytest.raises(ValueError, match="Only 2D and 3D"):
        generate_intensity_difference_map(a, b)
