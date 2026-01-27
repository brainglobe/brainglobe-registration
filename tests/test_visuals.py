"""Tests for brainglobe_registration.utils.visuals."""

import numpy as np
import pytest

from brainglobe_registration.utils.visuals import (
    generate_intensity_difference_map,
)


def test_generate_intensity_difference_map_2d_basic():
    """Same shape, normalized: output in [0,1], float32."""
    a = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    b = np.array([[0.0, 2.0], [1.0, 3.0]], dtype=np.float32)
    out = generate_intensity_difference_map(a, b, normalize=True)
    assert out.shape == (2, 2)
    assert out.dtype == np.float32
    assert 0 <= out.min() and out.max() <= 1.0
    # After norm: a->[0,1], b->[0,1]; diff at (0,1)=|1-2/3|~0.33, etc.
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
