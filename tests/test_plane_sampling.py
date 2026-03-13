"""Tests for the plane sampling utility module."""

import numpy as np
import pytest

from brainglobe_registration.utils.plane_sampling import (
    build_rotation_matrix,
    sample_annotation_plane,
    sample_plane,
)


class TestBuildRotationMatrix:
    """Tests for build_rotation_matrix()."""

    def test_identity_rotation(self):
        """Zero angles should produce identity matrix."""
        result = build_rotation_matrix(0, 0, 0)
        np.testing.assert_array_almost_equal(result, np.eye(3))

    def test_output_shape(self):
        """Should always return a 3x3 matrix."""
        result = build_rotation_matrix(10, 20, 30)
        assert result.shape == (3, 3)

    def test_orthogonal_matrix(self):
        """Rotation matrices are orthogonal: R @ R.T = I."""
        result = build_rotation_matrix(45, 30, 60)
        product = result @ result.T
        np.testing.assert_array_almost_equal(product, np.eye(3))

    def test_determinant_is_one(self):
        """Rotation matrices have determinant 1 (proper rotation)."""
        result = build_rotation_matrix(10, 20, 30)
        assert np.isclose(np.linalg.det(result), 1.0)

    def test_90_degree_roll(self):
        """90-degree roll around X-axis should swap Y and Z."""
        result = build_rotation_matrix(90, 0, 0)
        # X stays the same, Y->Z, Z->-Y
        expected = np.array(
            [
                [1, 0, 0],
                [0, 0, -1],
                [0, 1, 0],
            ]
        )
        np.testing.assert_array_almost_equal(result, expected)


class TestSamplePlane:
    """Tests for sample_plane()."""

    @pytest.fixture
    def simple_volume(self):
        """Create a simple 3D volume with known values."""
        volume = np.zeros((10, 20, 30), dtype=np.float64)
        # Put a known value at the center slice
        volume[5, :, :] = 1.0
        return volume

    @pytest.fixture
    def gradient_volume(self):
        """Create a volume where each z-slice has a distinct value."""
        volume = np.zeros((10, 20, 30), dtype=np.float64)
        for z in range(10):
            volume[z, :, :] = z
        return volume

    def test_identity_rotation_matches_direct_slice(self, gradient_volume):
        """With identity rotation, sampled plane should match direct slice."""
        z_index = 5
        result = sample_plane(
            gradient_volume,
            z_index=float(z_index),
            rotation_matrix=np.eye(3),
            interpolation_order=0,
        )
        expected = gradient_volume[z_index, :, :]
        np.testing.assert_array_almost_equal(result, expected)

    def test_output_shape_default(self, simple_volume):
        """Default output shape should be (H, W) of the volume."""
        result = sample_plane(
            simple_volume,
            z_index=5.0,
            rotation_matrix=np.eye(3),
        )
        assert result.shape == (20, 30)

    def test_output_shape_custom(self, simple_volume):
        """Custom output shape should be respected."""
        result = sample_plane(
            simple_volume,
            z_index=5.0,
            rotation_matrix=np.eye(3),
            output_shape=(10, 15),
        )
        assert result.shape == (10, 15)

    def test_out_of_bounds_returns_zero(self, simple_volume):
        """Sampling outside the volume should return zeros."""
        result = sample_plane(
            simple_volume,
            z_index=100.0,  # Way outside
            rotation_matrix=np.eye(3),
        )
        np.testing.assert_array_equal(result, np.zeros((20, 30)))

    def test_rotated_plane_not_equal_to_straight(self, gradient_volume):
        """A rotated sample should differ from a straight slice."""
        straight = sample_plane(
            gradient_volume,
            z_index=5.0,
            rotation_matrix=np.eye(3),
        )
        rotation_matrix = build_rotation_matrix(45, 0, 0)
        rotated = sample_plane(
            gradient_volume,
            z_index=5.0,
            rotation_matrix=rotation_matrix,
        )
        # They should NOT be the same
        assert not np.allclose(straight, rotated)

    def test_interpolation_order_0(self, gradient_volume):
        """Nearest-neighbor interpolation should give integer-like values."""
        result = sample_plane(
            gradient_volume,
            z_index=5.0,
            rotation_matrix=np.eye(3),
            interpolation_order=0,
        )
        # With identity rotation and integer z, all values should be exactly 5
        np.testing.assert_array_equal(result, 5.0)

    def test_mode_nearest_no_zeros_at_edge(self, gradient_volume):
        """With mode='nearest', edge pixels should clamp (not be zero)."""
        rot = build_rotation_matrix(15, 0, 0)
        result_nearest = sample_plane(
            gradient_volume,
            z_index=5.0,
            rotation_matrix=rot,
            interpolation_order=1,
            mode="nearest",
        )
        result_constant = sample_plane(
            gradient_volume,
            z_index=5.0,
            rotation_matrix=rot,
            interpolation_order=1,
            mode="constant",
            cval=0.0,
        )
        # nearest mode should have no zeros where constant mode does
        # (at least at the borders that go outside the volume)
        constant_zeros = np.sum(result_constant == 0)
        nearest_zeros = np.sum(result_nearest == 0)
        assert nearest_zeros <= constant_zeros

    def test_mode_constant_default(self, simple_volume):
        """Default mode should be 'constant' with cval=0."""
        result = sample_plane(
            simple_volume,
            z_index=100.0,  # outside volume
            rotation_matrix=np.eye(3),
        )
        # Should be all zeros with default mode='constant', cval=0
        np.testing.assert_array_equal(result, 0.0)


class TestSampleAnnotationPlane:
    """Tests for sample_annotation_plane()."""

    @pytest.fixture
    def annotation_volume(self):
        """Create a simple annotation volume with integer labels."""
        volume = np.zeros((10, 20, 30), dtype=np.int32)
        volume[5, :10, :15] = 1
        volume[5, 10:, 15:] = 2
        volume[3, :, :] = 3
        return volume

    def test_preserves_integer_labels(self, annotation_volume):
        """Annotation sampling should preserve integer label values."""
        result = sample_annotation_plane(
            annotation_volume,
            z_index=5.0,
            rotation_matrix=np.eye(3),
        )
        unique_values = set(np.unique(result))
        # Should only contain original labels (0, 1, 2)
        assert unique_values.issubset({0, 1, 2})

    def test_preserves_dtype(self, annotation_volume):
        """Output dtype should match input dtype."""
        result = sample_annotation_plane(
            annotation_volume,
            z_index=5.0,
            rotation_matrix=np.eye(3),
        )
        assert result.dtype == annotation_volume.dtype

    def test_identity_matches_direct_slice(self, annotation_volume):
        """With identity rotation, should match direct slice."""
        z_index = 5
        result = sample_annotation_plane(
            annotation_volume,
            z_index=float(z_index),
            rotation_matrix=np.eye(3),
        )
        expected = annotation_volume[z_index, :, :]
        np.testing.assert_array_equal(result, expected)

    def test_no_interpolation_artifacts(self, annotation_volume):
        """Even with rotation, should not produce fractional labels."""
        rotation_matrix = build_rotation_matrix(10, 5, 3)
        result = sample_annotation_plane(
            annotation_volume,
            z_index=5.0,
            rotation_matrix=rotation_matrix,
        )
        # All values should be integers (no interpolation artifacts)
        np.testing.assert_array_equal(result, result.astype(int))
