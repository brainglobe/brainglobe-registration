"""
Tests for similarity metrics functions in
brainglobe_registration.similarity_metrics
"""

from pathlib import Path

import numpy as np
import pytest
from skimage import io

from brainglobe_registration.similarity_metrics import (
    compare_image_to_atlas_slices,
    mutual_information,
    normalized_cross_correlation,
    structural_similarity_index,
)


class TestSimilarityMetrics:
    """Test class for similarity metrics functions."""

    def setup_method(self):
        """Set up test data using existing test images."""
        # Define path to test images
        test_images_dir = Path(__file__).parent / "test_images"

        # Create simple test images for basic testing
        self.img_zeros = np.zeros((10, 10))
        self.img_ones = np.ones((10, 10))
        self.img_random = np.random.rand(10, 10)

        # Image with gradient
        self.img_gradient = np.linspace(0, 1, 100).reshape(10, 10)

        # Different sized images
        self.img_larger = np.ones((15, 15))

        # Image with NaN
        self.img_with_nan = np.ones((10, 10))
        self.img_with_nan[5, 5] = np.nan

        # Simple mock atlas volume
        self.atlas_volume = np.zeros((5, 10, 10))
        for i in range(5):
            self.atlas_volume[i, :, :] = i * np.ones((10, 10))

        # Try to load specific test images
        sample_hipp_path = test_images_dir / "sample_hipp.tif"
        atlas_hipp_path = test_images_dir / "Atlas_Hipp.tif"

        # Load sample image if it exists
        if sample_hipp_path.exists():
            try:
                self.test_image = io.imread(str(sample_hipp_path))

                # If image is 3D or RGB, handle appropriately
                if len(self.test_image.shape) > 2:
                    if len(
                        self.test_image.shape
                    ) == 3 and self.test_image.shape[2] in [3, 4]:
                        self.test_image = np.mean(
                            self.test_image[:, :, :3], axis=2
                        )
                    else:
                        self.test_image = self.test_image[0]
            except Exception as e:
                print(f"Could not load sample_hipp.tif: {e}")
                self.test_image = self.img_random  # Fallback
        else:
            self.test_image = self.img_random

        # Load atlas image if it exists
        if atlas_hipp_path.exists():
            try:
                atlas_image = io.imread(str(atlas_hipp_path))

                # Handle 3D or RGB atlas
                if len(atlas_image.shape) > 2:
                    if len(atlas_image.shape) == 3 and atlas_image.shape[
                        2
                    ] in [3, 4]:
                        atlas_image = np.mean(atlas_image[:, :, :3], axis=2)
                    else:
                        atlas_image = atlas_image[0]

                # Create mini atlas from the 2D image
                self.atlas_volume = np.stack([atlas_image for _ in range(5)])
            except Exception as e:
                print(f"Could not load Atlas_Hipp.tif: {e}")
                # Fallback to synthetic atlas (already created)

    def test_normalized_cross_correlation_identical(self):
        """Test NCC with identical images."""
        # Create non-constant images for testing correlation
        img_varying = np.arange(100).reshape(10, 10) / 100.0

        # Test with varying identical images
        result = normalized_cross_correlation(img_varying, img_varying)
        assert result > 0.99  # Should be close to 1 for identical images

    def test_normalized_cross_correlation_different(self):
        """Test NCC with completely different images."""
        result = normalized_cross_correlation(self.img_ones, self.img_zeros)
        assert result == 0  # Should be 0 for uncorrelated images

    def test_normalized_cross_correlation_different_sizes(self):
        """Test NCC with different sized images."""
        result = normalized_cross_correlation(self.img_ones, self.img_larger)
        assert isinstance(result, float)
        # Should handle different sizes correctly and return a float

    def test_mutual_information_identical(self):
        """Test MI with identical images."""
        result = mutual_information(self.img_ones, self.img_ones)
        assert result > 0  # Should be positive for identical images

    def test_mutual_information_different(self):
        """Test MI with different images."""
        result = mutual_information(self.img_random, self.img_gradient)
        assert isinstance(result, float)
        # Should return a valid MI value

    def test_mutual_information_different_sizes(self):
        """Test MI with different sized images."""
        result = mutual_information(self.img_ones, self.img_larger)
        assert isinstance(result, float)
        # Should handle different sizes correctly and return a float

    def test_structural_similarity_identical(self):
        """Test SSIM with identical images."""
        result = structural_similarity_index(self.img_ones, self.img_ones)
        assert result > 0.99  # Should be close to 1 for identical images

    def test_structural_similarity_different(self):
        """Test SSIM with different images."""
        result = structural_similarity_index(self.img_ones, self.img_zeros)
        assert result < 0.1  # Should be close to 0 for very different images

    def test_structural_similarity_different_sizes(self):
        """Test SSIM with different sized images."""
        result = structural_similarity_index(self.img_ones, self.img_larger)
        assert isinstance(result, float)
        # Should handle different sizes correctly and return a float

    def test_handling_nan_values(self):
        """Test handling of NaN values."""
        # NCC should handle NaN values appropriately
        result_ncc = normalized_cross_correlation(
            self.img_with_nan, self.img_ones
        )
        assert not np.isnan(result_ncc)

        # MI should handle NaN values appropriately
        result_mi = mutual_information(self.img_with_nan, self.img_ones)
        assert not np.isnan(result_mi)

        # SSIM should handle NaN values appropriately
        result_ssim = structural_similarity_index(
            self.img_with_nan, self.img_ones
        )
        assert not np.isnan(result_ssim)

    def test_compare_image_to_atlas_slices(self):
        """Test the compare_image_to_atlas_slices function."""
        # Test with different metrics
        for metric in ["ncc", "mi", "ssim"]:
            results = compare_image_to_atlas_slices(
                self.img_ones,
                self.atlas_volume,
                slice_range=(0, 5),
                metric=metric,
            )

            # Should return a dictionary
            assert isinstance(results, dict)

            # Should have entries for each slice
            assert len(results) == 5

            # Each result should be a float
            for slice_idx, similarity in results.items():
                assert isinstance(similarity, float) or similarity == float(
                    "-inf"
                )

    def test_compare_image_to_atlas_slices_invalid_metric(self):
        """Test compare_image_to_atlas_slices with invalid metric."""
        with pytest.raises(ValueError):
            compare_image_to_atlas_slices(
                self.img_ones, self.atlas_volume, metric="invalid_metric"
            )

    def test_compare_image_to_atlas_slices_with_nan(self):
        """Test compare_image_to_atlas_slices with NaN values in atlas."""
        # Create atlas with NaN - ensure it's float type first
        atlas_with_nan = self.atlas_volume.copy().astype(np.float32)
        atlas_with_nan[2, 5, 5] = np.nan

        results = compare_image_to_atlas_slices(
            self.img_gradient, atlas_with_nan, slice_range=(0, 5), metric="ncc"
        )

        # Should still return results for all slices
        assert len(results) == 5

        # The slice with NaN should have a very low value or -inf
        assert results[2] == float("-inf") or results[2] < 0
