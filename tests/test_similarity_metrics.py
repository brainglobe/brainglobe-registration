"""
Tests for similarity metrics functions in brainglobe_registration.similarity_metrics
"""

import numpy as np
import pytest
from numpy import typing as npt

from brainglobe_registration.similarity_metrics import (
    normalized_cross_correlation,
    mutual_information,
    structural_similarity_index,
    compare_image_to_atlas_slices,
)


class TestSimilarityMetrics:
    """Test class for similarity metrics functions."""

    def setup_method(self):
        """Set up test data."""
        # Create simple test images
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
        # Make each slice have increasing values
        for i in range(5):
            self.atlas_volume[i, :, :] = i * np.ones((10, 10))

    def test_normalized_cross_correlation_identical(self):
        """Test NCC with identical images."""
        result = normalized_cross_correlation(self.img_ones, self.img_ones)
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
        result_ncc = normalized_cross_correlation(self.img_with_nan, self.img_ones)
        assert not np.isnan(result_ncc)
        
        # MI should handle NaN values appropriately
        result_mi = mutual_information(self.img_with_nan, self.img_ones)
        assert not np.isnan(result_mi)
        
        # SSIM should handle NaN values appropriately
        result_ssim = structural_similarity_index(self.img_with_nan, self.img_ones)
        assert not np.isnan(result_ssim)
    
    def test_compare_image_to_atlas_slices(self):
        """Test the compare_image_to_atlas_slices function."""
        # Test with different metrics
        for metric in ["ncc", "mi", "ssim"]:
            results = compare_image_to_atlas_slices(
                self.img_ones, self.atlas_volume, slice_range=(0, 5), metric=metric
            )
            
            # Should return a dictionary
            assert isinstance(results, dict)
            
            # Should have entries for each slice
            assert len(results) == 5
            
            # Each result should be a float
            for slice_idx, similarity in results.items():
                assert isinstance(similarity, float) or similarity == float('-inf')
    
    def test_compare_image_to_atlas_slices_invalid_metric(self):
        """Test compare_image_to_atlas_slices with invalid metric."""
        with pytest.raises(ValueError):
            compare_image_to_atlas_slices(
                self.img_ones, self.atlas_volume, metric="invalid_metric"
            )
    
    def test_compare_image_to_atlas_slices_with_nan(self):
        """Test compare_image_to_atlas_slices with NaN values in atlas."""
        # Create atlas with NaN
        atlas_with_nan = self.atlas_volume.copy()
        atlas_with_nan[2, 5, 5] = np.nan
        
        results = compare_image_to_atlas_slices(
            self.img_ones, atlas_with_nan, slice_range=(0, 5), metric="mi"
        )
        
        # Should still return results for all slices
        assert len(results) == 5
        
        # The slice with NaN should have a very low value or -inf
        assert results[2] == float('-inf') or results[2] < 0