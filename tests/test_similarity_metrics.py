"""
Tests for similarity metrics functions in
brainglobe_registration.similarity_metrics
"""

import numpy as np
import pytest

from brainglobe_registration.similarity_metrics import (
    compare_image_to_atlas_slices,
    find_best_atlas_slice,
    mutual_information,
    normalized_cross_correlation,
    structural_similarity_index,
)


class TestSimilarityMetrics:
    """Test class for similarity metrics functions."""

    def setup_method(self):
        """Set up test data using existing test images."""
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

        # Create a controlled test atlas with constant slices of
        # values 0, 1, 2, 3, 4
        self.test_atlas_volume = np.zeros((5, 10, 10))
        for i in range(5):
            self.test_atlas_volume[i, :, :] = i * np.ones((10, 10))

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
        """Test MI with identical constant images."""
        result = mutual_information(self.img_ones, self.img_ones)
        assert result == 1.0  # Max NMI for identical constant images

    def test_mutual_information_different(self):
        """Test MI with different images."""
        result = mutual_information(self.img_random, self.img_gradient)
        # Should return a valid NMI value. NMI is typically [0,1] but sklearn's
        # implementation might exceed 1 in some cases. Check non-negativity.
        assert isinstance(result, float)
        assert result >= 0.0

    def test_mutual_information_different_sizes(self):
        """Test MI with different sized, but identical constant images."""
        result = mutual_information(self.img_ones, self.img_larger)
        # Both are constant '1', should return max NMI
        assert result == 1.0

    def test_mutual_information_constant_different(self):
        """Test MI with different constant images."""
        result = mutual_information(self.img_ones, self.img_zeros)
        # Different constant images should return min NMI
        assert result == 0.0

    def test_mutual_information_one_constant(self):
        """Test MI with one constant image and one varying image."""
        result = mutual_information(self.img_ones, self.img_gradient)
        # One constant image should return min NMI
        assert result == 0.0

    def test_ncc_handling_nan(self):
        """Test NCC handling of NaN values."""
        result_ncc = normalized_cross_correlation(
            self.img_with_nan, self.img_ones
        )
        assert not np.isnan(result_ncc)

    def test_mi_handling_nan(self):
        """Test MI handling of NaN values."""
        result_mi = mutual_information(self.img_with_nan, self.img_ones)
        assert not np.isnan(result_mi)

    def test_ssim_handling_nan(self):
        """Test SSIM handling of NaN values."""
        result_ssim = structural_similarity_index(
            self.img_with_nan, self.img_ones
        )
        assert not np.isnan(result_ssim)

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

    def test_compare_image_to_atlas_slices(self):
        """Test the compare_image_to_atlas_slices function."""
        # Test with different metrics
        for metric in ["ncc", "mi", "ssim"]:
            results = compare_image_to_atlas_slices(
                self.img_ones,  # Constant image (value 1)
                self.test_atlas_volume,  # Controlled atlas (values 0-4)
                slice_range=(0, 5),
                metric=metric,
            )

            # Should return a dictionary
            assert isinstance(results, dict)

            # Should have entries for each slice
            assert len(results) == 5

            # Test passing an iterable range
            iterable_indices = [0, 2, 4]
            iterable_results = compare_image_to_atlas_slices(
                self.img_ones,
                self.test_atlas_volume,
                slice_range=iterable_indices,
                metric=metric,
            )
            assert len(iterable_results) == len(iterable_indices)
            assert all(idx in iterable_results for idx in iterable_indices)

            # Check specific values for MI with constant images
            if metric == "mi":
                # For MI: 1.0 if identical constant, 0.0 if different constant
                expected = {
                    0: 0.0, 1: 1.0, 2: 0.0, 3: 0.0, 4: 0.0
                }  # fmt: skip
                # Check MI values match expected values
                for slice_idx, similarity in results.items():
                    assert np.isclose(
                        similarity, expected[slice_idx]
                    ), f"MI mismatch for slice {slice_idx}"
            else:
                # Each result should be a float (or -inf for
                # NCC/SSIM edge cases)
                for slice_idx, similarity in results.items():
                    assert isinstance(
                        similarity, float
                    ) or similarity == float("-inf")

    def test_compare_image_to_atlas_slices_invalid_metric(self):
        """Test compare_image_to_atlas_slices with invalid metric."""
        with pytest.raises(ValueError):
            compare_image_to_atlas_slices(
                self.img_ones,
                self.test_atlas_volume,
                metric="invalid_metric",
                slice_range=(0, 1),
            )

    def test_compare_image_to_atlas_slices_with_nan(self):
        """Test compare_image_to_atlas_slices with NaN values in atlas."""
        # Create atlas with NaN - ensure it's float type first
        atlas_with_nan = self.test_atlas_volume.copy().astype(np.float32)
        atlas_with_nan[2, 5, 5] = np.nan

        results = compare_image_to_atlas_slices(
            self.img_gradient, atlas_with_nan, slice_range=(0, 5), metric="ncc"
        )

        # Should still return results for all slices
        assert len(results) == 5

        # The slice with NaN should have a very low value or -inf
        assert results[2] == float("-inf")

    def test_mutual_information_constant_images(self):
        """Test MI with constant images.

        Tests both same and different constant value cases.
        """
        # Both images are constant with same value
        constant_img1 = np.ones((10, 10)) * 5
        constant_img2 = np.ones((10, 10)) * 5
        result = mutual_information(constant_img1, constant_img2)
        # Perfect mutual information for identical constant images
        assert result == 1.0

        # Both images are constant with different values
        constant_img3 = np.ones((10, 10)) * 10
        result = mutual_information(constant_img1, constant_img3)
        # No mutual information for different constant images
        assert result == 0.0

        # One image is constant, the other varies
        result = mutual_information(constant_img1, self.img_gradient)
        # No mutual information when one image is constant
        assert result == 0.0

    def test_mutual_information_error_handling(self):
        """Test MI error handling."""
        # Create images that might cause calculation errors
        img_zeros = np.zeros((10, 10))

        # For two identical constant images (even zeros), NMI returns 1.0
        result = mutual_information(img_zeros, img_zeros)
        # Two identical constant images have perfect MI
        assert result == 1.0

    def test_structural_similarity_error_handling(self):
        """Test SSIM error handling."""
        # Create images that might cause calculation errors
        img_zeros = np.zeros((10, 10))

        # This should handle potential errors and return a valid result
        result = structural_similarity_index(img_zeros, img_zeros)
        assert not np.isnan(result)

    # --- Tests for find_best_atlas_slice ---

    def test_find_best_slice_basic_metrics(self):
        """Test finding best slice with default range for MI, NCC, SSIM."""
        # First, find expected best slices for NCC/SSIM using compare
        metrics_expected_best = {}
        for metric in ["ncc", "ssim"]:
            results = compare_image_to_atlas_slices(
                self.img_ones,
                self.test_atlas_volume,
                slice_range=(0, self.test_atlas_volume.shape[0]),
                metric=metric,
            )
            # Filter out -inf before finding max
            valid_results = {
                k: v for k, v in results.items() if v != float("-inf")
            }
            if valid_results:
                best_slice = max(valid_results.items(), key=lambda x: x[1])[0]
                metrics_expected_best[metric] = best_slice
            else:
                # Handle case where all results are -inf, though unlikely here
                metrics_expected_best[metric] = -1  # Or some other indicator

        # Test MI (known to be slice 3 for constant images)
        best_slice_mi = find_best_atlas_slice(
            self.img_ones, self.test_atlas_volume, metric="mi"
        )
        assert (
            best_slice_mi == 1
        ), f"Expected slice 1 for MI, got {best_slice_mi}"

        # Test NCC and SSIM against their calculated best slices
        for metric in ["ncc", "ssim"]:
            best_slice = find_best_atlas_slice(
                self.img_ones, self.test_atlas_volume, metric=metric
            )
            expected = metrics_expected_best[metric]
            assert (
                best_slice == expected
            ), f"Expected slice {expected} for {metric}, got {best_slice}"

    def test_find_best_slice_custom_range(self):
        """Test finding best slice with a custom search_range."""
        # Use MI metric. Create image matching slice 1.
        test_image_slice1 = 1 * np.ones((10, 10))
        search_range = (0, 3)  # Indices 0, 1, 2
        best_slice = find_best_atlas_slice(
            test_image_slice1,
            self.test_atlas_volume,
            metric="mi",
            search_range=search_range,
        )
        # Atlas values: 0, 1, 2. Image value: 1.
        # MI scores: 0.0, 1.0, 0.0. Best is slice 1.
        assert (
            best_slice == 1
        ), f"Expected slice 1 within range {search_range}, got {best_slice}"

    def test_find_best_slice_all_invalid(self):
        """Test finding best slice when all comparisons are invalid (-inf)."""
        atlas_all_nan = np.full_like(self.test_atlas_volume, np.nan)
        search_range = (1, 4)  # Search slices 1, 2, 3

        # Should now capture the UserWarning correctly
        with pytest.warns(
            UserWarning,
            match="All compared slices yielded invalid similarity values.",
        ):
            best_slice_all_bad = find_best_atlas_slice(
                self.img_ones,
                atlas_all_nan,
                metric="mi",
                search_range=search_range,
            )

        # Should return the first index of the searched range as fallback
        expected_fallback = search_range[0]
        assert best_slice_all_bad == expected_fallback, (
            f"Expected fallback index {expected_fallback} when all invalid, "
            f"got {best_slice_all_bad}"
        )

    def test_find_best_slice_step(self):
        """Test finding best slice with a search_step."""
        # Create image matching slice 4
        test_image_slice4 = 4 * np.ones((10, 10))
        step_size = 2

        best_slice = find_best_atlas_slice(
            test_image_slice4,
            self.test_atlas_volume,
            metric="mi",
            search_step=step_size,
        )
        # With step=2, searches slices 0, 2, 4.
        # Atlas values: 0, 2, 4. Image value: 4.
        # MI scores: 0.0, 0.0, 1.0. Best is slice 4.
        assert (
            best_slice == 4
        ), f"Expected slice 4 with step size {step_size}, got {best_slice}"

    def test_compare_image_to_atlas_slices_error_handling(self, mocker):
        """Test error handling in compare_image_to_atlas_slices."""
        # Test with out-of-bounds indices in iterable
        invalid_indices = [-1, 0, 5, 10]  # Atlas shape[0] is 5
        results = compare_image_to_atlas_slices(
            self.img_ones,
            self.test_atlas_volume,
            slice_range=invalid_indices,
            metric="mi",
        )

        # Should return results for all requested indices
        assert len(results) == len(invalid_indices)
        # Valid index (0) should have a valid score (MI=0.0 for 1 vs 0)
        assert np.isclose(results[0], 0.0)
        # Invalid indices should have -inf
        assert results[-1] == float("-inf")
        assert results[5] == float("-inf")
        assert results[10] == float("-inf")

        # Test when the underlying metric function raises an exception
        mocker.patch(
            "brainglobe_registration.similarity_metrics.normalized_cross_correlation",
            side_effect=ValueError("Deliberate test exception"),
        )

        # This should catch the exception from the mocked NCC and return -inf
        results_on_error = compare_image_to_atlas_slices(
            self.img_gradient,
            self.test_atlas_volume,
            metric="ncc",
            slice_range=(0, self.test_atlas_volume.shape[0]),
        )

        # Should still return results for all slices
        assert len(results_on_error) == self.test_atlas_volume.shape[0]

        # All results should be -inf because the mocked function always fails
        for slice_idx, similarity in results_on_error.items():
            assert similarity == float("-inf"), (
                f"Expected -inf for slice {slice_idx} on error, "
                f"got {similarity}"
            )

    def test_compare_image_to_atlas_slices_metric_exception(self, mocker):
        """
        Test compare_image_to_atlas_slices handles exceptions from
        metric functions (e.g., NCC, MI, SSIM)
        """
        # Patch the target function using mocker.patch
        # Use ValueError as it's a plausible exception type here
        # Set side_effect to raise the desired exception
        mocker.patch(
            "brainglobe_registration.similarity_metrics.normalized_cross_correlation",
            side_effect=ValueError("Deliberate test exception"),
        )

        # This should catch the exception from the mocked NCC and return -inf
        results = compare_image_to_atlas_slices(
            self.img_gradient,
            self.test_atlas_volume,
            metric="ncc",
            slice_range=(0, self.test_atlas_volume.shape[0]),
        )

        # Should return results for all requested indices, but -inf
        # for invalid ones
        assert len(results) == self.test_atlas_volume.shape[0]

        # All results should be -inf because the mocked function always fails
        for slice_idx, similarity in results.items():
            assert similarity == float("-inf"), (
                f"Expected -inf for slice {slice_idx} on error, "
                f"got {similarity}"
            )
