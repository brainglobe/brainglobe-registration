import warnings
from typing import Dict, Iterable, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import skimage.metrics


def _match_image_sizes(
    img1: npt.NDArray, img2: npt.NDArray
) -> Tuple[Optional[npt.NDArray], Optional[npt.NDArray], bool]:
    """
    Match the sizes of two images by cropping to the smaller dimensions.
    Also checks for NaNs in the cropped regions and handles them.

    Parameters
    ----------
    img1 : npt.NDArray
        First image
    img2 : npt.NDArray
        Second image

    Returns
    -------
    Tuple[Optional[npt.NDArray], Optional[npt.NDArray], bool]
        Tuple of (cropped_img1, cropped_img2, contains_nan).
        Images are None if contains_nan is True.
        If contains_nan is False, images are guaranteed to be NaN-free.
    """
    min_shape = np.minimum(img1.shape, img2.shape)
    img1_crop = img1[: min_shape[0], : min_shape[1]]
    img2_crop = img2[: min_shape[0], : min_shape[1]]

    # Check for NaNs BEFORE converting them
    contains_nan = np.isnan(img1_crop).any() or np.isnan(img2_crop).any()

    if contains_nan:
        return None, None, True
    else:
        img1_processed = np.nan_to_num(img1_crop)
        img2_processed = np.nan_to_num(img2_crop)
        return img1_processed, img2_processed, False


def _normalize_image(img: npt.NDArray) -> npt.NDArray:
    """Normalize image array to the range [0, 1]."""
    img_float = img.astype(np.float32)
    max_val = np.max(img_float)
    if max_val != 0:
        return img_float / max_val
    else:
        return img_float


def normalized_cross_correlation(
    img1: npt.NDArray, img2: npt.NDArray
) -> float:
    """
    Calculate the normalized cross-correlation between two images.

    Parameters
    ----------
    img1 : numpy.ndarray
        First image
    img2 : numpy.ndarray
        Second image

    Returns
    -------
    float
        Normalized cross-correlation value between -1 and 1
    """
    # Ensure both images are the same size and handle NaNs
    img1_proc, img2_proc, contains_nan = _match_image_sizes(img1, img2)

    if contains_nan:
        return float("-inf")

    # At this point we know img1_proc and img2_proc are not None
    assert img1_proc is not None and img2_proc is not None

    # Calculate means
    img1_mean = np.mean(img1_proc)
    img2_mean = np.mean(img2_proc)

    # Calculate normalized cross-correlation
    numerator = np.sum((img1_proc - img1_mean) * (img2_proc - img2_mean))
    denominator = np.sqrt(
        np.sum((img1_proc - img1_mean) ** 2)
        * np.sum((img2_proc - img2_mean) ** 2)
    )

    # Check for zero denominator (happens when one or both images are
    # constant)
    if denominator == 0:
        return 0.0

    return numerator / denominator


def mutual_information(
    img1: npt.NDArray, img2: npt.NDArray, bins: int = 256
) -> float:
    """
    Calculate the mutual information between two images.

    Parameters
    ----------
    img1 : numpy.ndarray
        First image
    img2 : numpy.ndarray
        Second image
    bins : int, optional
        Number of bins for histogram, by default 256

    Returns
    -------
    float
        Mutual information value
    """
    # Ensure both images are the same size and handle NaNs
    img1_proc, img2_proc, contains_nan = _match_image_sizes(img1, img2)

    if contains_nan:
        return float("-inf")

    # Check for constant images BEFORE normalization
    # At this point we know img1_proc and img2_proc are not None
    assert img1_proc is not None and img2_proc is not None
    is_img1_constant = np.all(img1_proc == img1_proc.flat[0])
    is_img2_constant = np.all(img2_proc == img2_proc.flat[0])

    if is_img1_constant and is_img2_constant:
        # Both are constant images
        return 1.0 if img1_proc.flat[0] == img2_proc.flat[0] else 0.0
    elif is_img1_constant or is_img2_constant:
        # If only one image is constant, MI is 0
        return 0.0

    # Normalize images to [0, 1] range
    img1_norm = _normalize_image(img1_proc)
    img2_norm = _normalize_image(img2_proc)

    try:
        # Calculate normalized mutual information using scikit-image
        mi_value = skimage.metrics.normalized_mutual_information(
            img1_norm, img2_norm
        )

        # Handle potential NaN results - Although NMI should be robust
        if np.isnan(mi_value):
            return 0.0

        return mi_value
    except (ValueError, RuntimeError, TypeError) as e:
        # If any error occurs during calculation, print warning and return -inf
        warnings.warn(f"Warning: Error calculating mutual information: {e}")
        return float("-inf")


def structural_similarity_index(img1: npt.NDArray, img2: npt.NDArray) -> float:
    """
    Calculate the structural similarity index between two images.

    Parameters
    ----------
    img1 : numpy.ndarray
        First image
    img2 : numpy.ndarray
        Second image

    Returns
    -------
    float
        Structural similarity index value between -1 and 1
    """
    # Ensure both images are the same size and handle NaNs
    img1_proc, img2_proc, contains_nan = _match_image_sizes(img1, img2)

    if contains_nan:
        return float("-inf")

    # At this point we know img1_proc and img2_proc are not None
    assert img1_proc is not None and img2_proc is not None

    # Normalize images to [0,1]
    img1_norm = _normalize_image(img1_proc)
    img2_norm = _normalize_image(img2_proc)

    try:
        # Use scikit-image SSIM function
        ssim_value = skimage.metrics.structural_similarity(
            img1_norm, img2_norm, data_range=1.0
        )

        # Handle potential NaN results
        if np.isnan(ssim_value):
            return 0.0

        return ssim_value
    except (ValueError, RuntimeError, TypeError) as e:
        # If any error occurs during calculation, print warning and return -inf
        warnings.warn(
            f"Warning: Error calculating structural similarity index: {e}"
        )
        return float("-inf")


def compare_all_metrics(image1: npt.NDArray, image2: npt.NDArray) -> dict:
    """
    Calculate all similarity metrics between two images.

    Parameters
    ----------
    image1 : npt.NDArray
        First input image
    image2 : npt.NDArray
        Second input image

    Returns
    -------
    dict
        Dictionary with all similarity values
    """
    return {
        "ncc": normalized_cross_correlation(image1, image2),
        "mi": mutual_information(image1, image2),
        "ssim": structural_similarity_index(image1, image2),
    }


# Dictionary mapping metric names to functions
METRIC_FUNCTIONS = {
    "ncc": lambda img1, img2: normalized_cross_correlation(img1, img2),
    "mi": lambda img1, img2: mutual_information(img1, img2),
    "ssim": lambda img1, img2: structural_similarity_index(img1, img2),
}


def compare_image_to_atlas_slices(
    moving_image: npt.NDArray,
    atlas_volume: npt.NDArray,
    slice_range: Union[Tuple[int, int], Iterable[int]],
    metric: str = "mi",
) -> Dict[int, float]:
    """
    Compare an input image with multiple atlas slices.

    Parameters
    ----------
    moving_image : npt.NDArray
        Input image to compare
    atlas_volume : npt.NDArray
        3D atlas volume
    slice_range : Union[Tuple[int, int], Iterable[int]],
        Either a tuple (start_slice, end_slice) defining a contiguous range
        (exclusive of end_slice), or an iterable of specific slice indices
        to compare.
    metric : str, optional
        Similarity metric ("ncc", "mi", "ssim"), by default "mi"

    Returns
    -------
    Dict[int, float]
        Dictionary with slice indices as keys and similarity values as values
    """
    # Determine the actual indices to iterate over
    indices_to_process: Iterable[int]
    if isinstance(slice_range, tuple) and len(slice_range) == 2:
        indices_to_process = range(slice_range[0], slice_range[1])
    elif hasattr(slice_range, "__iter__"):
        indices_to_process = slice_range
    else:
        raise TypeError(
            "slice_range must be a tuple (start, end) or an iterable of "
            "indices"
        )

    if metric not in METRIC_FUNCTIONS:
        raise ValueError(
            f"Invalid metric: {metric}. "
            f"Choose from {list(METRIC_FUNCTIONS.keys())}"
        )

    metric_func = METRIC_FUNCTIONS[metric]

    results = {}
    for slice_idx in indices_to_process:
        # Ensure index is within bounds
        if not (0 <= slice_idx < atlas_volume.shape[0]):
            results[slice_idx] = float("-inf")  # Or skip/warn
            continue

        atlas_slice = atlas_volume[slice_idx, :, :]

        # Let metric functions handle specific cases like constant images or
        # NaNs.
        # Returns -inf if appropriate.
        try:
            results[slice_idx] = metric_func(moving_image, atlas_slice)
        except (ValueError, RuntimeError, TypeError) as e:
            # Handle any errors that might occur during metric calculation
            warnings.warn(f"Error processing slice {slice_idx}: {e}")
            results[slice_idx] = float("-inf")

    return results


def find_best_atlas_slice(
    moving_image: npt.NDArray,
    atlas_volume: npt.NDArray,
    metric: str = "mi",
    search_range: Optional[Tuple[int, int]] = None,
    search_step: int = 1,
) -> int:
    """
    Find the best matching atlas slice for a given image.

    Parameters
    ----------
    moving_image : npt.NDArray
        Input image to compare
    atlas_volume : npt.NDArray
        3D atlas volume
    metric : str, optional
        Metric to use for comparison, by default "mi"
        Options: "ncc", "mi", "ssim"
    search_range : tuple, optional
        Range of slices to search (start, end), by default None
        If None, searches all slices
    search_step : int, optional
        Step size for searching slices, by default 1

    Returns
    -------
    int
        Index of the best matching slice
    """
    if search_range is None:
        slice_indices = range(0, atlas_volume.shape[0], search_step)
    else:
        start_slice, end_slice = search_range
        # Ensure start/end are within bounds
        start_slice = max(0, start_slice)
        end_slice = min(atlas_volume.shape[0], end_slice)
        slice_indices = range(start_slice, end_slice, search_step)

    # Use compare_image_to_atlas_slices to get similarities
    similarity_results = compare_image_to_atlas_slices(
        moving_image, atlas_volume, slice_indices, metric=metric
    )

    # Check if similarity_results is empty (e.g., invalid range)
    if not similarity_results:
        warnings.warn("No valid slices found in the specified search range.")
        # Return first index of the search range as fallback
        return slice_indices[0] if len(slice_indices) > 0 else 0

    # Find the slice index with the maximum similarity value
    best_slice_index = max(
        similarity_results,
        key=lambda k: similarity_results.get(k, float("-inf")),
    )

    # Check if the best score is -inf (meaning all calculations failed or
    # had NaNs)
    if similarity_results[best_slice_index] == float("-inf"):
        warnings.warn("All compared slices yielded invalid similarity values.")
        # Return first index of the search range as fallback
        return slice_indices[0] if len(slice_indices) > 0 else 0

    return best_slice_index
