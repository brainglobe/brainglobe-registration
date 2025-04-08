import warnings
from typing import Dict, Iterable, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import skimage.metrics


def _match_image_sizes(
    img1: npt.NDArray, img2: npt.NDArray
) -> Tuple[npt.NDArray, npt.NDArray]:
    """
    Match the sizes of two images by cropping to the smaller dimensions.

    Parameters
    ----------
    img1 : npt.NDArray
        First image
    img2 : npt.NDArray
        Second image

    Returns
    -------
    Tuple[npt.NDArray, npt.NDArray]
        Tuple of cropped images with matching dimensions
    """
    min_shape = np.minimum(img1.shape, img2.shape)
    img1_crop = img1[: min_shape[0], : min_shape[1]]
    img2_crop = img2[: min_shape[0], : min_shape[1]]
    return img1_crop, img2_crop


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
    # Ensure both images are the same size
    img1_crop, img2_crop = _match_image_sizes(img1, img2)

    # Explicitly create copies to avoid modifying original arrays/views
    img1_crop = img1_crop.copy()
    img2_crop = img2_crop.copy()

    # Check for NaNs BEFORE converting them to numbers
    if np.isnan(img1_crop).any() or np.isnan(img2_crop).any():
        return float("-inf")

    # Handle NaN values by replacing them with zeros
    img1_crop = np.nan_to_num(img1_crop)
    img2_crop = np.nan_to_num(img2_crop)

    # Calculate means
    img1_mean = np.mean(img1_crop)
    img2_mean = np.mean(img2_crop)

    # Calculate normalized cross-correlation
    numerator = np.sum((img1_crop - img1_mean) * (img2_crop - img2_mean))
    denominator = np.sqrt(
        np.sum((img1_crop - img1_mean) ** 2)
        * np.sum((img2_crop - img2_mean) ** 2)
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
    # Ensure both images are the same size
    img1_crop, img2_crop = _match_image_sizes(img1, img2)

    # Check for NaNs BEFORE converting them to numbers
    if np.isnan(img1_crop).any() or np.isnan(img2_crop).any():
        return float("-inf")

    # Check for constant images BEFORE normalization
    is_img1_constant = np.all(img1_crop == img1_crop.flat[0])
    is_img2_constant = np.all(img2_crop == img2_crop.flat[0])

    if is_img1_constant and is_img2_constant:
        # Both are constant images
        return 1.0 if img1_crop.flat[0] == img2_crop.flat[0] else 0.0
    elif is_img1_constant or is_img2_constant:
        # If only one image is constant, MI is 0
        return 0.0

    # Handle NaN values by replacing them with zeros
    img1_crop = np.nan_to_num(img1_crop)
    img2_crop = np.nan_to_num(img2_crop)

    # Normalize images to [0, 1] range
    if np.max(img1_crop) != 0:
        img1_norm = img1_crop.astype(np.float32) / np.max(img1_crop)
    else:
        img1_norm = img1_crop.astype(np.float32)

    if np.max(img2_crop) != 0:
        img2_norm = img2_crop.astype(np.float32) / np.max(img2_crop)
    else:
        img2_norm = img2_crop.astype(np.float32)

    try:
        # Calculate normalized mutual information using scikit-image
        mi_value = skimage.metrics.normalized_mutual_information(
            img1_norm, img2_norm
        )

        # Handle potential NaN results - Although NMI should be robust
        if np.isnan(mi_value):
            return 0.0

        return mi_value
    except Exception as e:
        # If any error occurs during calculation, print warning and return -inf
        print(f"Warning: Error calculating mutual information: {e}")
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
    # Ensure both images are the same size
    img1_crop, img2_crop = _match_image_sizes(img1, img2)

    # Explicitly create copies to avoid modifying original arrays/views
    img1_crop = img1_crop.copy()
    img2_crop = img2_crop.copy()

    # Check for NaNs BEFORE converting them to numbers
    if np.isnan(img1_crop).any() or np.isnan(img2_crop).any():
        return float("-inf")

    # Handle NaN values by replacing them with zeros
    img1_crop = np.nan_to_num(img1_crop)
    img2_crop = np.nan_to_num(img2_crop)

    # Normalize images to [0,1]
    if np.max(img1_crop) != 0:
        img1_norm = img1_crop.astype(np.float32) / np.max(img1_crop)
    else:
        img1_norm = img1_crop.astype(np.float32)

    if np.max(img2_crop) != 0:
        img2_norm = img2_crop.astype(np.float32) / np.max(img2_crop)
    else:
        img2_norm = img2_crop.astype(np.float32)

    try:
        # Use scikit-image SSIM function
        ssim_value = skimage.metrics.structural_similarity(
            img1_norm, img2_norm, data_range=1.0
        )

        # Handle potential NaN results
        if np.isnan(ssim_value):
            return 0.0

        return ssim_value
    except Exception as e:
        # If any error occurs during calculation, print warning and return -inf
        print(f"Warning: Error calculating structural similarity index: {e}")
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

    # Select the appropriate metric function
    if metric == "ncc":
        metric_func = normalized_cross_correlation
    elif metric == "mi":
        metric_func = mutual_information
    elif metric == "ssim":
        metric_func = structural_similarity_index
    else:
        raise ValueError(f"Unknown metric: {metric}")

    results = {}
    for slice_idx in indices_to_process:
        # Ensure index is within bounds
        if not (0 <= slice_idx < atlas_volume.shape[0]):
            results[slice_idx] = float("-inf")  # Or skip/warn
            continue

        atlas_slice = atlas_volume[slice_idx, :, :]

        # Let metric functions handle specific cases like constant images or
        # NaNs.
        # The decorator/metric should return -inf if appropriate.
        pass

        try:
            # Pass images directly; metric functions handle size matching
            # internally
            results[slice_idx] = metric_func(moving_image, atlas_slice)
        except Exception as e:
            # Handle any errors that might occur during metric calculation
            print(f"Error processing slice {slice_idx}: {e}")
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
        search_range = (0, atlas_volume.shape[0])

    # Create a range with the specified step size
    slice_indices = range(search_range[0], search_range[1], search_step)

    # Compare the image with the specified slices using the generated indices
    similarities = compare_image_to_atlas_slices(
        moving_image,
        atlas_volume,
        slice_range=slice_indices,  # Pass the iterable directly
        metric=metric,
    )

    # Check if similarities dictionary is empty
    if not similarities:
        raise ValueError(
            "No slices were compared, possibly due to invalid range or step."
        )

    # Find the slice index with the highest similarity
    # Filter out -inf values before finding the max
    valid_similarities = {
        k: v for k, v in similarities.items() if v != float("-inf")
    }
    if not valid_similarities:
        # Handle case where all compared slices resulted in -inf
        # Optionally, could return the first index of slice_indices
        # or raise error
        warnings.warn(
            "All compared slices yielded invalid similarity values.",
            UserWarning,
        )
        # Optionally, could return the first index of slice_indices
        # or raise error
        return slice_indices[0] if len(slice_indices) > 0 else -1

    # Find the key corresponding to the maximum value
    # Mypy struggles with dict.get, use explicit lambda
    best_slice_idx = max(
        valid_similarities, key=lambda k: valid_similarities[k]
    )

    return best_slice_idx
