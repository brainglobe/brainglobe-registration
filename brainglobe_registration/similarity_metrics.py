from typing import Callable, Dict, Iterable, Optional, Tuple

import numpy as np
import numpy.typing as npt
import skimage.metrics


def _match_image_sizes(
    img1: npt.NDArray, img2: npt.NDArray
) -> Tuple[Optional[npt.NDArray], Optional[npt.NDArray]]:
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
    Tuple[Optional[npt.NDArray], Optional[npt.NDArray]]
        Tuple of (cropped_img1, cropped_img2).
        If img1 or img2 has NaNs, first or second element is None.
        If both are valid, both are returned as cropped arrays.
    """
    min_shape = np.minimum(img1.shape, img2.shape)
    img1_crop = img1[: min_shape[0], : min_shape[1]]
    img2_crop = img2[: min_shape[0], : min_shape[1]]

    img1_proc = None if np.isnan(img1_crop).any() else img1_crop
    img2_proc = None if np.isnan(img2_crop).any() else img2_crop

    return img1_proc, img2_proc


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
    img1_proc, img2_proc = _match_image_sizes(img1, img2)
    if img1_proc is None or img2_proc is None:
        raise ValueError("Input images contain NaN values.")
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
    img1_proc, img2_proc = _match_image_sizes(img1, img2)
    if img1_proc is None or img2_proc is None:
        raise ValueError("Input images contain NaN values.")
    # Check for constant images BEFORE normalization
    is_img1_constant = np.all(img1_proc == img1_proc.flat[0])
    is_img2_constant = np.all(img2_proc == img2_proc.flat[0])

    if is_img1_constant and is_img2_constant:
        # Both are constant images
        return 2.0 if img1_proc.flat[0] == img2_proc.flat[0] else 1.0
    elif is_img1_constant or is_img2_constant:
        # If only one image is constant, MI is 0
        return 1.0

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
            raise ValueError("Mutual information calculation resulted in NaN.")

        return mi_value
    except (ValueError, RuntimeError, TypeError) as e:
        # If any error occurs during calculation, print warning and return -inf
        raise RuntimeError(f"Error calculating mutual information: {e}")


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
    img1_proc, img2_proc = _match_image_sizes(img1, img2)
    if img1_proc is None or img2_proc is None:
        raise ValueError("Input images contain NaN values.")
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
            raise ValueError("SSIM calculation resulted in NaN.")

        return ssim_value
    except (ValueError, RuntimeError, TypeError) as e:
        # If any error occurs during calculation, print warning and return -inf
        raise RuntimeError(
            f"Error calculating structural similarity index: {e}"
        )


def compare_image_to_atlas_slices(
    moving_image: npt.NDArray,
    atlas_volume: npt.NDArray,
    slice_range: Optional[
        Tuple[int, int]
    ] = None,  # (start,end), exclusive of end
    slice_indices: Optional[Iterable[int]] = None,  # if not None, use this
    search_step: int = 1,
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
    slice_range : Optional[Tuple[int, int]], optional
        Tuple (start_slice, end_slice) defining a contiguous range
        (exclusive of end_slice), or an iterable of specific slice indices
        to compare.
    slice_indices : Optional[Iterable[int]], optional
        Iterable of specific slice indices to compare. If None, use
        slice_range.
    search_step: int, optional
        Step size for searching slices, by default 1
    metric : str, optional
        Similarity metric ("ncc", "mi", "ssim"), by default "mi"

    Returns
    -------
    Dict[int, float]
        Dictionary with slice indices as keys and similarity values as values
    """
    # Dictionary mapping metric names to functions
    MetricFunction = Callable[[npt.NDArray, npt.NDArray], float]
    metric_functions: dict[str, MetricFunction] = {
        "ncc": normalized_cross_correlation,
        "mi": mutual_information,
        "ssim": structural_similarity_index,
    }
    if slice_indices is not None:
        indices_to_process = slice_indices
    elif slice_range is not None:
        start_slice, end_slice = slice_range
        start_slice = max(0, start_slice)
        end_slice = min(atlas_volume.shape[0], end_slice)
        indices_to_process = range(start_slice, end_slice, search_step)
    else:
        indices_to_process = range(0, atlas_volume.shape[0], search_step)

    if metric not in metric_functions:
        raise ValueError(
            f"Invalid metric: {metric}. "
            f"Choose from {list(metric_functions.keys())}"
        )
    metric_func = metric_functions[metric]

    results = {}
    for slice_idx in indices_to_process:
        # Ensure index is within bounds
        if not (0 <= slice_idx < atlas_volume.shape[0]):
            raise IndexError(f"{slice_idx} is out of bounds for atlas volume")

        atlas_slice = atlas_volume[slice_idx, :, :]

        # Let metric functions handle specific cases like constant images or
        # NaNs.
        try:
            results[slice_idx] = metric_func(moving_image, atlas_slice)
        except (ValueError, RuntimeError, TypeError) as e:
            # Handle any errors that might occur during metric calculation
            raise RuntimeError(f"Error processing slice {slice_idx}: {e}")

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
    # Use compare_image_to_atlas_slices to get similarities
    similarity_results = compare_image_to_atlas_slices(
        moving_image,
        atlas_volume,
        slice_range=search_range,
        search_step=search_step,
        metric=metric,
    )

    # Check if similarity_results is empty (e.g., invalid range)
    if not similarity_results:
        raise ValueError(
            "No slices were compared; check your search_range and search_step."
        )

    # Find the slice index with the maximum similarity value
    best_slice_index = max(
        similarity_results,
        key=lambda k: similarity_results[k],
    )

    if similarity_results[best_slice_index] == float("-inf"):
        raise ValueError(
            "All compared slices yielded invalid similarity values."
        )

    return best_slice_index
