from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
from skimage.metrics import (
    normalized_mutual_information,
    structural_similarity,
)


def normalized_cross_correlation(
    image1: npt.NDArray, image2: npt.NDArray
) -> float:
    """
    Calculate normalized cross-correlation between two images.

    Parameters
    ----------
    image1 : npt.NDArray
        First input image
    image2 : npt.NDArray
        Second input image

    Returns
    -------
    float
        Normalized cross-correlation value
        Higher values indicate better similarity
    """
    # Ensure the images have the same shape
    min_shape = np.minimum(image1.shape, image2.shape)
    img1 = image1[: min_shape[0], : min_shape[1]].astype(np.float32)
    img2 = image2[: min_shape[0], : min_shape[1]].astype(np.float32)

    # Handle empty or constant images
    if np.std(img1) < 1e-10 or np.std(img2) < 1e-10:
        return 0.0

    # Normalize the images
    img1_norm = (img1 - np.mean(img1)) / np.std(img1)
    img2_norm = (img2 - np.mean(img2)) / np.std(img2)

    # Calculate cross-correlation
    correlation = np.sum(img1_norm * img2_norm) / (img1_norm.size)

    return correlation


def mutual_information(
    image1: npt.NDArray, image2: npt.NDArray, bins: int = 32
) -> float:
    """
    Calculate mutual information between two images.

    Parameters
    ----------
    image1 : npt.NDArray
        First input image
    image2 : npt.NDArray
        Second input image
    bins : int, optional
        Number of bins for histogram calculation, by default 32

    Returns
    -------
    float
        Mutual information value
        Higher values indicate better similarity
    """
    # Normalize images to have same intensity range for histogram calculation
    img1_norm = image1.astype(np.float32) / np.max(image1)
    img2_norm = image2.astype(np.float32) / np.max(image2)

    # Need to ensure that the images have the same shape
    min_shape = np.minimum(img1_norm.shape, img2_norm.shape)
    img1_norm = img1_norm[: min_shape[0], : min_shape[1]]
    img2_norm = img2_norm[: min_shape[0], : min_shape[1]]

    return normalized_mutual_information(img1_norm, img2_norm, bins=bins)


def structural_similarity_index(
    image1: npt.NDArray, image2: npt.NDArray
) -> float:
    """
    Calculate structural similarity index (SSIM) between two images.

    Parameters
    ----------
    image1 : npt.NDArray
        First input image
    image2 : npt.NDArray
        Second input image

    Returns
    -------
    float
        SSIM value between -1 and 1
        Higher values indicate better similarity
    """
    # Need to ensure that the images have the same shape
    min_shape = np.minimum(image1.shape, image2.shape)
    img1_crop = image1[: min_shape[0], : min_shape[1]]
    img2_crop = image2[: min_shape[0], : min_shape[1]]

    # Normalize images to 0-1 range
    img1_norm = img1_crop.astype(np.float32) / np.max(img1_crop)
    img2_norm = img2_crop.astype(np.float32) / np.max(img2_crop)

    return structural_similarity(img1_norm, img2_norm, data_range=1.0)


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
        # "elastix": elastix_metric(image1, image2)
    }


def compare_image_to_atlas_slices(
    moving_image: npt.NDArray,
    atlas_volume: npt.NDArray,
    slice_range: Optional[Tuple[int, int]] = None,
    metric: str = "mi",
) -> dict:
    """
    Compare an input image with multiple atlas slices.

    Parameters
    ----------
    moving_image : npt.NDArray
        Input image to compare
    atlas_volume : npt.NDArray
        3D atlas volume
    slice_range : tuple, optional
        Range of slices to compare (start, end), by default None
        If None, compares with all slices
    metric : str, optional
        Metric to use for comparison, by default "mi"
        Options: "ncc", "mi", "ssim", "elastix"

    Returns
    -------
    dict
        Dictionary with slice indices as keys and similarity values as values
    """
    if slice_range is None:
        slice_range = (0, atlas_volume.shape[0])

    # Select the appropriate metric function
    if metric == "ncc":
        metric_func = normalized_cross_correlation
    elif metric == "mi":
        metric_func = mutual_information
    elif metric == "ssim":
        metric_func = structural_similarity_index
    # elif metric == "elastix":
    #    metric_func = elastix_metric
    else:
        raise ValueError(f"Unknown metric: {metric}")

    results = {}
    for slice_idx in range(slice_range[0], slice_range[1]):
        atlas_slice = atlas_volume[slice_idx, :, :]

        # Check if the atlas slice contains NaN values or is empty
        if np.isnan(atlas_slice).any() or np.all(atlas_slice == 0):
            # Skip this slice or assign a very low similarity
            results[slice_idx] = float("-inf")  # Indicates this is a bad match
            continue

        try:
            # Make sure dimensions match before passing to metric function
            min_height = min(moving_image.shape[0], atlas_slice.shape[0])
            min_width = min(moving_image.shape[1], atlas_slice.shape[1])
            cropped_moving = moving_image[:min_height, :min_width]
            cropped_atlas = atlas_slice[:min_height, :min_width]

            similarity = metric_func(cropped_atlas, cropped_moving)
            results[slice_idx] = similarity
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
        Options: "ncc", "mi", "ssim", "elastix"
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

    # Compare the image with the specified slices
    similarities = compare_image_to_atlas_slices(
        moving_image,
        atlas_volume,
        slice_range=(slice_indices[0], slice_indices[-1] + 1),
        metric=metric,
    )

    # Find the slice with the highest similarity
    best_slice = max(similarities.items(), key=lambda x: x[1])[0]

    return best_slice
