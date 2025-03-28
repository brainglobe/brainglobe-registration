from typing import Optional, Tuple

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
    # constant I guess)
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

    # Handle NaN values by replacing them with zeros
    img1_crop = np.nan_to_num(img1_crop)
    img2_crop = np.nan_to_num(img2_crop)

    # Check for constant images
    img1_is_constant = np.all(img1_crop == img1_crop.flat[0])
    img2_is_constant = np.all(img2_crop == img2_crop.flat[0])

    # Special case: both images are constant
    if img1_is_constant and img2_is_constant:
        # If both constant images have the same value, they have perfect mutual
        # information
        if img1_crop.flat[0] == img2_crop.flat[0]:
            return 1.0  # Return positive value for identical constant images
        else:
            return 0.0  # Different constant values have no mutual information

    # If only one image is constant, there's no mutual information
    if img1_is_constant or img2_is_constant:
        return 0.0

    try:
        # Normalize images to [0,1] for histogram calculation
        if np.max(img1_crop) != 0:
            img1_norm = img1_crop.astype(np.float32) / np.max(img1_crop)
        else:
            img1_norm = img1_crop.astype(np.float32)

        if np.max(img2_crop) != 0:
            img2_norm = img2_crop.astype(np.float32) / np.max(img2_crop)
        else:
            img2_norm = img2_crop.astype(np.float32)

        # Calculate mutual information using scikit-image
        mi_value = skimage.metrics.mutual_information_2d(
            img1_norm, img2_norm, bins=bins
        )

        # Handle potential NaN results
        if np.isnan(mi_value):
            return 0.0

        return mi_value
    except Exception:
        # If any error occurs during calculation, return 0
        return 0.0


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
    except Exception:
        # If any error occurs during calculation, return 0
        return 0.0


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
