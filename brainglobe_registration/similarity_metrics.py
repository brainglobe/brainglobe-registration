from typing import Tuple

import numpy as np
from skimage.metrics import structural_similarity
from skimage.transform import rescale
from sklearn.feature_selection import mutual_info_regression


def pad_to_match_shape(
    moving: np.ndarray,
    fixed: np.ndarray,
    mode: str,
    constant_values: int = 0,
):
    """
    Symmetrically pad smaller 2D array to match shape of larger one.

    Parameters
    ----------
    moving : np.ndarray
        Moving (sample) image.
    fixed : np.ndarray
        Atlas image.
    mode : str
        Padding mode (e.g., 'constant', 'edge', 'reflect', etc.).
    constant_values : int
        Constant value to use for 'constant' mode padding.
        Default = 0

    Returns
    -------
    np.ndarray
        The moving image.
    np.ndarray
        The fixed image.
    """

    if moving.shape != fixed.shape:
        if (
            moving.shape[0] < fixed.shape[0]
            or moving.shape[1] < fixed.shape[1]
        ):
            pad_moving = True
            smaller = moving
            target_shape = fixed.shape
        elif (
            fixed.shape[0] < moving.shape[0]
            or fixed.shape[1] < moving.shape[1]
        ):
            smaller = fixed
            target_shape = moving.shape

        pad_height = max(0, target_shape[0] - smaller.shape[0])
        pad_width = max(0, target_shape[1] - smaller.shape[1])

        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left

        padded = np.pad(
            smaller,
            pad_width=((pad_top, pad_bottom), (pad_left, pad_right)),
            mode=mode,
            constant_values=constant_values,
        )

        if pad_moving:
            return padded, fixed
        else:
            return moving, padded
    else:
        return moving, fixed


def normalise_image(img):
    """
    Normalise a NumPy array to the range [0, 1].

    Parameters
    ----------
    img : np.ndarray
        Input image array.

    Returns
    -------
    np.ndarray
        Normalised image with values in [0, 1].
    """
    img = img.astype(np.float32)
    img_min = img.min()
    img_max = img.max()
    return (img - img_min) / (img_max - img_min + 1e-8)


def scale_moving_image(
    moving_image: np.ndarray,
    atlas_res: Tuple,
    moving_res: Tuple[float, float, float] = (1.0, 1.0, 1.0),
):
    """
    Scale the moving image to have resolution equal to the atlas.

    Parameters
    ----------
    moving_image : np.ndarray
        Image to be scaled.
    atlas_res : tuple
        Resolution (z, y, x) of the atlas.
    moving_res : tuple of float
        Resolution (z, y, x) of moving image.
        Defaults to (1.0, 1.0, 1.0).

    Returns
    -------
    np.ndarray
        Rescaled image with the same shape as `moving_image`, adjusted
        to match the target atlas resolution.

    Will show an error if the pixel sizes are less than or equal to 0.
    """

    x, y, z = moving_res[2], moving_res[1], moving_res[0]

    if x <= 0 or y <= 0 or z <= 0:
        raise ValueError("Pixel sizes must be greater than 0")

    x_factor = x / atlas_res[2]
    y_factor = y / atlas_res[1]
    scale: Tuple[float, ...] = (y_factor, x_factor)

    if moving_image.ndim == 3:
        z_factor = z / atlas_res[0]
        scale = (z_factor, *scale)

    scaled = rescale(
        moving_image,
        scale,
        mode="constant",
        preserve_range=True,
        anti_aliasing=True,
    ).astype(moving_image.dtype)

    return scaled


def prepare_images(moving: np.ndarray, fixed: np.ndarray):
    """
    Pad and normalise moving and fixed images for registration.

    Parameters
    ----------
    moving : np.ndarray
        Moving image to be aligned.
    fixed : np.ndarray
        Fixed reference image.

    Returns
    -------
    moving : np.ndarray
        The preprocessed moving image, scaled, padded, and normalised.
    fixed : np.ndarray
        The preprocessed fixed image, padded and normalised.
    """

    # Match shape by padding the smaller image
    moving, fixed = pad_to_match_shape(moving, fixed, "constant")

    # Normalise
    moving = normalise_image(moving)
    fixed = normalise_image(fixed)

    return moving, fixed


def safe_ncc(img1, img2):
    """
    Compute Normalised Cross-Correlation (NCC) between two images.
    Returns 0.0 if either image has zero standard deviation.

    Parameters
    ----------
    img1 : np.ndarray
        First image.
    img2 : np.ndarray
        Second image.

    Returns
    -------
    float
        NCC score between -1.0 and 1.0.
    """
    if img1.shape != img2.shape:
        raise ValueError(
            f"Input shapes must match. Got {img1.shape} and {img2.shape}"
        )
    if np.std(img1) == 0 or np.std(img2) == 0:
        return 0.0  # return a very low score if there's no signal
    return np.corrcoef(img1.ravel(), img2.ravel())[0, 1]


def compute_similarity_metric(moving, fixed, metric="mi"):
    """
    Compute similarity between two images using SSIM, NCC, or MI.

    Parameters
    ----------
    moving : np.ndarray
        Moving image.
    fixed : np.ndarray
        Fixed reference image.
    metric: str
        Similarity metric to use for comparison. One of:
        - "mi"       : Mutual Information
        - "ncc"      : Normalised Cross-Correlation
        - "ssim"     : Structural Similarity Index
        - "combined" : 0.7 * mi + 0.15 * ncc + 0.15 * ssim
        Defaults to "mi".

    Returns
    -------
    float
        Combined similarity score.
    """
    moving, fixed = prepare_images(moving, fixed)

    # Similarity metrics
    ncc = safe_ncc(moving, fixed)
    ssim = structural_similarity(moving, fixed, data_range=1.0)
    mi = mutual_info_regression(moving.ravel().reshape(-1, 1), fixed.ravel())[
        0
    ]

    if metric == "mi":
        return mi
    elif metric == "ncc":
        return ncc
    elif metric == "ssim":
        return ssim
    elif metric == "combined":
        return 0.7 * mi + 0.15 * ncc + 0.15 * ssim
    else:
        raise ValueError(
            f"Unsupported metric '{metric}'. "
            f"Choose from 'mi', 'ncc', 'ssim', 'combined'."
        )
