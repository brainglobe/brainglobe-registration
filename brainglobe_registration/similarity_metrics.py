import warnings
from typing import Literal, Tuple

import numpy as np
from skimage.metrics import structural_similarity
from sklearn.feature_selection import mutual_info_regression


def pad_to_match_shape(
    moving: np.ndarray,
    fixed: np.ndarray,
    mode: str,
    constant_values: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Symmetrically pad both 2D arrays to match the
    largest shape along each axis.

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
        Padded moving image.
    np.ndarray
        Padded fixed image.
    """

    target_height = max(moving.shape[0], fixed.shape[0])
    target_width = max(moving.shape[1], fixed.shape[1])

    def pad_to_shape(img, target_shape):
        pad_height = target_shape[0] - img.shape[0]
        pad_width = target_shape[1] - img.shape[1]

        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left

        kwargs = {"mode": mode}
        if mode == "constant":
            kwargs["constant_values"] = constant_values

        return np.pad(
            img,
            pad_width=((pad_top, pad_bottom), (pad_left, pad_right)),
            **kwargs,
        )

    moving_padded = pad_to_shape(moving, (target_height, target_width))
    fixed_padded = pad_to_shape(fixed, (target_height, target_width))

    return moving_padded, fixed_padded


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
        warnings.warn(
            "One or both input images are constant. NCC is undefined."
        )
        return 0.0  # return very low score if either image is constant.
    return np.corrcoef(img1.ravel(), img2.ravel())[0, 1]


def compute_similarity_metric(
    moving: np.ndarray,
    fixed: np.ndarray,
    metric: Literal["mi", "ncc", "ssim", "combined"] = "mi",
    weights: Tuple[float, float, float] = (0.7, 0.15, 0.15),
):
    """
    Compute similarity between two images using SSIM, NCC, MI, or combined.

    Parameters
    ----------
    moving : np.ndarray
        Moving image.
    fixed : np.ndarray
        Fixed reference image.
    metric : Literal["mi", "ncc", "ssim", "combined"], optional
        Similarity metric used to compare the sample and atlas slice:
        - "mi"       : Mutual Information
        - "ncc"      : Normalised Cross-Correlation
        - "ssim"     : Structural Similarity Index
        - "combined" : weights[0]*MI + weights[1]*NCC + weights[2]*SSIM
        Defaults to "mi".
    weights : Tuple[float, float, float], optional
        3-tuple specifying weights for (MI, NCC, SSIM) in the combined metric.
        Only used if metric="combined". Must sum to 1.
        Defaults to (0.7, 0.15, 0.15).

    Returns
    -------
    float
        Similarity score for chosen metric.
    """
    moving, fixed = prepare_images(moving, fixed)

    if metric == "mi":
        mi = mutual_info_regression(
            moving.ravel().reshape(-1, 1), fixed.ravel()
        )[0]
        return mi
    elif metric == "ncc":
        ncc = safe_ncc(moving, fixed)
        return ncc
    elif metric == "ssim":
        ssim = structural_similarity(moving, fixed, data_range=1.0)
        return ssim

    elif metric == "combined":
        if (
            not isinstance(weights, tuple)
            or len(weights) != 3
            or not all(isinstance(w, (int, float)) for w in weights)
        ):
            raise ValueError(
                f"Invalid weights: {weights}." f"Must be a 3-tuple of floats."
            )
        mi = mutual_info_regression(
            moving.ravel().reshape(-1, 1), fixed.ravel()
        )[0]
        ncc = safe_ncc(moving, fixed)
        ssim = structural_similarity(moving, fixed, data_range=1.0)

        return weights[0] * mi + weights[1] * ncc + weights[2] * ssim

    else:
        raise ValueError(
            f"Unsupported metric '{metric}'. "
            f"Choose from 'mi', 'ncc', 'ssim', 'combined'."
        )
