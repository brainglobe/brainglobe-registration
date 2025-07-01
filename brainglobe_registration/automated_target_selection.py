import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
import tifffile as tiff
from bayes_opt import BayesianOptimization
from brainglobe_atlasapi import BrainGlobeAtlas
from skimage.metrics import structural_similarity
from skimage.transform import rescale
from sklearn.feature_selection import mutual_info_regression

from brainglobe_registration.elastix.register import (
    run_registration,
    transform_image,
)
from brainglobe_registration.utils.transforms import (
    create_rotation_matrix,
    rotate_volume,
)
from brainglobe_registration.utils.utils import (
    open_parameter_file,
)


def pad_to_match_shape(
    smaller: np.ndarray,
    target_shape: Tuple[int, int],
    mode: str,
    constant_values: int = 0,
):
    """
    Symmetrically pad a 2D NumPy array with zeros to match a target shape.

    Parameters
    ----------
    smaller : np.ndarray
        The 2D array to be padded. Must have shape (H, W).
    target_shape : tuple of int
        The desired shape to match, given as (target_height, target_width).
    mode : str
        Padding mode to use, as accepted by `np.pad`.
        (e.g., 'constant', 'edge', 'reflect', etc.).
    constant_values : int, optional
        The constant value to use if mode='constant'. Ignored for other modes.
        Default is 0.

    Returns
    -------
    np.ndarray
        The padded array with the specified target shape.
        If input array >= target shape in dimension X, no padding in X.
    """
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
    return padded


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
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0),
):
    """
    Scale the moving image to have resolution equal to the atlas.

    Parameters
    ----------
    moving_image : np.ndarray
        Image to be scaled.
    atlas_res : tuple
        Resolution of the atlas.
    scale : tuple of float
        Scale (z, y, x) moving image to match atlas.
        Defaults to (1.0, 1.0, 1.0).

    Returns
    -------
    np.ndarray
        Rescaled image with the same shape as `moving_image`, adjusted
        to match the target atlas resolution.

    Will show an error if the pixel sizes are less than or equal to 0.
    """

    x, y, z = scale[2], scale[1], scale[0]

    if x <= 0 or y <= 0 or z <= 0:
        raise ValueError("Pixel sizes must be greater than 0")

    x_factor = x / atlas_res[0]
    y_factor = y / atlas_res[1]
    rescale_factors: Tuple[float, ...] = (y_factor, x_factor)

    if moving_image.ndim == 3:
        z_factor = z / atlas_res[2]
        rescale_factors = (z_factor, *rescale_factors)

    scaled = rescale(
        moving_image,
        rescale_factors,
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
    if moving.shape != fixed.shape:
        if (
            moving.shape[0] < fixed.shape[0]
            or moving.shape[1] < fixed.shape[1]
        ):
            moving = pad_to_match_shape(moving, fixed.shape, "constant")
        elif (
            fixed.shape[0] < moving.shape[0]
            or fixed.shape[1] < moving.shape[1]
        ):
            fixed = pad_to_match_shape(fixed, moving.shape, "constant")

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


# ------------ OPTIMISATION OBJECTIVE ------------ #


def registration_objective(
    pitch,
    yaw,
    roll,
    z_slice,
    atlas_volume,
    sample,
):
    """
    Perform image registration with given rotation and slice parameters
    and return a similarity score.

    Parameters
    ----------
    pitch, yaw, roll : float
        Rotation angles (degrees) to apply to the atlas.
    z_slice : float
        Index of the atlas slice to use after rotation.
    atlas_volume : np.ndarray
        3D atlas volume used as the fixed reference.
    sample : np.ndarray
        2D moving image to be aligned to the atlas slice.

    Returns
    -------
    float
        Similarity score between registered images, or -1.0 on failure.

    Raises
    ------
    IndexError
        If z_slice is out of bounds for the atlas volume.
    """
    if z_slice < 0 or z_slice >= atlas_volume.shape[0]:
        raise IndexError(
            f"z_slice index {z_slice} is out of bounds for "
            f"atlas volume with shape {atlas_volume.shape}"
        )
    try:
        # Create rotation matrix
        rot_matrix, bounding_box = create_rotation_matrix(
            roll, yaw, pitch, img_shape=atlas_volume.shape
        )

        # Rotate atlas
        rotated_volume = rotate_volume(
            atlas_volume, atlas_volume.shape, rot_matrix, bounding_box
        )

        # Convert float z to int index and clip to bounds
        z_idx = int(np.clip(z_slice, 0, rotated_volume.shape[0] - 1))
        current_atlas_slice = rotated_volume[z_idx].compute()

        # RUN REGISTRATION #

        transform_type = "affine"

        file_path = (
            Path(__file__).parent
            / "parameters"
            / "brainglobe_registration"
            / f"{transform_type}.txt"
        )

        transform_params = open_parameter_file(file_path)

        # Force internal pixel type to float before wrapping in list
        transform_params["FixedInternalImagePixelType"] = ["float"]
        transform_params["MovingInternalImagePixelType"] = ["float"]

        transform_param_list = [(transform_type, transform_params)]

        moving, fixed = prepare_images(sample, current_atlas_slice)

        if sample.ndim == 2:
            atlas_image = fixed.astype(np.float32)
            moving_image = moving.astype(np.float32)

            for transform_selection in transform_param_list:
                # Can't use a short for internal pixels on 2D images
                fixed_pixel_type = transform_selection[1].get(
                    "FixedInternalImagePixelType", []
                )
                moving_pixel_type = transform_selection[1].get(
                    "MovingInternalImagePixelType", []
                )
                if "float" not in fixed_pixel_type:
                    print(
                        f"Can not use {fixed_pixel_type} "
                        f"for internal pixels on 2D images, switching to float"
                    )
                    transform_selection[1]["FixedInternalImagePixelType"] = [
                        "float"
                    ]
                if "float" not in moving_pixel_type:
                    print(
                        f"Can not use {moving_pixel_type} "
                        f"for internal pixels on 2D images, switching to float"
                    )
                    transform_selection[1]["MovingInternalImagePixelType"] = [
                        "float"
                    ]
        else:
            print(
                f"Error: expected 2D input for 'sample', but received array "
                f"with shape {sample.shape} (ndim={sample.ndim})."
            )
            return -1.0
            # Only dealing with 2D for now

        parameters = run_registration(
            atlas_image=atlas_image,
            moving_image=moving_image,
            parameter_lists=transform_param_list,
            output_directory=None,
            filter_images=False,
        )

        atlas_in_data_space = transform_image(atlas_image, parameters)

        # ---------------- #

        # Compute similarity with current slice
        score = compute_similarity_metric(
            moving=sample,
            fixed=atlas_in_data_space,
        )
        if np.isnan(score):
            print("Warning: computed similarity score is NaN.")
            return -1.0
        return score

    except Exception as e:
        warnings.warn(f"Failed registration attempt: {e}")
        return -1.0


def run_bayesian(atlas_volume, sample, manual_z_range):
    """
    Run Bayesian optimisation to align an atlas volume to a sample image.

    Uses the registration objective function to optimise pitch, yaw, roll,
    and z-slice for best similarity between the atlas and sample.

    Parameters
    ----------
    atlas_volume : np.ndarray
        3D atlas volume used as the reference.
    sample : np.ndarray
        2D image to be aligned to the atlas.
    manual_z_range : tuple of float
        Lower and upper bounds for z-slice selection.

    Returns
    -------
    None
        Prints the best registration parameters and similarity score.
    """
    # Bounds in degrees and slice index
    pbounds = {
        "pitch": (-5, 5),
        "yaw": (-5, 5),
        "roll": (-5, 5),
        "z_slice": manual_z_range,
    }

    optimizer = BayesianOptimization(
        f=lambda pitch, yaw, roll, z_slice: registration_objective(
            pitch,
            yaw,
            roll,
            z_slice,
            atlas_volume,
            sample,
        ),
        pbounds=pbounds,
        verbose=2,
        random_state=42,
    )

    optimizer.maximize(init_points=5, n_iter=15)

    best_params = optimizer.max["params"]
    best_score = optimizer.max["target"]

    pitch = round(best_params["pitch"], 2)
    yaw = round(best_params["yaw"], 2)
    roll = round(best_params["roll"], 2)
    z_slice = round(best_params["z_slice"])

    print(f"\n[Bayesian] Optimal result:\nScore: {best_score:.4f}")
    print(f"pitch: {pitch}, yaw: {yaw}, roll: {roll}, z_slice: {z_slice}")

    return pitch, yaw, roll, z_slice


def main():
    atlas_name = "allen_mouse_100um"
    atlas = BrainGlobeAtlas(atlas_name)
    atlas_volume = atlas.reference
    atlas_res = atlas.resolution  # (z, y, x)

    sample = tiff.imread("resources/sample_hipp.tif")

    # Scale moving image to match atlas resolution
    my_scale = [25.0, 25.0, 25.0]  # z, y, x
    scaled_sample = scale_moving_image(
        moving_image=sample, atlas_res=atlas_res, scale=my_scale
    )

    manual_z_range = (50, 90)  # let this be gui input

    run_bayesian(atlas_volume, scaled_sample, manual_z_range)


if __name__ == "__main__":
    main()
