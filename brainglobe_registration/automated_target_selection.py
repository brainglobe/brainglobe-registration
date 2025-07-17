import warnings
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
from bayes_opt import BayesianOptimization
from skimage.transform import rotate

from brainglobe_registration.elastix.register import (
    run_registration,
    transform_image,
)
from brainglobe_registration.similarity_metrics import (
    compute_similarity_metric,
    pad_to_match_shape,
    prepare_images,
)
from brainglobe_registration.utils.transforms import (
    create_rotation_matrix,
    rotate_volume,
)
from brainglobe_registration.utils.utils import (
    open_parameter_file,
)


def registration_objective(
    pitch: float,
    yaw: float,
    z_slice: float,
    atlas_volume: np.ndarray,
    sample: np.ndarray,
    metric: Literal["mi", "ncc", "ssim", "combined"] = "mi",
    weights: Tuple[float, float, float] = (0.7, 0.15, 0.15),
):
    """
    Compute a similarity score between a 2D sample image and a
    rotated slice from a 3D atlas volume.

    1. Applies 3D rotation (pitch, yaw, roll) to the atlas volume.
    2. Extracts the specified z-slice (can be fractional)
       from the rotated volume.
    3. Registers the extracted atlas slice to the 2D sample image.
    4. Computes a similarity score based on the selected metric.


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
            0, yaw, pitch, img_shape=atlas_volume.shape
        )

        # Rotate atlas
        rotated_volume = rotate_volume(
            atlas_volume, atlas_volume.shape, rot_matrix, bounding_box
        )

        # Convert float z to int index and clip to bounds
        z_idx = int(np.clip(z_slice, 0, rotated_volume.shape[0] - 1))
        current_atlas_slice = rotated_volume[z_idx].compute()

        # Run registration
        transform_type = "affine"
        file_path = (
            Path(__file__).parent
            / "parameters"
            / "brainglobe_registration"
            / f"automated_reg_{transform_type}.txt"
        )

        transform_params = open_parameter_file(file_path)

        transform_param_list = [(transform_type, transform_params)]

        moving, fixed = prepare_images(sample, current_atlas_slice)

        if sample.ndim == 2:
            atlas_image = fixed.astype(np.float32)
            moving_image = moving.astype(np.float32)
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

        # Compute similarity with current slice
        score = compute_similarity_metric(
            moving=sample,
            fixed=atlas_in_data_space,
            metric=metric,
            weights=weights,
        )
        if np.isnan(score):
            print("Warning: computed similarity score is NaN.")
            return -1.0
        return score

    except Exception as e:
        warnings.warn(f"Failed registration attempt: {e}")
        return -1.0


def similarity_only_objective(
    roll: float,
    target_slice: np.ndarray,
    sample: np.ndarray,
    metric: Literal["mi", "ncc", "ssim", "combined"] = "mi",
    weights: Tuple[float, float, float] = (0.7, 0.15, 0.15),
):
    """
    Compute similarity score between rotated fixed slice and 2D sample image.
    Used to optimise roll independently.

    Parameters
    ----------
    roll : float
        Rotation angle (in degrees) to apply to the atlas.
    target_slice : np.ndarray
        Slice from the rotated atlas volume.
    sample : np.ndarray
        2D image to be aligned.
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
        Similarity score between rotated fixed slice and sample image.
    """
    rotated_slice = rotate(target_slice, roll, clip=False)
    sample_padded, rotated_slice_padded = pad_to_match_shape(
        sample, rotated_slice, mode="constant", constant_values=0
    )

    score = compute_similarity_metric(
        moving=sample_padded,
        fixed=rotated_slice_padded,
        metric=metric,
        weights=weights,
    )
    return score


def run_bayesian_generator(
    atlas_volume: np.ndarray,
    sample: np.ndarray,
    manual_z_range: Optional[Tuple[float, float]] = None,
    pitch_bounds: Tuple[float, float] = (-5, 5),
    yaw_bounds: Tuple[float, float] = (-5, 5),
    roll_bounds: Tuple[float, float] = (-5, 5),
    init_points: int = 5,
    n_iter: int = 15,
    metric: Literal["mi", "ncc", "ssim", "combined"] = "mi",
    weights: Tuple[float, float, float] = (0.7, 0.15, 0.15),
):
    """
    Run Bayesian optimisation to estimate the position of the
    sample image within the atlas volume.
    Uses the registration objective function to optimise pitch, yaw, roll,
    and z-slice for best similarity between the atlas and sample.

    Parameters
    ----------
    atlas_volume : np.ndarray
        3D atlas volume used as the reference.
    sample : np.ndarray
        2D image to be aligned to the atlas.
    manual_z_range : Tuple[float, float], optional
        Lower and upper bounds for z-slice selection.
        Defaults to None i.e. entire range.
    pitch_bounds, yaw_bounds, roll_bounds : Tuple[float, float], optional
        Bounds for rotation angles (default: (-5, 5) degrees).
    init_points : int, optional
        Number of initial random points for Bayesian optimisation (default: 5).
    n_iter : int, optional
        Number of optimisation iterations (default: 15).
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
    dict
        Dictionary containing the optimal alignment parameters:
        - "best_pitch" : float
            Optimal pitch angle (in degrees).
        - "best_yaw" : float
            Optimal yaw angle (in degrees).
        - "best_roll" : float
            Optimal roll angle (in degrees).
        - "best_z_slice" : int
            Optimal z-slice index within the rotated atlas.
        - "done" : bool
            Indicates that optimisation is complete (always True).

    Prints
    ------
    Optimal parameters and similarity score for alignment.
    """

    if manual_z_range is None:
        manual_z_range = (0, atlas_volume.shape[0] - 1)

    pbounds = {
        "pitch": pitch_bounds,
        "yaw": yaw_bounds,
        "z_slice": manual_z_range,
    }

    optimizer = BayesianOptimization(
        f=None,
        pbounds=pbounds,
        verbose=2,
        random_state=42,
        allow_duplicate_points=True,
    )
    # Customise Gaussian Progress
    optimizer.set_gp_params(alpha=1e-3, n_restarts_optimizer=5)

    # Initial random points
    for _ in range(init_points):
        point = optimizer.suggest()
        score = registration_objective(
            **point,
            atlas_volume=atlas_volume,
            sample=sample,
            metric=metric,
            weights=weights,
        )
        optimizer.register(params=point, target=score)

        yield {
            "stage": "coarse",
            "pitch": round(point["pitch"], 2),
            "yaw": round(point["yaw"], 2),
            "z_slice": round(point["z_slice"]),
            "score": score,
        }

    # Iterative Bayesian updates
    for _ in range(n_iter):
        point = optimizer.suggest()
        score = registration_objective(
            **point,
            atlas_volume=atlas_volume,
            sample=sample,
            metric=metric,
            weights=weights,
        )
        optimizer.register(params=point, target=score)

        yield {
            "stage": "coarse",
            "pitch": round(point["pitch"], 2),
            "yaw": round(point["yaw"], 2),
            "z_slice": round(point["z_slice"]),
            "score": score,
        }

    best_params = optimizer.max["params"]
    best_score = optimizer.max["target"]

    pitch = round(best_params["pitch"], 2)
    yaw = round(best_params["yaw"], 2)
    z_slice = round(best_params["z_slice"])

    rot_matrix, out_shape = create_rotation_matrix(
        0, yaw, pitch, atlas_volume.shape
    )
    transformed_atlas = rotate_volume(
        atlas_volume, atlas_volume.shape, rot_matrix, out_shape
    )
    target_slice = transformed_atlas[z_slice].compute()

    # Roll Optimisation
    opt_roll = BayesianOptimization(
        f=None,
        pbounds={"roll": roll_bounds},
        verbose=2,
        random_state=42,
        allow_duplicate_points=True,
    )
    opt_roll.set_gp_params(alpha=1e-3, n_restarts_optimizer=5)

    # Initial roll points
    for _ in range(init_points):
        point = opt_roll.suggest()
        score = similarity_only_objective(
            **point,
            target_slice=target_slice,
            sample=sample,
            metric=metric,
            weights=weights,
        )
        opt_roll.register(params=point, target=score)

        yield {
            "stage": "fine",
            "roll": round(point["roll"], 2),
            "roll_score": score,
        }

    # Iterative roll tuning
    for _ in range(n_iter):
        point = opt_roll.suggest()
        score = similarity_only_objective(
            **point,
            target_slice=target_slice,
            sample=sample,
            metric=metric,
            weights=weights,
        )
        opt_roll.register(params=point, target=score)

        yield {
            "stage": "fine",
            "roll": round(point["roll"], 2),
            "roll_score": score,
        }

    best_roll = round(opt_roll.max["params"]["roll"], 2)
    best_roll_score = opt_roll.max["target"]

    print(
        f"\n[Bayesian] Optimal result:"
        f"\nScore (without roll): {best_score:.4f}"
        f"\nScore (including roll): {best_roll_score:.4f}"
    )
    print(f"pitch: {pitch}, yaw: {yaw}, roll: {best_roll}, z_slice: {z_slice}")

    return {
        "best_pitch": pitch,
        "best_yaw": yaw,
        "best_roll": best_roll,
        "best_z_slice": z_slice,
        "done": True,
    }
