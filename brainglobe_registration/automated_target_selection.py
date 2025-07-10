import warnings
from pathlib import Path

import numpy as np
from bayes_opt import BayesianOptimization
from skimage.transform import rotate

from brainglobe_registration.elastix.register import (
    run_registration,
    transform_image,
)
from brainglobe_registration.similarity_metrics import (
    compute_similarity_metric,
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
    pitch,
    yaw,
    roll,
    z_slice,
    atlas_volume,
    sample,
    metric="mi",
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
    metric : str
        Similarity metric to use for comparison. One of:
        - "mi"       : Mutual Information
        - "ncc"      : Normalised Cross-Correlation
        - "ssim"     : Structural Similarity Index
        - "combined" : 0.7 * mi + 0.15 * ncc + 0.15 * ssim
        Defaults to "mi".

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

        # Run registration
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
        )
        if np.isnan(score):
            print("Warning: computed similarity score is NaN.")
            return -1.0
        return score

    except Exception as e:
        warnings.warn(f"Failed registration attempt: {e}")
        return -1.0


def similarity_only_objective(roll, target_slice, sample, metric="mi"):
    """
    Compute similarity score between rotated fixed slice and 2D sample image.
    Used during fine registration to optimise roll independently.

    Parameters
    ----------
    roll : float
        Rotation angle (in degrees) to apply to the atlas.
    target_slice : np.ndarray
        Slice from the rotated atlas volume.
    sample : np.ndarray
        2D image to be aligned.
    metric : str
        Similarity metric to use for comparison. One of:
        - "mi"       : Mutual Information
        - "ncc"      : Normalised Cross-Correlation
        - "ssim"     : Structural Similarity Index
        - "combined" : 0.7 * mi + 0.15 * ncc + 0.15 * ssim
        Defaults to "mi".

    Returns
    -------
    float
        Similarity score between rotated fixed slice and sample image.
    """
    rotated_slice = rotate(target_slice, roll)
    score = compute_similarity_metric(
        moving=sample,
        fixed=rotated_slice,
        metric=metric,
    )
    return score


def run_bayesian_generator(
    atlas_volume,
    sample,
    manual_z_range,
    pitch_bounds=(-5, 5),
    yaw_bounds=(-5, 5),
    roll_bounds=(-5, 5),
    init_points=5,
    n_iter=15,
    metric="mi",
):
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
    pitch_bounds, yaw_bounds, roll_bounds : tuple of float, optional
        Bounds for rotation angles (default: (-5, 5) degrees).
    init_points : int, optional
        Number of initial random points for Bayesian optimisation (default: 5).
    n_iter : int, optional
        Number of optimisation iterations (default: 15).
    metric : str
        Similarity metric to use for comparison. One of:
        - "mi"       : Mutual Information
        - "ncc"      : Normalised Cross-Correlation
        - "ssim"     : Structural Similarity Index
        - "combined" : 0.7 * mi + 0.15 * ncc + 0.15 * ssim
        Defaults to "mi".

    Returns
    -------
    tuple
        Optimal pitch, yaw, roll, z_slice parameters based on similarity score.

    Prints
    ------
    Optimal parameters and similarity score for alignment.
    """
    # Bounds in degrees and slice index
    pbounds = {
        "pitch": pitch_bounds,
        "yaw": yaw_bounds,
        "z_slice": manual_z_range,
    }

    optimizer = BayesianOptimization(
        f=lambda pitch, yaw, z_slice: registration_objective(
            pitch,
            yaw,
            0,
            z_slice,
            atlas_volume,
            sample,
            metric,
        ),
        pbounds=pbounds,
        verbose=2,
        random_state=42,
    )

    optimizer.maximize(init_points=init_points, n_iter=n_iter)
    for result in optimizer.res:
        current_params = result["params"]
        current_score = result["target"]
        yield {
            "pitch": round(current_params["pitch"], 2),
            "yaw": round(current_params["yaw"], 2),
            "z_slice": round(current_params["z_slice"]),
            "score": current_score,
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

    pbounds_roll = {
        "roll": roll_bounds,
    }

    opt_roll = BayesianOptimization(
        f=lambda roll: similarity_only_objective(
            roll,
            target_slice,
            sample,
            metric,
        ),
        pbounds=pbounds_roll,
        verbose=2,
        random_state=42,
    )

    opt_roll.maximize(init_points=init_points, n_iter=n_iter)
    for result in opt_roll.res:
        current_roll = result["params"]
        current_roll_score = result["target"]
        yield {
            "roll": round(current_roll["roll"], 2),
            "roll_score": current_roll_score,
        }

    best_roll = opt_roll.max["params"]
    best_roll_score = opt_roll.max["target"]

    roll = round(best_roll["roll"], 2)

    print(
        f"\n[Bayesian] Optimal result:"
        f"\nScore (without roll): {best_score:.4f}"
        f"\nScore (including roll): {best_roll_score:.4f}"
    )
    print(f"pitch: {pitch}, yaw: {yaw}, roll: {roll}, z_slice: {z_slice}")

    yield {
        "best_pitch": pitch,
        "best_yaw": yaw,
        "best_roll": roll,
        "best_z_slice": z_slice,
        "done": True,
    }
