import warnings
from itertools import product
from typing import Tuple

import dask.array as da
import dask_image.ndinterp as ndi
import numpy as np
import tifffile as tiff
from bayes_opt import BayesianOptimization
from brainglobe_atlasapi import BrainGlobeAtlas
from brainglobe_atlasapi.list_atlases import get_downloaded_atlases
from napari.utils.notifications import show_error
from pytransform3d.rotations import active_matrix_from_angle
from skimage.metrics import structural_similarity
from skimage.transform import rescale

from brainglobe_registration.utils.utils import (
    calculate_rotated_bounding_box,
)


def pad_to_match_shape(smaller, target_shape):
    """Pad a 2D array symmetrically to match target shape."""
    pad_height = max(0, target_shape[0] - smaller.shape[0])
    pad_width = max(0, target_shape[1] - smaller.shape[1])

    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    padded = np.pad(
        smaller,
        pad_width=((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
        constant_values=0,  # or 'reflect', 'edge', etc.
    )
    return padded


def normalise_image(img):
    img = img.astype(np.float32)
    return (img - img.min()) / (img.max() - img.min() + 1e-8)


def scale_moving_image(
    moving_image, atlas_res, x: float = 1.0, y: float = 1.0, z: float = 1.0
):
    """
    Scale the moving image to have resolution equal to the atlas.

    Parameters
    ------------
    x : float
        Moving image x pixel size (> 0.0).
    y : float
        Moving image y pixel size (> 0.0).
    z : float
        Moving image z pixel size (> 0.0).

    Will show an error if the pixel sizes are less than or equal to 0.
    """
    if x <= 0 or y <= 0 or z <= 0:
        show_error("Pixel sizes must be greater than 0")
        return

    x_factor = x / atlas_res[0]
    y_factor = y / atlas_res[1]
    scale: Tuple[float, ...] = (y_factor, x_factor)

    if moving_image.ndim == 3:
        z_factor = z / atlas_res[2]
        scale = (z_factor, *scale)

    scaled = rescale(
        moving_image,
        scale,
        mode="constant",
        preserve_range=True,
        anti_aliasing=True,
    ).astype(moving_image.dtype)

    return scaled


def safe_ncc(img1, img2):
    if np.std(img1) == 0 or np.std(img2) == 0:
        return 0.0  # return a very low score if there's no signal
    return np.corrcoef(img1.ravel(), img2.ravel())[0, 1]


def compute_similarity_metric(moving, fixed, atlas_res=(1, 1, 1)):
    # Scale moving image to match atlas resolution
    moving = scale_moving_image(moving, atlas_res, x=23.0, y=23.0, z=23.0)

    # Match shape by padding the smaller image
    if moving.shape != fixed.shape:
        if (
            moving.shape[0] < fixed.shape[0]
            or moving.shape[1] < fixed.shape[1]
        ):
            moving = pad_to_match_shape(moving, fixed.shape)
        elif (
            fixed.shape[0] < moving.shape[0]
            or fixed.shape[1] < moving.shape[1]
        ):
            fixed = pad_to_match_shape(fixed, moving.shape)

    # Normalise
    moving = normalise_image(moving)
    fixed = normalise_image(fixed)

    # Similarity metrics
    ncc = safe_ncc(moving, fixed)
    ssim = structural_similarity(moving, fixed, data_range=1.0)
    # mi = mutual_info_regression(moving.ravel().reshape(-1, 1),
    #                            fixed.ravel())[0]

    # combined = 0.4 * ssim + 0.3 * ncc + 0.3 * mi
    combined = 0.6 * ssim + 0.4 * ncc
    return combined


def create_rotation_matrix(roll: float, yaw: float, pitch: float):
    """Create a combined 3D rotation matrix from roll, yaw,
    and pitch (in degrees)."""
    # Create the rotation matrix
    roll_matrix = active_matrix_from_angle(0, np.deg2rad(roll))
    yaw_matrix = active_matrix_from_angle(1, np.deg2rad(yaw))
    pitch_matrix = active_matrix_from_angle(2, np.deg2rad(pitch))

    # Combine rotation matrices
    rotation_matrix = yaw_matrix @ pitch_matrix @ roll_matrix

    full_matrix = np.eye(4)
    full_matrix[:3, :3] = rotation_matrix
    return full_matrix


def rotate_volume(
    data: np.ndarray,
    rotation_matrix: np.ndarray,
    reference_shape: Tuple[int, int, int],
    interpolation_order: int = 2,
):
    """Rotate a 3D volume using a given rotation matrix and return
    transformed data and transform matrix."""

    # Translate the origin to the center of the image
    origin = np.asarray(reference_shape) / 2

    translate_to_center = np.eye(4)
    translate_to_center[:3, -1] = -origin

    bounding_box = calculate_rotated_bounding_box(
        reference_shape, rotation_matrix
    )
    new_translation = np.asarray(bounding_box) / 2

    post_rotate_translation = np.eye(4)
    post_rotate_translation[:3, -1] = new_translation

    # Combine the matrices. The order of operations is:
    # 1. Translate the origin to the center of the image
    # 2. Rotate the image
    # 3. Translate the origin back to the top left corner

    final_transform = np.linalg.inv(
        post_rotate_translation @ rotation_matrix @ translate_to_center
    )

    transformed = ndi.affine_transform(
        da.from_array(
            data, chunks=(2, reference_shape[1], reference_shape[2])
        ),
        matrix=final_transform,
        output_shape=bounding_box,
        order=interpolation_order,
    ).astype(data.dtype)

    return transformed, final_transform, bounding_box


# Optimisation objective
# ----------------------------------------
# Load data
# ----------------------------------------
atlas_name = (get_downloaded_atlases())[0]  # e.g. Allen 100um
atlas = BrainGlobeAtlas(atlas_name)
atlas_volume = atlas.reference

# FOR TESTING
# atlas_volume = atlas_volume[68:79, :, :]

atlas_res = atlas.resolution  # (z, y, x)

sample = tiff.imread("resources/sample_hipp.tif")


# ----------------------------------------
# Optimisation Objective
# ----------------------------------------
def registration_objective(pitch, yaw, roll, z_slice):
    try:
        # Create rotation matrix
        rot_matrix = create_rotation_matrix(roll, yaw, pitch)

        # Rotate atlas
        rotated_volume, _, _ = rotate_volume(
            atlas_volume, rot_matrix, reference_shape=atlas_volume.shape
        )

        # Convert float z to int index and clip to bounds
        z_idx = int(np.clip(z_slice, 0, rotated_volume.shape[0] - 1))
        atlas_slice = rotated_volume[z_idx].compute()

        # Compute similarity with current slice
        score = compute_similarity_metric(
            moving=sample, fixed=atlas_slice, atlas_res=atlas_res
        )
        return 0.0 if np.isnan(score) else score

    except Exception as e:
        warnings.warn(f"Failed registration attempt: {e}")
        return -1.0  # Penalize bad transformations


# ----------------------------------------
# Bayesian Optimisation
# ----------------------------------------

# Bounds in degrees and slice index
pbounds = {
    "pitch": (-5, 5),
    "yaw": (-5, 5),
    "roll": (-5, 5),
    "z_slice": (1, 130),
}

optimizer = BayesianOptimization(
    f=registration_objective,
    pbounds=pbounds,
    verbose=2,
    random_state=42,
)

optimizer.maximize(
    init_points=10,
    n_iter=30,
)

# Extract best parameters
best_params_bayes = optimizer.max["params"]
print(
    f"\n[Bayesian] Optimal result:\n" f"Score: {optimizer.max['target']:.4f}"
)
for k, v in best_params_bayes.items():
    print(f"{k}: {v:.2f}")


# ----------------------------------------
# Adaptive Grid Search
# ----------------------------------------


def adaptive_grid_search(
    objective_fn, bounds, initial_step=10, refinement_steps=[5, 2, 1]
):
    param_names = list(bounds.keys())
    best_params = None
    best_score = -np.inf
    current_center = {k: np.mean(v) for k, v in bounds.items()}
    current_range = {k: (v[1] - v[0]) / 2 for k, v in bounds.items()}

    for step in [initial_step] + refinement_steps:
        search_space = {
            k: np.arange(
                max(bounds[k][0], current_center[k] - current_range[k]),
                min(bounds[k][1], current_center[k] + current_range[k]) + 1e-3,
                step,
            )
            for k in param_names
        }

        print(f"\n[INFO] Grid search with step size {step}")
        for values in product(*search_space.values()):
            params = dict(zip(param_names, values))
            score = objective_fn(**params)
            if not np.isnan(score) and score > best_score:
                best_score = score
                best_params = params

        current_center = best_params
        current_range = {k: step for k in param_names}

    return best_params, best_score


# ----------------------------------------
# Run Grid Search (Toggle with flag)
# ----------------------------------------

run_grid_search = True

if run_grid_search:
    best_params_grid, best_score_grid = adaptive_grid_search(
        objective_fn=registration_objective,
        bounds=pbounds,
        initial_step=10,
        refinement_steps=[5, 2, 1],
    )

    print(f"\n[Grid Search] Optimal result:\n" f"Score: {best_score_grid:.4f}")
    for k, v in best_params_grid.items():
        print(f"{k}: {v:.2f}")

    if best_score_grid > optimizer.max["target"]:
        print("\nGrid search found a better result.")
    else:
        print("\nBayesian optimisation had the better result.")
