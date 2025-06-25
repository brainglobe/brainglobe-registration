import warnings
from pathlib import Path
from typing import Tuple

import dask.array as da
import dask_image.ndinterp as ndi
import numpy as np
import tifffile as tiff
from bayes_opt import BayesianOptimization
from brainglobe_atlasapi import BrainGlobeAtlas
from napari.utils.notifications import show_error
from pytransform3d.rotations import active_matrix_from_angle
from skimage.metrics import structural_similarity
from skimage.transform import rescale
from sklearn.feature_selection import mutual_info_regression

from brainglobe_registration.elastix.register import (
    run_registration,
    transform_image,
)
from brainglobe_registration.utils.utils import (
    calculate_rotated_bounding_box,
    open_parameter_file,
)


def pad_to_match_shape(smaller, target_shape):
    """
    Symmetrically pad a 2D NumPy array with zeros to match a target shape.

    Parameters
    ----------
    smaller : np.ndarray
        The 2D array to be padded. Must have shape (H, W).
    target_shape : tuple of int
        The desired shape to match, given as (target_height, target_width).

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
        mode="constant",
        constant_values=0,
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
    return (img - img.min()) / (img.max() - img.min() + 1e-8)


def scale_moving_image(moving_image, atlas_res, x: float, y: float, z: float):
    """
    Scale the moving image to have resolution equal to the atlas.

    Parameters
    ------------
    moving_image : tifffile.TiffImage
        Image to be scaled.
    atlas_res : tuple
        Resolution of the atlas.
    x : float
        Moving image x pixel size (> 0.0).
    y : float
        Moving image y pixel size (> 0.0).
    z : float
        Moving image z pixel size (> 0.0).

    Returns
    -------
    np.ndarray
        Rescaled image with the same shape as `moving_image`, adjusted
        to match the target atlas resolution.

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


def prepare_images(moving, fixed, atlas_res, scale=[1.0, 1.0, 1.0]):
    """
    Scale, pad, and normalise moving and fixed images for registration.

    Parameters
    ----------
    moving : np.ndarray
        Moving image to be aligned.
    fixed : np.ndarray
        Fixed reference image.
    atlas_res : list or tuple of float
        The resolution of the atlas.
    scale : list of float
        Scale (x, y, z) moving image to match atlas.

    Returns
    -------
    tuple of np.ndarray
        Preprocessed (moving, fixed) image pair.
    """
    # Scale moving image to match atlas resolution
    moving = scale_moving_image(
        moving, atlas_res, x=scale[0], y=scale[1], z=scale[2]
    )

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


def compute_similarity_metric(
    moving, fixed, atlas_res, atlas_to_moving_scale=[1.0, 1.0, 1.0]
):
    """
    Compute similarity between two images using SSIM, NCC, and MI.

    Parameters
    ----------
    moving : np.ndarray
        Moving image.
    fixed : np.ndarray
        Fixed reference image.
    atlas_res : list or tuple of float
        The resolution of the atlas.
    atlas_to_moving_scale : list of float, optional
        Scaling [x, y, z] applied to moving image to match atlas resolution.
        Defaults to [1.0, 1.0, 1.0].

    Returns
    -------
    float
        Combined similarity score.
    """
    moving, fixed = prepare_images(
        moving, fixed, atlas_res, scale=atlas_to_moving_scale
    )

    # Similarity metrics
    ncc = safe_ncc(moving, fixed)
    ssim = structural_similarity(moving, fixed, data_range=1.0)
    mi = mutual_info_regression(moving.ravel().reshape(-1, 1), fixed.ravel())[
        0
    ]

    combined = False
    if combined:
        final_metric = 0.55 * ssim + 0.35 * ncc + 0.1 * mi
    else:
        final_metric = mi
    return final_metric


def create_rotation_matrix(
    roll: float, yaw: float, pitch: float, img_shape: Tuple[int, int, int]
):
    """
    Creates a 3D affine transformation matrix from roll, yaw, and pitch angles.
    Builds a composite 4×4 rotation matrix from roll (X-axis), yaw (Y-axis),
    and pitch (Z-axis) angles in degrees. Rotation is applied about the centre
    of the input volume. Output includes a translation to fit rotated volume
    into a new bounding box.
    Parameters:
    ----------
    roll : float
        Rotation around the X-axis (in degrees).
    yaw : float
        Rotation around the Y-axis (in degrees).
    pitch : float
        Rotation around the Z-axis (in degrees).
    img_shape : Tuple[int, int, int]
        Shape of the original 3D image volume (Z, Y, X).
    Returns:
    -------
    final_transform : np.ndarray
        4×4 affine transformation matrix.
    bounding_box : Tuple[int, int, int]
        Shape of the rotated volume that fully contains the transformed data.
    """
    # Create the rotation matrix
    roll_matrix = active_matrix_from_angle(0, np.deg2rad(roll))
    yaw_matrix = active_matrix_from_angle(1, np.deg2rad(yaw))
    pitch_matrix = active_matrix_from_angle(2, np.deg2rad(pitch))

    # Combine rotation matrices
    rotation_matrix = yaw_matrix @ pitch_matrix @ roll_matrix

    full_matrix = np.eye(4)
    full_matrix[:3, :3] = rotation_matrix

    # Translate the origin to the center of the image
    origin = np.asarray(img_shape) / 2

    translate_to_center = np.eye(4)
    translate_to_center[:3, -1] = -origin

    bounding_box = calculate_rotated_bounding_box(img_shape, full_matrix)
    new_translation = np.asarray(bounding_box) / 2

    post_rotate_translation = np.eye(4)
    post_rotate_translation[:3, -1] = new_translation

    # Combine the matrices. The order of operations is:
    # 1. Translate the origin to the center of the image
    # 2. Rotate the image
    # 3. Translate the origin back to the top left corner

    final_transform = np.linalg.inv(
        post_rotate_translation @ full_matrix @ translate_to_center
    )

    return final_transform, bounding_box


def rotate_volume(
    data: np.ndarray,
    reference_shape: Tuple[int, int, int],
    final_transform: np.ndarray,
    bounding_box: Tuple[int, int, int],
    interpolation_order: int = 2,
) -> np.ndarray:
    """
    Apply a 3D affine transformation to a volume using a precomputed transform.
    Parameters
    ----------
    data : np.ndarray
        The 3D input volume (Z, Y, X) to be transformed.
    reference_shape : Tuple[int, int, int]
        Shape of the original reference volume.
    final_transform : np.ndarray
        4×4 affine transformation matrix to apply.
    bounding_box : Tuple[int, int, int]
        Shape of the output (rotated) volume.
    interpolation_order : int, optional
        Spline interpolation order (default is 2).
    Returns
    -------
    transformed : np.ndarray
        Transformed 3D volume resampled into the new bounding box.
    """
    transformed = ndi.affine_transform(
        da.from_array(
            data, chunks=(2, reference_shape[1], reference_shape[2])
        ),
        matrix=final_transform,
        output_shape=bounding_box,
        order=interpolation_order,
    ).astype(data.dtype)

    return transformed


# ------------ OPTIMISATION OBJECTIVE ------------ #


def registration_objective(
    pitch,
    yaw,
    roll,
    z_slice,
    atlas_volume,
    sample,
    atlas_to_moving_scale,
    atlas_res,
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
    atlas_to_moving_scale : list of float
        Scaling [x, y, z] applied to moving image to match atlas resolution.
    atlas_res : list or tuple of float
        The resolution of the atlas.

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

        file_path = Path(
            "parameters/brainglobe_registration/" f"{transform_type}.txt"
        )

        transform_params = open_parameter_file(file_path)

        # Force internal pixel type to float before wrapping in list
        transform_params["FixedInternalImagePixelType"] = ["float"]
        transform_params["MovingInternalImagePixelType"] = ["float"]

        transform_param_list = [(transform_type, transform_params)]

        moving, fixed = prepare_images(
            sample, current_atlas_slice, atlas_res, scale=atlas_to_moving_scale
        )

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
            return 0
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
            atlas_res=atlas_res,
            atlas_to_moving_scale=atlas_to_moving_scale,
        )
        return 0.0 if np.isnan(score) else score

    except Exception as e:
        warnings.warn(f"Failed registration attempt: {e}")
        return -1.0


def run_bayesian(
    atlas_volume, sample, manual_z_range, atlas_to_moving_scale, atlas_res
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
    atlas_to_moving_scale : list of float
        Scaling [x, y, z] applied to moving image to match atlas resolution.
    atlas_res : list or tuple of float
        The resolution of the atlas.

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

    z_slices = range(atlas_volume.shape[0])
    best_result = {"target": -np.inf}

    iterate_through_all_z = False

    if iterate_through_all_z:
        z_slices = range(max(0, 20), min(atlas_volume.shape[0], 61))

        for z in z_slices:
            print(f"\n[INFO] Bayesian search at z-slice {z}")

            optimizer = BayesianOptimization(
                f=lambda pitch, yaw, roll: registration_objective(
                    pitch,
                    yaw,
                    roll,
                    z,
                    atlas_volume,
                    sample,
                    atlas_to_moving_scale,
                    atlas_res,
                ),
                pbounds={k: v for k, v in pbounds.items() if k != "z_slice"},
                verbose=2,
                random_state=42,
            )

            optimizer.maximize(init_points=5, n_iter=15)

            result = {
                "target": optimizer.max["target"],
                "params": {**optimizer.max["params"], "z_slice": z},
            }

            if result["target"] > best_result["target"]:
                best_result = result
    else:
        optimizer = BayesianOptimization(
            f=lambda pitch, yaw, roll, z_slice: registration_objective(
                pitch,
                yaw,
                roll,
                z_slice,
                atlas_volume,
                sample,
                atlas_to_moving_scale,
                atlas_res,
            ),
            pbounds=pbounds,
            verbose=2,
            random_state=42,
        )

        optimizer.maximize(init_points=5, n_iter=15)

        best_result = {
            "target": optimizer.max["target"],
            "params": optimizer.max["params"],
        }

    print(f"\n[Bayesian] Optimal result:\nScore: {best_result['target']:.4f}")
    for k, v in best_result["params"].items():
        print(f"{k}: {v:.2f}")


def main():
    atlas_name = "allen_mouse_100um"
    atlas = BrainGlobeAtlas(atlas_name)
    atlas_volume = atlas.reference
    atlas_res = atlas.resolution  # (z, y, x)

    # Scale moving image to atlas (i.e. just like in napari)
    my_scale = [25.0, 25.0, 25.0]

    # CHOOSE SAMPLE #

    sample = tiff.imread("resources/sample_hipp.tif")

    # rot_matrix, bounding_box = create_rotation_matrix(
    #    roll=2.10, yaw=-0.30, pitch=1.20, img_shape=atlas_volume.shape
    # )
    # rotated_volume = rotate_volume(
    #    atlas_volume, atlas_volume.shape, rot_matrix, bounding_box
    # )
    # sample = rotated_volume[202].compute()

    manual_z_range = (50, 90)  # let this be gui input

    run_bayesian(atlas_volume, sample, manual_z_range, my_scale, atlas_res)


if __name__ == "__main__":
    main()
