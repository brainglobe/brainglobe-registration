from typing import Tuple

import dask.array as da
import dask_image.ndinterp as ndi
import numpy as np
from pytransform3d.rotations import active_matrix_from_angle

from brainglobe_registration.utils.utils import (
    calculate_rotated_bounding_box,
)


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
