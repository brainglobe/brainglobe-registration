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
