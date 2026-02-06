from typing import Tuple

import dask.array as da
import dask_image.ndinterp as ndi
import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation
from skimage.transform import rescale


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
    offset : np.ndarray
        Translation vector to apply after rotation to fit the rotated volume
        into the new bounding box.
    bounding_box : Tuple[int, int, int]
        Shape of the rotated volume that fully contains the transformed data.
    """
    # Create the rotation matrix
    # roll_matrix = active_matrix_from_angle(0, np.deg2rad(roll))
    # yaw_matrix = active_matrix_from_angle(1, np.deg2rad(yaw))
    # pitch_matrix = active_matrix_from_angle(2, np.deg2rad(pitch))
    #
    # # Combine rotation matrices
    # rotation_matrix = yaw_matrix @ pitch_matrix @ roll_matrix

    rotation_matrix = Rotation.from_euler(
        "xyz", [roll, yaw, pitch], degrees=True
    ).as_matrix()

    inv_rot = rotation_matrix.T

    input_shape = np.array(img_shape)
    corners = (
        np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 1, 0],
                [1, 0, 1],
                [0, 1, 1],
                [1, 1, 1],
            ]
        )
        * input_shape
    )

    # Rotate corners around input center
    input_center = input_shape / 2.0
    rotated_corners = (
        rotation_matrix @ (corners - input_center).T
    ).T + input_center

    # Find bounding box of rotated volume
    min_coords = rotated_corners.min(axis=0)
    max_coords = rotated_corners.max(axis=0)
    output_shape = np.ceil(max_coords - min_coords).astype(int)

    # Output center
    output_center = output_shape / 2.0

    # Offset: map output coords to input coords
    offset = input_center - inv_rot @ output_center

    # # Translate the origin to the center of the image
    # origin = np.asarray(img_shape) / 2

    # translate_to_center = np.eye(4)
    # translate_to_center[:3, -1] = -origin
    #
    # bounding_box = calculate_rotated_bounding_box(img_shape, full_matrix)
    # new_translation = np.asarray(bounding_box) / 2
    #
    # post_rotate_translation = np.eye(4)
    # post_rotate_translation[:3, -1] = new_translation
    #
    # # Combine the matrices. The order of operations is:
    # # 1. Translate the origin to the center of the image
    # # 2. Rotate the image
    # # 3. Translate the origin back to the top left corner
    #
    # final_transform = np.linalg.inv(
    #     post_rotate_translation @ full_matrix @ translate_to_center
    # )

    return inv_rot, offset, output_shape


def rotate_volume(
    data: np.ndarray,
    reference_shape: Tuple[int, int, int],
    final_transform: np.ndarray,
    bounding_box: Tuple[int, int, int],
    interpolation_order: int = 2,
    offset: np.ndarray = None,
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
        offset=offset,
        output_shape=bounding_box,
        order=interpolation_order,
    ).astype(data.dtype)

    return transformed


def scale_moving_image(
    moving_image: np.ndarray,
    atlas_res: Tuple[float, float, float],
    moving_res: Tuple[float, float, float] = (1.0, 1.0, 1.0),
):
    """
    Scale the moving image to have resolution equal to the atlas.

    Parameters
    ----------
    moving_image : np.ndarray
        Image to be scaled.
    atlas_res : Tuple[float, float, float]
        Resolution (z, y, x) of the atlas.
    moving_res : Tuple[float, float, float]
        Resolution (z, y, x) of moving image.
        Defaults to (1.0, 1.0, 1.0).

    Returns
    -------
    np.ndarray
        Rescaled image with a shape that reflects conversion from
        `moving_res` to `atlas_res`, preserving physical dimensions.


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


def calculate_rotated_bounding_box(
    image_shape: Tuple[int, int, int], rotation_matrix: npt.NDArray
) -> Tuple[int, int, int]:
    """
    Calculates the bounding box of the rotated image.

    This function calculates the bounding box of the rotated image given the
    image shape and rotation matrix. The bounding box is calculated by
    transforming the corners of the image and finding the minimum and maximum
    values of the transformed corners.

    Parameters
    ------------
    image_shape : Tuple[int, int, int]
        The shape of the image.
    rotation_matrix : npt.NDArray
        The rotation matrix.

    Returns
    --------
    Tuple[int, int, int]
        The bounding box of the rotated image.
    """
    corners = np.array(
        [
            [0, 0, 0, 1],
            [image_shape[0], 0, 0, 1],
            [0, image_shape[1], 0, 1],
            [0, 0, image_shape[2], 1],
            [image_shape[0], image_shape[1], 0, 1],
            [image_shape[0], 0, image_shape[2], 1],
            [0, image_shape[1], image_shape[2], 1],
            [image_shape[0], image_shape[1], image_shape[2], 1],
        ]
    )

    transformed_corners = np.dot(rotation_matrix, corners.T)
    min_corner = np.min(transformed_corners, axis=1)
    max_corner = np.max(transformed_corners, axis=1)

    return (
        int(np.round(max_corner[0] - min_corner[0])),
        int(np.round(max_corner[1] - min_corner[1])),
        int(np.round(max_corner[2] - min_corner[2])),
    )
