"""
Sample 2D planes from a 3D volume at arbitrary rotations.
"""

from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation


def build_rotation_matrix(
    roll: float, yaw: float, pitch: float
) -> npt.NDArray:
    """
    build a 3x3 rotation matrix from Euler angles.

    Parameters
    ----------
    roll : float
        Rotation around the X-axis (degrees).
    yaw : float
        Rotation around the Y-axis (degrees).
    pitch : float
        Rotation around the Z-axis (degrees).

    Returns
    -------
    npt.NDArray
        3x3 rotation matrix.
    """
    return Rotation.from_euler(
        "XYZ", [roll, yaw, pitch], degrees=True
    ).as_matrix()


def sample_plane(
    volume: npt.NDArray,
    z_index: float,
    rotation_matrix: npt.NDArray,
    output_shape: Optional[Tuple[int, int]] = None,
    interpolation_order: int = 2,
) -> npt.NDArray:
    """
    Sample a single 2D plane from a 3D volume at a given z-position
    and rotation, WITHOUT modifying the source volume.

    Parameters
    ----------
    volume : npt.NDArray
        The static 3D volume to sample from. Shape (D, H, W).
        This is never modified.
    z_index : float
        The slice position along the viewing axis (the napari slider value).
    rotation_matrix : npt.NDArray
        3x3 rotation matrix from build_rotation_matrix().
    output_shape : tuple of (int, int), optional
        (height, width) of the output 2D plane.
    interpolation_order : int, optional
        Spline interpolation order for map_coordinates.
        Default: 2.

    Returns
    -------
    npt.NDArray
        2D array of shape output_shape containing the sampled plane.
    """
    if output_shape is None:
        output_shape = (volume.shape[1], volume.shape[2])

    out_h, out_w = output_shape
    y_coords = np.arange(out_h, dtype=np.float64)
    x_coords = np.arange(out_w, dtype=np.float64)
    grid_y, grid_x = np.meshgrid(y_coords, x_coords, indexing="ij")

    plane_coords = np.stack(
        [
            np.full(grid_y.shape, z_index, dtype=np.float64),
            grid_y,
            grid_x,
        ],
        axis=0,
    ).reshape(3, -1)

    volume_center = np.array(volume.shape, dtype=np.float64) / 2.0

    # inverse = transpose in orthogonal:)
    inv_rotation = rotation_matrix.T

    # shift to center, rotate, shift back
    source_coords = (
        inv_rotation @ (plane_coords - volume_center[:, None])
    ) + volume_center[:, None]

    # map_coordinates handles interpolation and boundary conditions.
    # mode='constant', cval=0 means out-of-bounds regions are black.
    sampled = map_coordinates(
        volume,
        source_coords,
        order=interpolation_order,
        mode="constant",
        cval=0.0,
    ).reshape(output_shape)

    return sampled


def sample_annotation_plane(
    annotation_volume: npt.NDArray,
    z_index: float,
    rotation_matrix: npt.NDArray,
    output_shape: Optional[Tuple[int, int]] = None,
) -> npt.NDArray:
    """
    Sample a 2D annotation plane with nearest-neighbor interpolation.

    Convenience wrapper around sample_plane() that ensures integer
    labels are preserved (no interpolation artifacts at label boundaries).

    Parameters
    ----------
    annotation_volume : npt.NDArray
        The 3D annotation volume with integer labels.
    z_index : float
        Slice position.
    rotation_matrix : npt.NDArray
        3x3 rotation matrix.
    output_shape : tuple of (int, int), optional
        Output plane dimensions.

    Returns
    -------
    npt.NDArray
        2D annotation plane with integer labels preserved.
    """
    return sample_plane(
        annotation_volume,
        z_index,
        rotation_matrix,
        output_shape=output_shape,
        interpolation_order=0,  # NN:no label mixing
    ).astype(annotation_volume.dtype)
