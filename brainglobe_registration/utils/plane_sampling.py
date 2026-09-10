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
    Build a 3x3 rotation matrix from Euler angles.

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


def compute_rotation_offset(
    roll: float,
    yaw: float,
    pitch: float,
    volume_shape: Tuple[int, int, int],
) -> Tuple[npt.NDArray[np.float64], npt.NDArray, Tuple[int, int, int]]:
    """
    Compute rotation matrix, offset, and output shape.

    Parameters
    ----------
    roll : float
        Rotation around X-axis (degrees).
    yaw : float
        Rotation around Y-axis (degrees).
    pitch : float
        Rotation around Z-axis (degrees).
    volume_shape : tuple of (int, int, int)
        Shape of the 3D volume (D, H, W).

    Returns
    -------
    Tuple[npt.NDArray[np.float64], npt.NDArray, Tuple[int, int, int]]
        - inv_rotation: 3x3 inverse rotation matrix (for affine_transform)
        - offset: 3-element offset vector for coordinate mapping
        - output_shape: 3D bounding box shape (D, H, W)
    """
    # Create rotation matrix from Euler angles
    rotation_matrix = Rotation.from_euler(
        "XYZ", [roll, yaw, pitch], degrees=True
    ).as_matrix()

    # inverse=transpose for orthogonal:)
    inv_rotation = rotation_matrix.T
    input_shape = np.array(volume_shape, dtype=np.float64)
    input_center = input_shape / 2.0

    # Compute output shape by rotating corners
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
    rotated_corners = (
        rotation_matrix @ (corners - input_center).T
    ).T + input_center

    # Find bounding box of rotated volume
    min_coords = rotated_corners.min(axis=0)
    max_coords = rotated_corners.max(axis=0)
    output_shape = tuple(np.ceil(max_coords - min_coords).astype(int))

    # hack: Because we don't generate a new 3D volume, Napari's Z-slider
    # doesn't change size. We should anchor the Z-center to the original
    # input depth, while letting Y and X expand for the canvas.

    output_center = np.array(
        [
            input_shape[0] / 2.0,  # Z: Tied to Napari's static slider center
            output_shape[1] / 2.0,  # Y: New expanded canvas height
            output_shape[2] / 2.0,  # X: New expanded canvas width
        ]
    )

    # Offset: map output coords to input coords
    offset = input_center - inv_rotation @ output_center

    return inv_rotation, offset, output_shape


def sample_plane(
    volume: npt.NDArray,
    z_index: float,
    inv_rotation: npt.NDArray,
    offset: npt.NDArray,
    output_shape: Optional[Tuple[int, int]] = None,
    interpolation_order: int = 0,
    mode: str = "constant",
    cval: float = 0.0,
) -> npt.NDArray:
    """
    Sample a single 2D plane from a 3D volume at a given z-position
    and rotation, WITHOUT modifying the source volume.

    Uses the same coordinate transformation as transforms.rotate_volume
    to ensure consistent results.

    Parameters
    ----------
    volume : npt.NDArray
        The static 3D volume to sample from. Shape (D, H, W).
        This is never modified.
    z_index : float
        The slice position along the viewing axis (the napari slider value).
    inv_rotation : npt.NDArray
        3x3 INVERSE rotation matrix from compute_rotation_offset().
        This is rotation_matrix.T (the transpose).
    offset : npt.NDArray
        3-element offset vector from compute_rotation_offset().
        This is CRITICAL for preventing clipping.
    output_shape : tuple of (int, int), optional
        (height, width) of the output 2D plane.
    interpolation_order : int, optional
        Spline interpolation order for map_coordinates.
        Default: 0.
    mode : str, optional
        Boundary mode for map_coordinates ('constant', 'nearest', etc.).
        Default: 'constant'.
    cval : float, optional
        Fill value when mode='constant'. Default: 0.0.

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

    # Build plane coordinates at the given z_index
    plane_coords = np.stack(
        [
            np.full(grid_y.shape, z_index, dtype=np.float64),
            grid_y,
            grid_x,
        ],
        axis=0,
    ).reshape(3, -1)

    # Apply the affine transformation: source = inv_rotation @ (plane - offset)
    source_coords = inv_rotation @ plane_coords + offset[:, None]

    # map_coordinates handles interpolation and boundary conditions
    sampled = map_coordinates(
        volume,
        source_coords,
        order=interpolation_order,
        mode=mode,
        cval=cval,
    ).reshape(output_shape)

    return sampled


def sample_annotation_plane(
    annotation_volume: npt.NDArray,
    z_index: float,
    inv_rotation: npt.NDArray,
    offset: npt.NDArray,
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
    inv_rotation : npt.NDArray
        3x3 inverse rotation matrix from compute_rotation_offset().
    offset : npt.NDArray
        3-element offset vector from compute_rotation_offset().
    output_shape : tuple of (int, int), optional
        Output plane dimensions. If None, uses the volume's (H, W).

    Returns
    -------
    npt.NDArray
        2D annotation plane with integer labels preserved.
    """
    return sample_plane(
        annotation_volume,
        z_index,
        inv_rotation,
        offset,
        output_shape=output_shape,
        interpolation_order=0,  # NN: no label mixing
    ).astype(annotation_volume.dtype)
