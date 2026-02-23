"""
Sample 2D planes from a 3D volume at arbitrary rotations.

Instead of rotating the entire 3D atlas volume (slow, ~2-5s),
this module samples a single 2D slice directly from the static
volume using scipy.ndimage.map_coordinates (~10ms).

This implements the approach described in Issue #151:
https://github.com/brainglobe/brainglobe-registration/issues/151
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

    Uses the same convention as create_rotation_matrix() in transforms.py
    but returns ONLY the rotation matrix — no bounding box, no offset.
    This is all we need for plane sampling.

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

    The key idea: instead of rotating 77 million voxels (the entire
    atlas), we figure out which ~170,000 voxels (one slice) we need
    and read just those.

    Parameters
    ----------
    volume : npt.NDArray
        The static 3D volume to sample from. Shape (D, H, W).
        This is never modified.
    z_index : float
        The slice position along the viewing axis (the napari slider value).
    rotation_matrix : npt.NDArray
        3x3 rotation matrix from build_rotation_matrix().
        Use np.eye(3) for no rotation (identity = standard orthogonal slice).
    output_shape : tuple of (int, int), optional
        (height, width) of the output 2D plane.
        Defaults to volume.shape[1:] (i.e., H x W of the original volume).
    interpolation_order : int, optional
        Spline interpolation order for map_coordinates.
        Use 2 for reference images (smooth), 0 for annotations (preserve labels).
        Default: 2.

    Returns
    -------
    npt.NDArray
        2D array of shape output_shape containing the sampled plane.
    """
    if output_shape is None:
        output_shape = (volume.shape[1], volume.shape[2])

    out_h, out_w = output_shape

    # Step 1: Build a grid of (y, x) points for the output plane.
    # These are the pixel positions in the 2D image we want to produce.
    y_coords = np.arange(out_h, dtype=np.float64)
    x_coords = np.arange(out_w, dtype=np.float64)
    grid_y, grid_x = np.meshgrid(y_coords, x_coords, indexing="ij")

    # Step 2: Create 3D coordinates for this plane in "rotated space."
    # The plane lives at z=z_index, spanning the full (y, x) range.
    # Shape: (3, H*W) — each column is a [z, y, x] point.
    plane_coords = np.stack(
        [
            np.full(grid_y.shape, z_index, dtype=np.float64),
            grid_y,
            grid_x,
        ],
        axis=0,
    ).reshape(3, -1)

    # Step 3: Transform from "rotated space" back to the static volume.
    # We rotate around the volume center so the rotation looks natural.
    volume_center = np.array(volume.shape, dtype=np.float64) / 2.0

    # For plane sampling, we need the inverse rotation:
    # "given a point in the rotated view, where was it in the original volume?"
    # Since rotation matrices are orthogonal, inverse = transpose.
    inv_rotation = rotation_matrix.T

    # Shift to center, rotate, shift back
    source_coords = (
        inv_rotation @ (plane_coords - volume_center[:, None])
    ) + volume_center[:, None]

    # Step 4: Sample the volume at those coordinates.
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
        interpolation_order=0,  # Nearest-neighbor: no label mixing
    ).astype(annotation_volume.dtype)