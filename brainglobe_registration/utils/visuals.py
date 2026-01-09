"""
Visualization utilities for brainglobe-registration.

This module contains functions for generating visualization overlays
such as checkerboard patterns for comparing registered images.
"""

from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt


def generate_checkerboard(
    image1: npt.NDArray,
    image2: npt.NDArray,
    square_size: int = 32,
    normalize: bool = True,
) -> npt.NDArray:
    """
    Generate a checkerboard pattern by alternating between two images.

    This function creates a checkerboard visualization where alternating squares
    show either image1 or image2. This is useful for comparing registered images
    to assess registration quality.

    Parameters
    ----------
    image1 : npt.NDArray
        First image to display in checkerboard pattern (typically the original image).
    image2 : npt.NDArray
        Second image to display in checkerboard pattern (typically the registered image).
    square_size : int, optional
        Size of each checkerboard square in pixels, by default 32.
        For 3D images, this applies to the last two dimensions (y, x).
    normalize : bool, optional
        Whether to normalize the output to [0, 1] range, by default True.
        If False, output will be in the same dtype and range as the input images.

    Returns
    -------
    npt.NDArray
        The checkerboard pattern combining both images.
        Shape matches the input images.
        Dtype is float64 if normalize=True, otherwise matches input dtype.

    Examples
    --------
    >>> import numpy as np
    >>> img1 = np.random.rand(100, 100)
    >>> img2 = np.random.rand(100, 100)
    >>> checkerboard = generate_checkerboard(img1, img2, square_size=16)
    """
    # Ensure images have the same shape
    if image1.shape != image2.shape:
        raise ValueError(
            f"Images must have the same shape. "
            f"Got image1.shape={image1.shape}, image2.shape={image2.shape}"
        )

    # Handle different dimensionalities
    ndim = image1.ndim

    if ndim == 2:
        # 2D case: create checkerboard pattern in (y, x)
        checkerboard = _generate_2d_checkerboard(
            image1, image2, square_size
        )
    elif ndim == 3:
        # 3D case: create checkerboard pattern in each slice (z, y, x)
        # The pattern is the same for all z-slices
        checkerboard = _generate_3d_checkerboard(
            image1, image2, square_size
        )
    else:
        raise ValueError(
            f"Unsupported image dimensionality: {ndim}. "
            "Only 2D and 3D images are supported."
        )

    # Normalize to [0, 1] range if requested
    if normalize:
        # Convert to float for normalization
        checkerboard = checkerboard.astype(np.float64)
        min_val = checkerboard.min()
        max_val = checkerboard.max()
        if max_val > min_val:
            checkerboard = (checkerboard - min_val) / (max_val - min_val)
        else:
            # Handle case where all values are the same - set to midpoint for visibility
            checkerboard.fill(0.5)

    return checkerboard


def _generate_2d_checkerboard(
    image1: npt.NDArray, image2: npt.NDArray, square_size: int
) -> npt.NDArray:
    """Generate a 2D checkerboard pattern."""
    h, w = image1.shape
    checkerboard = np.zeros_like(image1, dtype=np.float64)

    # Create a grid of square indices
    y_indices = np.arange(h) // square_size
    x_indices = np.arange(w) // square_size

    # Create the checkerboard mask: (i + j) % 2 determines the pattern
    # Even sum -> use image1, odd sum -> use image2
    mask = (y_indices[:, np.newaxis] + x_indices[np.newaxis, :]) % 2 == 0

    # Apply the mask
    checkerboard[mask] = image1[mask]
    checkerboard[~mask] = image2[~mask]

    return checkerboard


def _generate_3d_checkerboard(
    image1: npt.NDArray, image2: npt.NDArray, square_size: int
) -> npt.NDArray:
    """Generate a 3D checkerboard pattern (pattern is the same for all z-slices)."""
    d, h, w = image1.shape
    checkerboard = np.zeros_like(image1, dtype=np.float64)

    # Create a grid of square indices for y and x dimensions
    y_indices = np.arange(h) // square_size
    x_indices = np.arange(w) // square_size

    # Create the checkerboard mask for each slice
    # The pattern is the same across all z-slices
    mask = (y_indices[:, np.newaxis] + x_indices[np.newaxis, :]) % 2 == 0

    # Apply the mask to each z-slice
    for z in range(d):
        checkerboard[z, mask] = image1[z, mask]
        checkerboard[z, ~mask] = image2[z, ~mask]

    return checkerboard
