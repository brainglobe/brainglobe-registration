"""
Visualization utilities for brainglobe-registration.

This module contains functions for generating visualization overlays
such as checkerboard patterns for comparing registered images.
"""

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

    This function creates a checkerboard visualization where alternating
    squares show either image1 or image2. This is useful for comparing
    registered images to assess registration quality.

    If normalize=True, both images are normalized separately to match their
    intensity ranges, ensuring fair visual comparison in the checkerboard.

    Parameters
    ----------
    image1 : npt.NDArray
        First image to display in checkerboard pattern (typically the
        original image).
    image2 : npt.NDArray
        Second image to display in checkerboard pattern (typically the
        registered image).
    square_size : int, optional
        Size of each checkerboard square in pixels, by default 32.
        For 3D images, this applies to the last two dimensions (y, x).
    normalize : bool, optional
        Whether to normalize both images separately to match their intensity
        ranges, by default True. This ensures both images appear at similar
        brightness levels for fair comparison.
        The output dtype will be uint16 if input is integer type, or
        float32 if input is float type.
        If False, output will be in the same dtype and range as the input
        images.

    Returns
    -------
    npt.NDArray
        The checkerboard pattern combining both images.
        Shape is the minimum of both input shapes (overlapping region) if
        shapes differ, otherwise matches input shape. Dtype matches input dtype
        if normalize=False, otherwise uint16 for integer types or float32 for
        float types.

    Examples
    --------
    >>> import numpy as np
    >>> img1 = np.random.rand(100, 100)
    >>> img2 = np.random.rand(100, 100)
    >>> checkerboard = generate_checkerboard(img1, img2, square_size=16)
    """
    # Handle shape mismatches by cropping to minimum shape (overlapping region)
    # Just a extra safety check before pad/crop
    if image1.ndim != image2.ndim:
        raise ValueError(
            f"Images must have the same number of dimensions. "
            f"Got image1.ndim={image1.ndim}, image2.ndim={image2.ndim}"
        )

    if image1.shape != image2.shape:
        # Crop both images to the minimum shape along each dimension
        crop_slices = tuple(
            slice(0, min(s1, s2)) for s1, s2 in zip(image1.shape, image2.shape)
        )
        image1 = image1[crop_slices]
        image2 = image2[crop_slices]

    # Determine output dtype based on original input dtypes
    if normalize:
        # Use uint16 for normalization of integer types, float32 for floats
        # Check original dtype before normalization
        original_dtype = np.result_type(image1, image2)
        if np.issubdtype(original_dtype, np.integer):
            output_dtype = np.uint16
        else:
            output_dtype = np.float32
    else:
        # Keep original dtype (use common dtype if they differ)
        output_dtype = np.result_type(image1, image2)

    # Normalize images separately if requested
    # This matches intensity ranges between the two images for fair comparison
    if normalize:
        image1, image2 = _normalize_images_for_comparison(
            image1, image2, output_dtype
        )

    # Generate checkerboard pattern (handles both 2D and 3D)
    ndim = image1.ndim
    if ndim not in (2, 3):
        raise ValueError(
            f"Unsupported image dimensionality: {ndim}. "
            "Only 2D and 3D images are supported."
        )

    checkerboard = _generate_checkerboard(
        image1, image2, square_size, output_dtype
    )

    return checkerboard


def _normalize_images_for_comparison(
    image1: npt.NDArray, image2: npt.NDArray, output_dtype: npt.DTypeLike
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Normalize two images separately to match their intensity ranges.

    This ensures both images appear at similar brightness levels for fair
    visual comparison in the checkerboard visualization.

    Parameters
    ----------
    image1 : npt.NDArray
        First image to normalize.
    image2 : npt.NDArray
        Second image to normalize.
    output_dtype : npt.DTypeLike
        Target output dtype (uint16 for integer types, float32 for float
        types).

    Returns
    -------
    tuple[npt.NDArray, npt.NDArray]
        Tuple of (normalized_image1, normalized_image2).
        Dtype matches the provided output_dtype parameter.
    """
    # Determine max value based on output dtype
    if np.issubdtype(output_dtype, np.integer):
        max_val = np.iinfo(output_dtype).max
    else:
        max_val = 1.0

    # Normalize image1
    img1_min = image1.min()
    img1_max = image1.max()
    if img1_max > img1_min:
        img1_normalized = (
            (image1.astype(np.float32) - img1_min) / (img1_max - img1_min)
        ) * max_val
    else:
        # Constant image - set to midpoint
        img1_normalized = np.full_like(image1, max_val / 2, dtype=np.float32)

    # Normalize image2
    img2_min = image2.min()
    img2_max = image2.max()
    if img2_max > img2_min:
        img2_normalized = (
            (image2.astype(np.float32) - img2_min) / (img2_max - img2_min)
        ) * max_val
    else:
        # Constant image - set to midpoint
        img2_normalized = np.full_like(image2, max_val / 2, dtype=np.float32)

    # Convert to output dtype
    return (
        img1_normalized.astype(output_dtype),
        img2_normalized.astype(output_dtype),
    )


def _generate_checkerboard(
    image1: npt.NDArray,
    image2: npt.NDArray,
    square_size: int,
    output_dtype: npt.DTypeLike,
) -> npt.NDArray:
    """
    Generate a checkerboard pattern for 2D or 3D images.

    For 2D images, creates a checkerboard pattern in (y, x).
    For 3D images, creates the same checkerboard pattern for each slice.

    Parameters
    ----------
    image1 : npt.NDArray
        First image (2D or 3D).
    image2 : npt.NDArray
        Second image (2D or 3D).
    square_size : int
        Size of each checkerboard square in pixels.
    output_dtype : npt.DTypeLike
        Output dtype for the checkerboard.

    Returns
    -------
    npt.NDArray
        Checkerboard pattern with same shape as input images.
    """
    # Get last two dimensions (y, x) for checkerboard pattern
    h, w = image1.shape[-2:]

    # Create sparse grid using ogrid for memory efficiency (C-speed)
    # ogrid returns (h x 1) and (1 x w) arrays instead of full (h x w) matrices
    y, x = np.ogrid[0:h, 0:w]

    # Create the checkerboard mask using broadcasting
    # (row_index // size + col_index // size) % 2 == 0
    # Even sum -> use image1, odd sum -> use image2
    # NumPy broadcasting automatically handles 2D vs 3D:
    # - For 2D (h, w): mask is (h, w), applies directly
    # - For 3D (d, h, w): mask is (h, w), broadcasting applies to all slices
    mask = ((y // square_size) + (x // square_size)) % 2 == 0

    # Use np.where for efficient single-pass assignment

    checkerboard = np.where(mask, image1, image2).astype(output_dtype)

    return checkerboard
