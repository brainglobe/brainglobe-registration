"""
Visualization helpers for registration QC (e.g. intensity difference map).
"""

import numpy as np
import numpy.typing as npt


def generate_intensity_difference_map(
    image1: npt.NDArray,
    image2: npt.NDArray,
    normalize: bool = True,
) -> npt.NDArray:
    """
    Compute pixel-wise absolute intensity difference between two images.

    Used as a simple QC heatmap: dark = low difference, bright = high
    difference. Optional normalization makes the map comparable across
    modalities. Pixel-wise differences can be misleading when contrast
    flips between images; use as an optional diagnostic only.

    Parameters
    ----------
    image1 : npt.NDArray
        First image (e.g. atlas in data space). 2D or 3D.
    image2 : npt.NDArray
        Second image (e.g. moving/registered). 2D or 3D.
    normalize : bool, optional
        If True, scale each image to [0, 1] by its own min/max before
        differencing, so the map is in a comparable range. Default True.

    Returns
    -------
    npt.NDArray
        Absolute difference map, shape = min(image1.shape, image2.shape)
        along each axis (cropped to overlap). Dtype float32 in [0, 1]
        if normalize=True, else matches input dtypes promoted to float.
    """
    if image1.ndim != image2.ndim:
        raise ValueError(
            f"Images must have same number of dimensions. "
            f"Got {image1.ndim}D and {image2.ndim}D."
        )
    ndim = image1.ndim
    if ndim not in (2, 3):
        raise ValueError(
            f"Only 2D and 3D images supported. Got ndim={ndim}."
        )

    # Crop to overlapping region
    shape = tuple(
        min(image1.shape[i], image2.shape[i]) for i in range(ndim)
    )
    img1 = image1[tuple(slice(0, s) for s in shape)].astype(np.float64)
    img2 = image2[tuple(slice(0, s) for s in shape)].astype(np.float64)

    if normalize:
        for arr in (img1, img2):
            lo, hi = np.min(arr), np.max(arr)
            if hi > lo:
                arr -= lo
                arr /= hi - lo
            # else constant; leave as-is (will be 0 in diff)

    diff = np.abs(img1 - img2)
    # Clamp to [0, 1] for heatmap display when normalized
    if normalize:
        diff = np.clip(diff, 0.0, 1.0)
    return diff.astype(np.float32)
