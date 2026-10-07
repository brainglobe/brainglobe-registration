from typing import Tuple

import numpy.typing as npt
from skimage.transform import rescale


def scale_moving_image(
    moving_image: npt.NDArray,
    atlas_res: Tuple[float, float, float],
    moving_res: Tuple[float, float, float] = (1.0, 1.0, 1.0),
):
    """
    Scale the moving image to have resolution equal to the atlas.

    Parameters
    ----------
    moving_image : npt.NDArray
        Image to be scaled.
    atlas_res : Tuple[float, float, float]
        Resolution (z, y, x) of the atlas.
    moving_res : Tuple[float, float, float]
        Resolution (z, y, x) of moving image.
        Defaults to (1.0, 1.0, 1.0).

    Returns
    -------
    npt.NDArray
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
