import numpy as np
import numpy.typing as npt
from brainglobe_utils.image.scale import scale_and_convert_to_16_bits
from scipy.ndimage import gaussian_filter
from skimage import morphology
from skimage.restoration import denoise_bilateral
from tqdm import trange


def filter_image(image: npt.NDArray) -> npt.NDArray:
    """
    Filter a 3D image using a despeckle filter and a pseudo flatfield filter.
    Scales and converts the image back to 16 bits following filtering.

    Originally from: https://github.com/brainglobe/brainreg/blob/v1.0.10/brainreg/core/utils/preprocess.py

    Parameters
    ----------
    image : npt.NDArray
        The image to filter

    Returns
    -------
    npt.NDArray
        The filtered image
    """
    image = image.astype(np.float64, copy=False)

    if image.ndim == 2:
        image = filter_plane(image)
    else:
        for i in trange(image.shape[-1], desc="filtering", unit="plane"):
            image[..., i] = filter_plane(image[..., i])

    image = scale_and_convert_to_16_bits(image)

    return image


def filter_plane(img_plane: npt.NDArray) -> npt.NDArray:
    """
    Apply a set of filter to the plane (typically to avoid overfitting details
    in the image during registration)
    The filter is composed of a despeckle filter using opening and a pseudo
    flatfield filter

    Originally from: https://github.com/brainglobe/brainreg/blob/v1.0.10/brainreg/core/utils/preprocess.py

    Parameters
    ----------
    img_plane : npt.NDArray
        A 2D array to filter

    Returns
    -------
    npt.NDArray
        Filtered image
    """
    img_plane = despeckle_by_opening(img_plane)
    img_plane = pseudo_flatfield(img_plane)
    img_plane = bilateral_filter(img_plane)

    return img_plane


def despeckle_by_opening(
    img_plane: npt.NDArray, radius: int = 2
) -> npt.NDArray:
    """
    Despeckle the image plane using a grayscale opening operation

    Originally from: https://github.com/brainglobe/brainreg/blob/v1.0.10/brainreg/core/utils/preprocess.py

    Parameters
    ----------
    img_plane : npt.NDArray
        The image to filter

    radius: int
        The radius of the opening kernel

    Returns
    -------
    npt.NDArray
        The despeckled image
    """
    kernel = morphology.disk(radius)
    morphology.opening(img_plane, out=img_plane, footprint=kernel)

    return img_plane


def pseudo_flatfield(img_plane: npt.NDArray, sigma: int = 5):
    """
    Pseudo flat field filter implementation using a de-trending by a
    heavily gaussian filtered copy of the image.

    Originally from: https://github.com/brainglobe/brainreg/blob/v1.0.10/brainreg/core/utils/preprocess.py

    Parameters
    ----------
    img_plane : npt.NDArray
        The image to filter

    sigma : int
        The sigma of the gaussian filter applied to the
        image used for de-trending

    Returns
    -------
    npt.NDArray
        The pseudo flat field filtered image
    """
    filtered_img = gaussian_filter(img_plane, sigma)

    return img_plane / (filtered_img + 1)


def bilateral_filter(
    img_plane: npt.NDArray,
    win_size: int = 5,
    sd_intensity: float = 0.1,
    sigma_spatial: float = 15.0,
) -> npt.NDArray:
    """
    Implementation of a non-linear and noise-reducing smoothing filter for the image plane
    that preserves its edges. The filter smoothes the image by taking weighted averages for
    each pixel, by taking into account spatial closeness and intensity of nearby pixels.
    This weight can be based on a Gaussian distribution.

    Parameters
    ----------
    img_plane : npt.NDArray
        The image to filter.
    win_size : int
        Size of each region around pixel to be considered for filtering.
    sigma_intensity : float
        Gaussian function of the Euclidean distance between two color values
        and a certain standard deviation
    sigma_spatial : float
        Radiometric similarity is measured by the Gaussian function of the
        Euclidean distance between two color values and a certain standard deviation
    multichannel : boolean
        False - Grayscale images
        True - Colour Images

    Returns
    -------
    npt.NDArray
        The filtered image
    """
    return denoise_bilateral(
        img_plane,
        win_size=win_size,
        sigma_color=sd_intensity,
        sigma_spatial=sigma_spatial,
        multichannel=False,
    )
