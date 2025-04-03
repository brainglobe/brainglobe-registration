import numpy as np
import numpy.typing as npt
from skimage.color import rgb2gray
from skimage.registration import optical_flow_ilk, optical_flow_tvl1


def compute_optical_flow_skimage(
    image1: npt.NDArray, image2: npt.NDArray, method: str = "tvl1"
) -> npt.NDArray:
    """
    Compute dense optical flow using skimage.

    Parameters
    ----------
    image1 : npt.NDArray
        First image (reference).
    image2 : npt.NDArray
        Second image (moving).
    method : str
        Optical flow method: "tvl1" or "ilk".

    Returns
    -------
    flow : npt.NDArray
        Optical flow (flow_y, flow_x)
    """
    # Ensure grayscale
    if image1.ndim == 3:
        image1 = rgb2gray(image1)
    if image2.ndim == 3:
        image2 = rgb2gray(image2)

    image1 = image1.astype(np.float32)
    image2 = image2.astype(np.float32)

    if method == "tvl1":
        flow = optical_flow_tvl1(image1, image2)
    elif method == "ilk":
        flow = optical_flow_ilk(image1, image2, radius=5)
    else:
        raise ValueError("Unsupported method. Use 'tvl1' or 'ilk'.")

    return flow  # shape: (2, H, W)


def compare_deformation_fields(
    elastix_field: npt.NDArray, optical_flow: npt.NDArray
) -> float:
    """
    Compare the elastix deformation field with optical flow field.

    Parameters
    ----------
    elastix_field : npt.NDArray
        The deformation field from elastix (H, W, 2).
    optical_flow : npt.NDArray
        The optical flow from skimage (2, H, W).

    Returns
    -------
    float
        Mean squared error between the fields.
    """
    # Convert skimage format (2, H, W) to (H, W, 2)
    optical_flow = np.moveaxis(optical_flow, 0, -1)

    # Resize if shapes don't match
    if elastix_field.shape != optical_flow.shape:
        raise ValueError(
            "Shape mismatch between elastix and optical flow fields"
        )

    diff = elastix_field - optical_flow
    mse = np.mean(np.square(diff))
    return mse
