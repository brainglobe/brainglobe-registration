import imageio
import numpy as np
import tifffile as tiff
from brainglobe_atlasapi import BrainGlobeAtlas
from brainglobe_atlasapi.list_atlases import get_downloaded_atlases
from skimage.metrics import normalized_root_mse, structural_similarity
from skimage.transform import rescale
from sklearn.feature_selection import mutual_info_regression


def normalise_image(img):
    img = img.astype(np.float32)
    return (img - img.min()) / (img.max() - img.min() + 1e-8)


def scale_to_match_resolution(moving_img, moving_res, atlas_res):
    """
    Rescale the moving image to match the atlas resolution.

    Parameters
    ----------
    moving_img : ndarray
        The image to be rescaled.
    moving_res : tuple of float
        Pixel sizes (z, y, x) of the moving image.
    atlas_res : tuple of float
        Pixel sizes (z, y, x) of the atlas.

    Returns
    -------
    ndarray
        Rescaled image.
    """
    # Assume 2D images → only use x and y resolution
    x_factor = moving_res[2] / atlas_res[2]
    y_factor = moving_res[1] / atlas_res[1]
    scale = (y_factor, x_factor)

    return rescale(
        moving_img,
        scale,
        mode="constant",
        preserve_range=True,
        anti_aliasing=True,
    ).astype(moving_img.dtype)


def compute_similarity_metric(
    moving, fixed, metric="ncc", moving_res=(1, 1, 1), atlas_res=(1, 1, 1)
):
    """
    Computes the similarity metric between two sets of images.
    Parameters
    ----------
    moving : ndarray
        Moving (histology) image
    fixed : ndarray
        Fixed (atlas) image
    metric : str
        Metric to compute: 'ncc', 'ssim', 'rmse', or 'mi'
    moving_res : tuple of float
        Resolution of the moving image (z, y, x)
    atlas_res : tuple of float
        Resolution of the atlas image (z, y, x)
    Returns
    -------
    float
        Similarity metric
    """

    # Scale moving image to match atlas resolution
    moving = scale_to_match_resolution(moving, moving_res, atlas_res)

    # Match shape
    min_shape = tuple(min(a, b) for a, b in zip(moving.shape, fixed.shape))
    moving = moving[: min_shape[0], : min_shape[1]]
    fixed = fixed[: min_shape[0], : min_shape[1]]

    # Save debug images
    imageio.imwrite(
        r"C:\Users\saara\Documents\debug_moving_scaled.tif",
        moving.astype(np.float32),
    )
    imageio.imwrite(
        r"C:\Users\saara\Documents\debug_fixed_atlas.tif",
        fixed.astype(np.float32),
    )

    # Normalise
    moving = normalise_image(moving)
    fixed = normalise_image(fixed)

    print(f"Sample: {type(moving)}, {moving.shape}")
    print(f"Atlas slice: {type(fixed)}, {fixed.shape}")

    if metric == "ncc":
        # Normalized Cross-Correlation (NCC) – works well when
        # intensity ranges are consistent.
        return np.corrcoef(moving.ravel(), fixed.ravel())[0, 1]
    elif metric == "ssim":
        # Structural Similarity Index (SSIM) – incorporates luminance,
        # contrast, and structure.
        return structural_similarity(
            moving, fixed, data_range=moving.max() - moving.min()
        )
    elif metric == "rmse":
        # Mean Squared Error (MSE) – for same-modality data,
        # less robust to noise.
        return -normalized_root_mse(moving, fixed)  # lower is better
    elif metric == "mi":
        # Mutual Information (MI) – good for multi-modal data.
        return mutual_info_regression(
            moving.ravel().reshape(-1, 1), fixed.ravel()
        )[0]
    else:
        raise ValueError("Unsupported metric")


# print(get_downloaded_atlases())
atlas_name = (get_downloaded_atlases())[0]
# Allen 100um

# Access the 3D image volume
atlas = BrainGlobeAtlas(atlas_name)
atlas_volume = atlas.reference
atlas_res = atlas.resolution  # (z, y, x)

# Choose random atlas slice (i.e. specific z value) for comparison
atlas_slice = atlas_volume[73, :, :]
sample = tiff.imread("resources/sample_hipp.tif")

# Define moving image resolution
moving_res = (23.0, 23.0, 23.0)  # z, y, x in microns

metrics = ["ncc", "ssim", "rmse", "mi"]
for metric in metrics:
    score = compute_similarity_metric(
        sample, atlas_slice, metric, moving_res, atlas_res
    )
    print(f"{metric.upper()} similarity: {score}")
