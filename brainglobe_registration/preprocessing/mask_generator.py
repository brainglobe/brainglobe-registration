import numpy as np
from bg_atlasapi import BrainGlobeAtlas


def generate_mask_from_atlas(atlas: BrainGlobeAtlas) -> np.ndarray:
    """
    Generate a binary mask from the atlas annotation array.

    Parameters:
        atlas (BrainGlobeAtlas): Atlas object containing annotation data.

    Returns:
        np.ndarray: Binary mask of the same shape as the annotation,
                    where pixels are 1 if the annotation is not zero, else 0.
    """
    annotation = atlas.annotation
    mask = annotation != 0
    return mask.astype(np.uint8)


def mask_atlas(atlas: BrainGlobeAtlas) -> np.ndarray:
    """
    Apply the annotation-based mask to the reference image of the atlas.

    Parameters:
        atlas (BrainGlobeAtlas): Atlas object containing reference and
        annotation data.

    Returns:
        np.ndarray: Reference image with mask applied (pixels outside
        mask set to zero).
    """
    mask = generate_mask_from_atlas(atlas)
    masked_reference = atlas.reference * mask
    return masked_reference
