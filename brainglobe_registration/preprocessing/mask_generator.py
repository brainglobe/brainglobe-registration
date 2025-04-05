import numpy as np
import tifffile as tiff
import os
from skimage.transform import resize  


def load_image(image_path):
    """Load a TIFF image as numpy array."""
    return tiff.imread(image_path)


def save_image(image, output_path):
    """Save numpy array as TIFF image."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tiff.imwrite(output_path, image.astype(np.uint16))  # uint16 for compatibility


def generate_mask(annotation_image):
    """
    Generate a binary mask from annotation image.
    Pixels with value != 1 are included in the mask.
    """
    mask = annotation_image != 1
    return mask.astype(np.uint8)  # 0 or 1 mask


def apply_mask(image, mask):
    """
    Apply binary mask to image. Pixels outside mask are set to 0.
    Resizes the mask to match the image dimensions if necessary.
    """
    if image.shape != mask.shape:
        print("Resizing mask to match image dimensions...")
        mask = resize(mask, image.shape, order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)
    masked_image = image * mask
    return masked_image


def mask_atlas_with_annotation(atlas_path, annotation_path, output_path):
    """
    Full pipeline:
    1. Load atlas and annotation images.
    2. Generate mask.
    3. Apply mask.
    4. Save the masked atlas image.
    """
    print(f"Loading atlas image: {atlas_path}")
    atlas_image = load_image(atlas_path)

    print(f"Loading annotation image: {annotation_path}")
    annotation_image = load_image(annotation_path)

    print("Generating binary mask...")
    mask = generate_mask(annotation_image)

    print("Applying mask to atlas image...")
    masked_atlas = apply_mask(atlas_image, mask)

    print(f"Saving masked atlas image to: {output_path}")
    save_image(masked_atlas, output_path)

    print("Masking completed successfully.")

    return masked_atlas



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Mask atlas image using annotation image.")
    parser.add_argument("--atlas", required=True, help="Path to the atlas image (TIFF).")
    parser.add_argument("--annotation", required=True, help="Path to the annotation image (TIFF).")
    parser.add_argument("--output", required=True, help="Path to save the masked atlas image.")

    args = parser.parse_args()

    mask_atlas_with_annotation(args.atlas, args.annotation, args.output)

