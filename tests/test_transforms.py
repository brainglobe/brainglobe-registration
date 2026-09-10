import numpy as np
import pytest

from brainglobe_registration.utils.transforms import scale_moving_image


def test_scale_moving_image_correct_shape_2d():
    """
    Test that a 2D moving image is correctly scaled to match atlas resolution.
    """
    img = np.random.rand(100, 200)  # Shape in (y, x)
    moving_res = (1.0, 2.0, 4.0)  # z, y, x (ignored z for 2D)
    atlas_res = (1.0, 1.0, 1.0)  # Target resolution

    # Expected output shape:
    expected_y = int(img.shape[0] * (moving_res[1] / atlas_res[1]))
    expected_x = int(img.shape[1] * (moving_res[2] / atlas_res[2]))

    scaled = scale_moving_image(img, atlas_res, moving_res)

    assert scaled.shape == (
        expected_y,
        expected_x,
    ), f"Expected {(expected_y, expected_x)}, got {scaled.shape}"
    assert isinstance(scaled, np.ndarray)


def test_scale_moving_image_invalid_scale():
    """
    Test that scaling with invalid resolution raises ValueError.
    """
    img = np.random.rand(50, 50)
    with pytest.raises(ValueError, match="Pixel sizes must be greater than 0"):
        _ = scale_moving_image(
            img, atlas_res=(25.0, 25.0, 25.0), moving_res=(0, 0, 0)
        )
