import numpy as np

from brainglobe_registration.similarity_metrics import (
    compute_similarity_metric,
    normalise_image,
    pad_to_match_shape,
    prepare_images,
    safe_ncc,
    scale_moving_image,
)


def test_pad_to_match_shape():
    """
    Test that padding correctly centers a smaller array
    within a larger one, with zero fill.
    """
    small = np.ones((2, 3))
    large = np.zeros((4, 6))
    padded_small, large_unchanged = pad_to_match_shape(
        small, large, mode="constant"
    )

    assert padded_small.shape == large.shape
    assert large_unchanged.shape == large.shape

    top = (4 - 2) // 2
    left = (6 - 3) // 2
    assert np.all(padded_small[top : top + 2, left : left + 3] == 1)
    assert np.all(padded_small[:top, :] == 0)
    assert np.all(padded_small[:, :left] == 0)


def test_pad_to_match_shape_same_shape():
    """
    Test that arrays of equal shape remain unchanged after padding.
    """
    arr1 = np.ones((4, 6))
    arr2 = np.zeros((4, 6))
    padded1, padded2 = pad_to_match_shape(arr1, arr2, mode="constant")
    assert np.all(padded1 == arr1)
    assert np.all(padded2 == arr2)


def test_normalise_image():
    """
    Test that image is normalised to the range [0, 1].
    """
    img = np.random.rand(100, 100) * 100
    normed = normalise_image(img)
    assert np.isclose(np.min(normed), 0.0, atol=1e-6)
    assert np.isclose(np.max(normed), 1.0, atol=1e-6)


def test_normalise_image_constant_input():
    """
    Test that normalisation of constant input returns all zeros.
    """
    img = np.ones((10, 10)) * 42
    normed = normalise_image(img)
    assert np.allclose(normed, 0.0)


def test_scale_moving_image():
    """
    Test that the moving image is successfully scaled to atlas resolution.
    """
    img = np.random.rand(100, 100)
    scaled = scale_moving_image(
        img, atlas_res=(25.0, 25.0, 25.0), moving_res=(2.0, 2.0, 2.0)
    )
    assert scaled.ndim == 2
    assert isinstance(scaled, np.ndarray)


def test_scale_moving_image_invalid_scale():
    """
    Test that scaling with invalid resolution raises ValueError.
    """
    img = np.random.rand(50, 50)
    try:
        _ = scale_moving_image(
            img, atlas_res=(25.0, 25.0, 25.0), moving_res=(0, 0, 0)
        )
        assert False, "Expected ValueError for zero scale"
    except ValueError:
        pass


def test_prepare_images():
    """
    Test that moving and fixed images are padded and
    normalised to the same shape and type.
    """
    fixed = np.random.rand(512, 512)
    moving = np.random.rand(256, 256)
    moving_img, fixed_img = prepare_images(moving, fixed)
    assert moving_img.shape == fixed_img.shape
    assert moving_img.dtype == np.float32
    assert fixed_img.dtype == np.float32


def test_safe_ncc():
    """
    Test that safe_ncc returns a valid correlation score between -1 and 1.
    """
    img1 = np.random.rand(256, 256)
    img2 = img1 + np.random.normal(0, 0.1, (256, 256))
    score = safe_ncc(img1, img2)
    assert isinstance(score, float)
    assert -1.0 <= score <= 1.0


def test_safe_ncc_mismatched_shapes():
    """
    Test that safe_ncc raises ValueError on shape mismatch.
    """
    img1 = np.ones((10, 10))
    img2 = np.ones((12, 12))
    try:
        _ = safe_ncc(img1, img2)
        assert False, "Expected ValueError for mismatched shapes"
    except ValueError:
        pass


def test_compute_similarity_metric_each():
    """
    Test that each similarity metric returns a valid float score.
    """
    moving = np.random.rand(256, 256)
    fixed = moving + np.random.normal(0, 0.05, (256, 256))

    for metric in ["mi", "ncc", "ssim", "combined"]:
        score = compute_similarity_metric(moving, fixed, metric=metric)
        assert isinstance(score, float)
