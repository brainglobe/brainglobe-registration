import warnings

import numpy as np
import pytest

from brainglobe_registration.similarity_metrics import (
    compute_similarity_metric,
    normalise_image,
    pad_to_match_shape,
    prepare_images,
    safe_ncc,
)


@pytest.mark.parametrize(
    "mode,constant_values",
    [
        ("constant", 5),
        ("edge", 0),
    ],
)
def test_pad_to_match_shape_shapes_and_values(mode, constant_values):
    """
    Test that padding produces correct output shapes and expected values
    for different padding modes.
    """
    # Create two arrays of different shapes
    moving = np.ones((10, 20))
    fixed = np.ones((16, 12))

    moving_padded, fixed_padded = pad_to_match_shape(
        moving, fixed, mode=mode, constant_values=constant_values
    )

    # Target shape should be the maximum shape of both
    expected_shape = (16, 20)
    assert moving_padded.shape == expected_shape
    assert fixed_padded.shape == expected_shape

    # Content checks:
    if mode == "constant":
        # Check padding value is correct at corners
        assert moving_padded[0, 0] == constant_values
        assert fixed_padded[0, 0] == constant_values
    elif mode == "edge":
        # Should repeat edge value (which is 1)
        assert moving_padded[0, 0] == 1
        assert fixed_padded[0, 0] == 1


def test_pad_to_match_shape_no_padding():
    """
    Test that no padding is applied when input shapes already match.
    """
    # Arrays already the same shape — should return unchanged
    a = np.random.rand(10, 10)
    b = np.random.rand(10, 10)
    a_padded, b_padded = pad_to_match_shape(a, b, mode="constant")

    assert np.allclose(a, a_padded)
    assert np.allclose(b, b_padded)
    assert a_padded.shape == (10, 10)
    assert b_padded.shape == (10, 10)


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


def test_safe_ncc_high_similarity():
    """
    Test that safe_ncc returns a high score (~1.0) for nearly identical images.
    """
    rng = np.random.default_rng(42)
    img1 = rng.random((256, 256))
    img2 = img1 + rng.normal(0, 1e-3, img1.shape)  # very slight noise

    score = safe_ncc(img1, img2)

    assert isinstance(score, float)
    assert -1.0 <= score <= 1.0
    assert (
        score > 0.99
    ), f"NCC score too low for highly similar images: {score}"


def test_safe_ncc_constant_image():
    """Test NCC returns 0.0 when either image is constant."""
    img_random = np.random.rand(100, 100)
    img_constant = np.ones((100, 100))

    # Case 1: img1 constant
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        score1 = safe_ncc(img_constant, img_random)
        assert score1 == 0.0
        assert any("constant" in str(warning.message) for warning in w)

    # Case 2: img2 constant
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        score2 = safe_ncc(img_random, img_constant)
        assert score2 == 0.0
        assert any("constant" in str(warning.message) for warning in w)


def test_safe_ncc_mismatched_shapes():
    """Test that safe_ncc raises ValueError on shape mismatch."""
    img1 = np.ones((10, 10))
    img2 = np.ones((12, 12))
    try:
        _ = safe_ncc(img1, img2)
        assert False, "Expected ValueError for mismatched shapes"
    except ValueError:
        pass


def test_safe_ncc_different_nonconstant():
    """Test NCC is low for different non-constant images."""
    rng = np.random.default_rng(seed=0)
    img1 = rng.normal(loc=0, scale=1, size=(100, 100))
    img2 = rng.normal(loc=10, scale=1, size=(100, 100))
    score = safe_ncc(img1, img2)
    assert -0.2 < score < 0.2


@pytest.mark.parametrize(
    "metric,weights",
    [
        ("mi", None),
        ("ncc", None),
        ("ssim", None),
        ("combined", (0.6, 0.2, 0.2)),
    ],
)
def test_compute_similarity_metric_with_weights(metric, weights):
    """
    Test that each similarity metric returns a float.
    """
    moving = np.random.rand(128, 128)
    fixed = moving + np.random.normal(0, 0.05, (128, 128))

    if metric == "combined":
        score = compute_similarity_metric(
            moving, fixed, metric=metric, weights=weights
        )
    else:
        score = compute_similarity_metric(moving, fixed, metric=metric)

    assert isinstance(score, float)
