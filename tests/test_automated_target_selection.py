import numpy as np
from brainglobe_atlasapi import BrainGlobeAtlas

from brainglobe_registration.automated_target_selection import (
    compute_similarity_metric,
    normalise_image,
    pad_to_match_shape,
    prepare_images,
    registration_objective,
    safe_ncc,
    scale_moving_image,
)

atlas = np.random.rand(512, 512).astype(np.float32)
moving = np.random.rand(256, 256).astype(np.float32)
scale = [2.0, 2.0, 2.0]
atlas_res = (25.0, 25.0, 25.0)


def test_pad_to_match_shape():
    small = np.ones((2, 3))
    target = (4, 6)
    padded = pad_to_match_shape(small, target)
    assert padded.shape == target
    # Check padding is symmetric
    top = (target[0] - 2) // 2
    left = (target[1] - 3) // 2
    assert np.all(padded[top : top + 2, left : left + 3] == 1)
    # Check zeros in padding areas
    assert np.all(padded[:top, :] == 0)
    assert np.all(padded[:, :left] == 0)


def test_pad_to_match_shape_same_shape():
    arr = np.ones((4, 6))
    padded = pad_to_match_shape(arr, (4, 6))
    assert padded.shape == (4, 6)
    assert np.all(padded == 1)


def test_normalise_image():
    img = np.random.rand(100, 100) * 100
    normed = normalise_image(img)
    assert np.isclose(np.min(normed), 0.0, atol=1e-6)
    assert np.isclose(np.max(normed), 1.0, atol=1e-6)


def test_normalise_image_constant_input():
    img = np.ones((10, 10)) * 42
    normed = normalise_image(img)
    assert np.all(normed == 0.0)


def test_scale_moving_image():
    img = np.random.rand(100, 100)
    scaled = scale_moving_image(img, (25.0, 25.0, 25.0), 2.0, 2.0, 2.0)
    assert scaled.ndim == 2
    assert isinstance(scaled, np.ndarray)


def test_scale_moving_image_invalid_scale():
    img = np.random.rand(50, 50)
    try:
        _ = scale_moving_image(img, (25.0, 25.0, 25.0), 0, 0, 0)
        assert False, "Expected failure due to zero scale"
    except Exception:
        pass


def test_prepare_images():
    fixed = np.random.rand(512, 512)
    moving = np.random.rand(256, 256)
    moving_img, fixed_img = prepare_images(moving, fixed, atlas_res, scale)
    assert moving_img.shape == fixed_img.shape
    assert moving_img.dtype == np.float32


def test_safe_ncc():
    img1 = np.random.rand(256, 256)
    img2 = img1 + np.random.normal(0, 0.1, (256, 256))
    score = safe_ncc(img1, img2)
    assert isinstance(score, float)
    assert -1.0 <= score <= 1.0


def test_safe_ncc_mismatched_shapes():
    img1 = np.ones((10, 10))
    img2 = np.ones((12, 12))
    try:
        _ = safe_ncc(img1, img2)
        assert False, "Expected an error for mismatched shapes"
    except ValueError:
        pass


def test_compute_similarity_metric():
    moving = np.random.rand(256, 256)
    fixed = moving + np.random.normal(0, 0.05, (256, 256))
    score = compute_similarity_metric(
        moving=moving,
        fixed=fixed,
        atlas_res=atlas_res,
        atlas_to_moving_scale=scale,
    )
    assert isinstance(score, float)
    assert -1.0 <= score <= 1.0


def test_registration_objective_dummy():
    atlas_obj = BrainGlobeAtlas("allen_mouse_100um")
    volume = atlas_obj.reference
    dummy_sample = volume[50]  # A slice
    score = registration_objective(
        pitch=0,
        yaw=0,
        roll=0,
        z_slice=50,
        atlas_volume=volume,
        sample=dummy_sample,
        atlas_to_moving_scale=[25.0, 25.0, 25.0],
        atlas_res=atlas_obj.resolution,
    )
    assert isinstance(score, float)
    assert -1.0 <= score <= 1.0 or np.isnan(score)


def test_registration_objective_out_of_bounds():
    atlas_obj = BrainGlobeAtlas("allen_mouse_100um")
    volume = atlas_obj.reference
    dummy_sample = volume[0]
    try:
        _ = registration_objective(
            pitch=0,
            yaw=0,
            roll=0,
            z_slice=9999,  # clearly too large
            atlas_volume=volume,
            sample=dummy_sample,
            atlas_to_moving_scale=[25.0, 25.0, 25.0],
            atlas_res=atlas_obj.resolution,
        )
        assert False, "Expected failure for out-of-bounds z-slice"
    except IndexError:
        pass
