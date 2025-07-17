import numpy as np
import pytest
from brainglobe_atlasapi import BrainGlobeAtlas

from brainglobe_registration.automated_target_selection import (
    registration_objective,
    run_bayesian_generator,
    similarity_only_objective,
)
from brainglobe_registration.utils.transforms import (
    create_rotation_matrix,
    rotate_volume,
    scale_moving_image,
)

ROLL = 2.10
YAW = -0.30
PITCH = 1.20
TRUE_SLICE = 43
SAMPLE_RES = [25.0, 25.0, 25.0]


@pytest.fixture(scope="module")
def atlas_and_sample():
    atlas = BrainGlobeAtlas("allen_mouse_100um")
    atlas_volume = atlas.reference
    atlas_res = atlas.resolution

    rot_matrix, bounding_box = create_rotation_matrix(
        roll=ROLL, yaw=YAW, pitch=PITCH, img_shape=atlas_volume.shape
    )
    rotated_volume = rotate_volume(
        atlas_volume, atlas_volume.shape, rot_matrix, bounding_box
    )
    sample = rotated_volume[TRUE_SLICE].compute()

    scaled_sample = scale_moving_image(
        sample, atlas_res=atlas_res, moving_res=SAMPLE_RES
    )

    return atlas_volume, scaled_sample, TRUE_SLICE


def test_registration_objective_valid_input(atlas_and_sample):
    """
    Test that registration_objective returns a valid
    positive score for correct inputs.
    """
    atlas_volume, sample, slice_idx = atlas_and_sample
    score = registration_objective(
        pitch=PITCH,
        yaw=YAW,
        z_slice=slice_idx,
        atlas_volume=atlas_volume,
        sample=sample,
    )
    assert isinstance(score, float)
    assert 0.0 < score <= 1.0


def test_registration_objective_invalid_slice_index(atlas_and_sample):
    """
    Test that registration_objective raises IndexError
    for invalid z-slice index.
    """
    atlas_volume, sample, _ = atlas_and_sample
    with pytest.raises(IndexError):
        registration_objective(
            pitch=0,
            yaw=0,
            z_slice=-1,
            atlas_volume=atlas_volume,
            sample=sample,
        )


def test_similarity_only_objective_returns_valid_score(atlas_and_sample):
    """
    Test that similarity_only_objective returns a positive similarity score.
    """
    atlas_volume, sample, slice_idx = atlas_and_sample
    target_slice = atlas_volume[slice_idx]
    score = similarity_only_objective(
        roll=0.0, target_slice=target_slice, sample=sample
    )
    assert isinstance(score, float)
    assert 0.0 < score <= 1.0


def test_run_bayesian_generator_returns_reasonable_z_slice(atlas_and_sample):
    """
    Test that run_bayesian_generator returns a z-slice
    close to the known true slice.
    """
    atlas_volume, sample, true_slice = atlas_and_sample
    gen = run_bayesian_generator(
        atlas_volume=atlas_volume,
        sample=sample,
        manual_z_range=(30, 50),
        init_points=3,
        n_iter=5,
    )

    try:
        while True:
            next(gen)
    except StopIteration as stop:
        final_result = stop.value

    assert final_result["done"] is True

    # Confirm best_z_slice is close to true
    best_z = final_result["best_z_slice"]
    assert abs(best_z - true_slice) <= 10


def test_registration_objective_nan_score_handling(
    monkeypatch, atlas_and_sample
):
    """
    Test that registration_objective returns -1 when similarity metric is NaN.
    """
    atlas_volume, sample, slice_idx = atlas_and_sample

    import brainglobe_registration.automated_target_selection as reg

    monkeypatch.setattr(
        reg,
        "compute_similarity_metric",
        lambda *args, **kwargs: float("nan"),
    )

    score = reg.registration_objective(
        pitch=1,
        yaw=0,
        z_slice=slice_idx,
        atlas_volume=atlas_volume,
        sample=sample,
    )
    assert score == -1.0


def test_registration_objective_rejects_non_2d_sample(atlas_and_sample):
    """
    Test that registration_objective returns -1.0 for invalid 3D sample input.
    """
    atlas_volume, sample, slice_idx = atlas_and_sample
    bad_sample = np.stack([sample, sample])  # Shape (2, H, W)

    score = registration_objective(
        pitch=0,
        yaw=0,
        z_slice=slice_idx,
        atlas_volume=atlas_volume,
        sample=bad_sample,
    )
    assert score == -1.0


def test_run_bayesian_generator_yield_count(atlas_and_sample):
    """
    Test that run_bayesian_generator yields the expected
    number of intermediate results (not including the final return).
    """
    atlas_volume, sample, _ = atlas_and_sample
    init_points = 3
    n_iter = 5
    gen = run_bayesian_generator(
        atlas_volume=atlas_volume,
        sample=sample,
        manual_z_range=(30, 50),
        init_points=init_points,
        n_iter=n_iter,
    )

    results = list(gen)
    assert len(results) == 2 * (init_points + n_iter)
