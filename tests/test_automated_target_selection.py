import numpy as np
import pytest
from brainglobe_atlasapi import BrainGlobeAtlas

from brainglobe_registration.automated_target_selection import (
    registration_objective,
    run_bayesian_generator,
    similarity_only_objective,
)
from brainglobe_registration.utils.plane_sampling import (
    build_rotation_matrix,
    sample_plane,
)
from brainglobe_registration.utils.transforms import (
    scale_moving_image,
)

ROLL = 2.10
YAW = -0.30
PITCH = 1.20
TRUE_SLICE = 43
SAMPLE_RES = [100.0, 100.0, 100.0]


@pytest.fixture(scope="module")
def atlas_and_sample():
    atlas = BrainGlobeAtlas("allen_mouse_100um")
    atlas_volume = atlas.reference
    atlas_res = atlas.resolution

    rot_matrix = build_rotation_matrix(roll=ROLL, yaw=YAW, pitch=PITCH)
    sample = sample_plane(
        atlas_volume,
        z_index=float(TRUE_SLICE),
        rotation_matrix=rot_matrix,
        interpolation_order=1,
        mode="nearest",
    )

    scaled_sample = scale_moving_image(
        sample, atlas_res=atlas_res, moving_res=SAMPLE_RES
    )

    return atlas_volume, scaled_sample, TRUE_SLICE


def test_registration_objective_valid_input_mi(atlas_and_sample):
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
        metric="mi",
    )
    assert isinstance(score, float)
    assert score > 0.8


def test_registration_objective_valid_input_ssim(atlas_and_sample):
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
        metric="ssim",
    )
    assert isinstance(score, float)
    assert 0.8 < score <= 1.0


def test_registration_objective_valid_input_ncc(atlas_and_sample):
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
        metric="ncc",
    )
    assert isinstance(score, float)
    assert 0.8 < score <= 1.0


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
        roll=ROLL, target_slice=target_slice, sample=sample
    )
    assert isinstance(score, float)
    assert score > 0.8


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


class TestPlaneSamplingIntegration:
    """Tests verifying plane_sampling integration in the automated pipeline."""

    def test_sample_plane_produces_valid_slice(self):
        """sample_plane with identity rotation should match direct indexing."""
        volume = np.random.default_rng(42).random((20, 30, 40))
        rot = build_rotation_matrix(0, 0, 0)
        result = sample_plane(
            volume,
            z_index=10.0,
            rotation_matrix=rot,
            interpolation_order=0,
        )
        expected = volume[10, :, :]
        np.testing.assert_array_almost_equal(result, expected)

    def test_sample_plane_with_small_rotation(self):
        """Small rotation should produce a non-identical but similar slice."""
        rng = np.random.default_rng(42)
        volume = rng.random((20, 30, 40))
        identity_slice = sample_plane(
            volume,
            z_index=10.0,
            rotation_matrix=build_rotation_matrix(0, 0, 0),
            interpolation_order=1,
            mode="nearest",
        )
        rotated_slice = sample_plane(
            volume,
            z_index=10.0,
            rotation_matrix=build_rotation_matrix(0, 2.0, 1.5),
            interpolation_order=1,
            mode="nearest",
        )
        # Should be similar but not identical
        assert rotated_slice.shape == identity_slice.shape
        assert not np.allclose(identity_slice, rotated_slice)
        # But correlated (small rotation)
        corr = np.corrcoef(identity_slice.ravel(), rotated_slice.ravel())[0, 1]
        assert corr > 0.75

    def test_nearest_mode_no_black_borders(self):
        """mode='nearest' should not produce zero-fill at edges."""
        rng = np.random.default_rng(42)
        volume = rng.random((20, 30, 40)) + 0.1  # ensure no true zeros
        rot = build_rotation_matrix(0, 5.0, 5.0)
        result = sample_plane(
            volume,
            z_index=10.0,
            rotation_matrix=rot,
            interpolation_order=1,
            mode="nearest",
        )
        # With nearest mode and a volume with no zeros,
        # the result should have no zeros
        assert np.all(result > 0)

    def test_registration_objective_uses_plane_sampling(
        self, monkeypatch, atlas_and_sample
    ):
        """Verify registration_objective internally calls sample_plane."""
        atlas_volume, sample, slice_idx = atlas_and_sample
        calls = []

        import brainglobe_registration.automated_target_selection as reg

        original_sample_plane = reg.sample_plane

        def tracking_sample_plane(*args, **kwargs):
            calls.append(True)
            return original_sample_plane(*args, **kwargs)

        monkeypatch.setattr(reg, "sample_plane", tracking_sample_plane)

        reg.registration_objective(
            pitch=PITCH,
            yaw=YAW,
            z_slice=slice_idx,
            atlas_volume=atlas_volume,
            sample=sample,
        )
        assert len(calls) == 1, "sample_plane should be called exactly once"
