import numpy as np
from real_robot_data_retime.interaction.evidence import score_hypothesis


def test_closure_without_object_motion_is_not_grasp():
    n = 100
    obj = np.zeros((n, 2))
    grip = obj.copy()
    aperture = np.r_[np.full(30, 20), np.full(70, 2)]
    result = score_hypothesis(obj, grip, aperture, 30, 100)
    assert not result["accepted"]
    assert "no_stable_correlated_pickup" in result["rejection_reasons"]


def test_full_future_evidence_with_release():
    n = 100
    obj = np.zeros((n, 2))
    obj[30:65, 0] = np.arange(35)
    obj[65:, 0] = 34
    grip = obj.copy()
    grip[65:, 0] = 34 + np.arange(35) * 2
    aperture = np.r_[np.full(30, 20), np.full(35, 2), np.full(35, 20)]
    result = score_hypothesis(obj, grip, aperture, 30, 100)
    assert result["accepted"], result
    assert 30 <= result["pickup_frame"] <= 32
    assert 65 <= result["release_frame"] <= 67


def test_occlusion_cannot_be_counted_as_correlated_motion():
    n = 100
    obj = np.zeros((n, 2))
    obj[30:70] = np.nan
    result = score_hypothesis(
        obj, np.zeros((n, 2)), np.r_[np.full(30, 20), np.full(70, 2)], 30, 100
    )
    assert not result["accepted"]
    assert "insufficient_future_visibility" in result["rejection_reasons"]


def test_release_accepts_gradual_opening_with_stationary_object():
    n = 100
    obj = np.zeros((n, 2))
    obj[30:65, 0] = np.arange(35)
    obj[65:, 0] = 34
    grip = obj.copy()
    grip[65:, 0] = 34 + np.arange(35) * 2
    aperture = np.r_[
        np.full(30, 20), np.full(30, 2), np.linspace(2, 20, 20), np.full(20, 20)
    ]
    result = score_hypothesis(obj, grip, aperture, 30, 100)
    assert result["accepted"], result
    assert result["release_frame"] is not None


def test_tiny_aperture_change_does_not_override_observed_manipulation():
    n = 100
    obj = np.zeros((n, 2))
    obj[30:65, 0] = np.arange(35)
    obj[65:, 0] = 34
    grip = obj.copy()
    grip[65:, 0] = 34 + np.arange(35) * 2
    result = score_hypothesis(obj, grip, np.full(n, 49.4), 30, 100)
    assert result["accepted"], result
    assert not result["release_opening_observed"]
    assert not result["closure_observed"]


def test_verified_destination_overrides_early_apparent_release():
    obj = np.zeros((120, 2))
    obj[30:65, 0] = np.arange(35)
    obj[65:, 0] = 34
    grip = obj.copy()
    grip[65:, 0] += np.arange(55) * 2
    aperture = np.full(120, 10)
    result = score_hypothesis(
        obj,
        grip,
        aperture,
        30,
        100,
        release_evidence=dict(verified=True, release_frame=95),
    )
    assert result["accepted"]
    assert result["release_frame"] == 95
