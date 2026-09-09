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


def test_nearby_motion_does_not_imply_attachment_without_matching_gripper_motion():
    from real_robot_data_retime.interaction.evidence import score_hypothesis

    n = 100
    obj = np.zeros((n, 2))
    obj[:, 0] = 50
    obj[40:55, 0] += np.arange(15)
    obj[55:, 0] += 14
    for direction in [0, -1]:
        grip = np.zeros((n, 2))
        grip[:, 0] = 50
        grip[40:55, 0] += direction * np.arange(15)
        grip[55:, 0] += direction * 14
        result = score_hypothesis(obj, grip, np.full(n, 10.0), 40, 200)
        assert result["pickup_frame"] is None


def test_release_confirmation_after_occluded_withdrawal():
    obj = np.zeros((100, 2))
    obj[30:65, 0] = np.arange(35)
    obj[65:, 0] = 34
    grip = obj.copy()
    grip[65:70, 0] += np.arange(5) * 5
    grip[70:, 0] += 20
    obj[65:75] = np.nan
    confirmed = np.full(100, -1)
    confirmed[75:] = np.arange(75, 100)
    result = score_hypothesis(
        obj, grip, np.full(100, 10), 30, 100, release_confirmation=confirmed
    )
    assert result["accepted"], result
    assert result["release_frame"] == 75
    assert result["release_confirmation_frame"] == 75
