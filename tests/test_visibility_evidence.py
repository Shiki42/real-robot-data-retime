import numpy as np
from real_robot_data_retime.interaction.verification import attachment_visibility


def test_occlusion_support_never_manufactures_measured_coordinates():
    obj = np.tile([15.0, 15.0], (20, 1))
    obj[5:10] = np.nan
    grip = np.tile([15.0, 15.0], (20, 1))
    masks = np.zeros((20, 30, 30), bool)
    masks[:, 10:21, 10:21] = True
    before = obj.copy()
    result = attachment_visibility(obj, grip, masks, 0, 20, object_radius_px=3)
    assert result["visible_fraction"] == 0.75
    assert result["robot_occluded_fraction"] == 0.25
    assert result["explained_fraction"] == 1.0
    assert np.array_equal(obj, before, equal_nan=True)
    assert (
        attachment_visibility(
            obj, grip, np.zeros_like(masks), 0, 20, object_radius_px=3
        )["explained_fraction"]
        == 0.75
    )


def test_partial_overlap_is_weaker_than_observed_or_fully_occluded_support():
    obj = np.tile([15.0, 15.0], (20, 1))
    obj[5:15] = np.nan
    grip = np.tile([15.0, 15.0], (20, 1))
    masks = np.zeros((20, 30, 30), bool)
    masks[:, 10:21, 18:25] = True
    result = attachment_visibility(obj, grip, masks, 0, 20, object_radius_px=5)
    assert result["visible_fraction"] == 0.5
    assert result["robot_occluded_fraction"] == 0
    assert result["partial_occlusion_consistent_fraction"] == 0.5
    assert result["evidence_confidence"] == 0.75
