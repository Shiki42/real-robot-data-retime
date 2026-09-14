import numpy as np
import pytest

from real_robot_data_retime.compositing.screw import ScrewStageCompositor


def occluded_stage():
    frames = np.full((3, 60, 80, 3), 210, np.uint8)
    masks = np.zeros((3, 2, 60, 80), bool)
    masks[:, 0, 25:40, 5:40] = True
    masks[0, 0, :, 25:] = False
    masks[0, 1, 20:45, 25:60] = True
    masks[1:, 1, 20:45, 60:78] = True
    for t in range(3):
        frames[t][masks[t, 0]] = 30
        frames[t][masks[t, 1]] = 90
    return ScrewStageCompositor(frames, masks, 40, []), frames


def test_real_same_pose_foreground_replaces_background_hole_only_for_its_arm():
    comp, frames = occluded_stage()
    original = frames.copy()
    comp.restore_foreground(0, 0, 1, 1, np.zeros((3, 7)))
    fixed = comp.frame(0, 1)
    np.testing.assert_array_equal(fixed[30, 32], [30, 30, 30])
    np.testing.assert_array_equal(comp.frame(0, 0), original[0])
    np.testing.assert_array_equal(comp.frame(1, 0)[30, 32], [90, 90, 90])
    np.testing.assert_array_equal(frames, original)
    assert comp.foreground_references[0]["repairs"][0]["pixels"] > 0


def test_fractional_foreground_uses_repaired_pixels_for_optical_flow():
    comp, _ = occluded_stage()
    comp.restore_foreground(0, 0, 1, 1, np.zeros((3, 7)))
    comp.frame(0.5, 1)
    pair = comp.foreground_pairs[(0, 0)]
    np.testing.assert_array_equal(pair.frames[0, 30, 32], [30, 30, 30])
    np.testing.assert_array_equal(comp.frames[0, 30, 32], [90, 90, 90])


def test_foreground_donor_with_different_measured_pose_is_rejected():
    comp, _ = occluded_stage()
    state = np.zeros((3, 7))
    state[1, 0] = 1
    with pytest.raises(ValueError, match="measured arm pose"):
        comp.restore_foreground(0, 0, 1, 1, state)


def test_foreground_registration_recovers_a_known_small_shift():
    import cv2

    from real_robot_data_retime.compositing.foreground_reference import (
        register_foreground,
    )

    rng = np.random.default_rng(8)
    target = rng.integers(0, 256, (100, 120, 3), dtype=np.uint8)
    mask = np.zeros((100, 120), np.uint8)
    mask[15:85, 15:105] = 1
    transform = np.float32([[1, 0, 3], [0, 1, -2]])
    donor = cv2.warpAffine(target, transform, (120, 100))
    donor_mask = cv2.warpAffine(mask, transform, (120, 100)).astype(bool)
    _, _, evidence = register_foreground(donor, target, donor_mask, mask.astype(bool))
    np.testing.assert_allclose(np.array(evidence["transform"])[:, 2], [-3, 2], atol=0.2)
    assert evidence["max_reprojection_error_px"] < 1


def test_foreground_registration_rejects_unobservable_alignment():
    from real_robot_data_retime.compositing.foreground_reference import (
        register_foreground,
    )

    image = np.zeros((50, 60, 3), np.uint8)
    mask = np.ones((50, 60), bool)
    with pytest.raises(ValueError, match="insufficient"):
        register_foreground(image, image, mask, mask)
