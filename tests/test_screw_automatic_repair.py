import numpy as np
import pytest

from real_robot_data_retime.compositing.automatic_repair import clean_background_masks
from real_robot_data_retime.compositing.foreground_reference import (
    ForegroundRegistrationError,
)
from real_robot_data_retime.compositing.screw import ScrewStageCompositor


def stage():
    frames = np.full((5, 120, 160, 3), 210, np.uint8)
    masks = np.zeros((5, 2, 120, 160), bool)
    masks[:, 0, 40:100, 10:80] = True
    masks[0, 0, :, 55:] = False
    masks[0, 1, 35:105, 55:120] = True
    masks[1:, 1, 35:105, 130:157] = True
    texture = np.random.default_rng(7).integers(15, 170, (120, 160, 3), dtype=np.uint8)
    for t in range(5):
        frames[t][masks[t, 0]] = texture[masks[t, 0]]
        frames[t][masks[t, 1]] = 90
    return ScrewStageCompositor(frames, masks, 80, []), frames


def test_detects_hole_and_selects_reference_without_frame_annotations():
    comp, frames = stage()
    report = comp.repair_automatically(np.zeros((5, 14)), np.array([[0, 1]]))
    assert len(report["applied"]) == 1
    assert report["applied"][0]["reference_frame"] == 1
    assert report["applied"][0]["repairs"][0]["source_frame"] == 0
    assert not report["requires_review"]
    fixed = comp.frame(0, 1)
    assert (
        np.abs(fixed[55:85, 60:75].astype(float) - frames[1, 55:85, 60:75]).mean() < 3
    )


def test_never_repairs_original_synchronized_frames():
    comp, frames = stage()
    report = comp.repair_automatically(np.zeros((5, 14)), np.array([[0, 0], [1, 1]]))
    assert not report["applied"] and not report["unresolved"]
    np.testing.assert_array_equal(comp.frame(0, 0), frames[0])


def test_gripper_excursion_rejects_same_endpoint_with_different_payload():
    comp, _ = stage()
    state = np.zeros((5, 14))
    state[1, 6] = 1
    report = comp.repair_automatically(state, np.array([[0, 2]]))
    assert not report["applied"]
    assert report["requires_review"]


def test_failed_registration_is_reported_and_unexpected_errors_propagate(monkeypatch):
    import real_robot_data_retime.compositing.screw as module

    comp, _ = stage()

    def reject(*args):
        raise ForegroundRegistrationError("not enough shared evidence")

    monkeypatch.setattr(module, "register_foreground", reject)
    report = comp.repair_automatically(np.zeros((5, 14)), np.array([[0, 1]]))
    assert report["requires_review"] and report["rejected_candidates"]

    def crash(*args):
        raise RuntimeError("unexpected implementation failure")

    monkeypatch.setattr(module, "register_foreground", crash)
    with pytest.raises(RuntimeError, match="unexpected"):
        comp.repair_automatically(np.zeros((5, 14)), np.array([[0, 1]]))


def test_textured_background_pollution_removed_but_flat_white_arm_preserved():
    rng = np.random.default_rng(2)
    plate = np.full((70, 90, 3), 230, np.uint8)
    plate[10:40, 10:40] = rng.integers(30, 200, (30, 30, 3), dtype=np.uint8)
    frames = np.repeat(plate[None], 4, axis=0)
    masks = np.zeros((4, 2, 70, 90), bool)
    masks[0, 1, 12:38, 12:38] = True
    masks[0, 0, 45:65, 45:75] = True
    cleaned, receipt = clean_background_masks(
        frames, masks, plate, np.full((70, 90), 3)
    )
    assert not cleaned[0, 1].any()
    np.testing.assert_array_equal(cleaned[0, 0], masks[0, 0])
    assert receipt[0]["removed_background_pixels"] > 0


def test_background_cleanup_requires_repeated_unmasked_observations():
    image = np.random.default_rng(3).integers(0, 255, (40, 50, 3), dtype=np.uint8)
    masks = np.ones((2, 2, 40, 50), bool)
    cleaned, receipt = clean_background_masks(
        np.stack([image, image]), masks, image, np.full((40, 50), 2)
    )
    np.testing.assert_array_equal(cleaned, masks)
    assert not receipt
