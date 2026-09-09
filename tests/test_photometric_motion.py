import cv2
import numpy as np

from real_robot_data_retime.interaction.photometric_motion import photometric_motion


def test_global_exposure_is_not_robot_motion_but_real_arm_remains():
    rng = np.random.default_rng(4)
    # A textured tabletop permits independent per-channel exposure fitting.
    background = rng.integers(120, 210, (120, 200, 3), dtype=np.uint8)
    background = cv2.GaussianBlur(background, (5, 5), 0)
    frames = np.repeat(background[None], 20, axis=0)
    for t in range(8, 20):
        frames[t] = np.clip(frames[t].astype(float) * 0.7, 0, 255).astype(np.uint8)
        frames[t, 65:95, :65] = 25
    original = frames.copy()
    evidence = photometric_motion(frames)
    raw, corrected, fits = evidence.raw, evidence.discovery, evidence.coefficients
    np.testing.assert_array_equal(frames, original)
    assert fits.shape == (20, 3, 2)
    # True dark foreground entering the workspace still supports the left arm.
    assert (corrected["masks"][12, 70:90, :60] == 1).mean() > 0.8
    # Global brightness change cannot create an arm on the opposite edge.
    assert (corrected["masks"][12, 65:100, 160:] > 0).mean() < 0.05
    assert np.count_nonzero(corrected["masks"][12]) < np.count_nonzero(raw["masks"][12])


def test_uncovering_dark_table_is_not_a_new_robot():
    from real_robot_data_retime.interaction.discovery import motion_and_grippers

    frames = np.full((20, 120, 200, 3), 90, np.uint8)
    frames[:8, 65:100, :75] = 20
    absolute = motion_and_grippers(frames)
    from real_robot_data_retime.interaction.photometric_motion import (
        opaque_motion_support,
    )

    directional, _ = opaque_motion_support(frames, frames[0])
    assert np.count_nonzero(absolute["masks"][12]) > 100
    assert not directional[12].any()


def test_darkening_cue_still_detects_a_new_real_arm():

    frames = np.full((20, 120, 200, 3), 90, np.uint8)
    frames[8:, 65:100, :75] = 20
    from real_robot_data_retime.interaction.photometric_motion import (
        opaque_motion_support,
    )

    result, _ = opaque_motion_support(frames, frames[0])
    assert result[12, 70:95, :70].mean() > 0.9


def test_shadow_like_attenuation_is_ambiguous_not_positive_robot_evidence():
    from real_robot_data_retime.interaction.photometric_motion import (
        opaque_motion_support,
    )

    background = np.full((80, 100, 3), 150, np.uint8)
    frame = background.copy()
    frame[30:60, :35] = 20
    frame[30:60, 35:75] = 90
    positive, ambiguous = opaque_motion_support(frame[None], background)
    assert positive[0, 35:55, :30].all()
    assert not positive[0, 35:55, 40:70].any()
    assert ambiguous[0, 35:55, 40:70].all()


def test_supported_audit_excludes_ambiguous_region_but_rejects_missing_robot():
    from real_robot_data_retime.interaction.robot_discovery import robot_mask_audit

    labels = np.zeros((20, 100, 200), np.uint8)
    labels[:, 50:80, :130] = 1
    labels[:, 80:95, 150:] = 2
    robots = np.stack([labels == 1, labels == 2], axis=1)
    robots[:, 0, :, 60:] = False
    support = robots.any(axis=1)
    geometry = {"masks": labels}
    packed = np.packbits(robots, axis=-1)
    assert not robot_mask_audit(packed, geometry, 30)["passed"]
    assert robot_mask_audit(packed, geometry, 30, pixel_support=support)["passed"]
    robots[:, 0, :, 25:] = False
    broken = robot_mask_audit(
        np.packbits(robots, axis=-1), geometry, 30, pixel_support=support
    )
    assert not broken["passed"]
    unresolved = robot_mask_audit(
        packed, geometry, 30, pixel_support=np.zeros_like(support)
    )
    assert not unresolved["passed"]
    assert unresolved["unassessed_frame_arm_pairs"] == 40
