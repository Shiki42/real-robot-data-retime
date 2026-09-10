import numpy as np
import pytest

from real_robot_data_retime.interaction.robot_discovery import robot_prompt


def test_robot_prompt_recovers_silhouette_across_overwritten_side_labels():
    frames = np.zeros((2, 100, 200, 3), np.uint8)
    masks = np.zeros((2, 100, 200), np.uint8)
    masks[:, 35:55, :110] = 1
    masks[:, 35:55, 60:110] = 2
    masks[:, 70:90, 150:] = 2
    geometry = dict(prompt_support=masks > 0, masks=masks, centers=np.zeros((2, 2, 2)))
    _, left = robot_prompt(frames, geometry, 0)
    _, right = robot_prompt(frames, geometry, 1)
    assert left["mask"][40, 100]
    assert not left["mask"][80, 170]
    assert right["mask"][80, 170]
    assert not right["mask"][40, 100]


def test_robot_prompt_does_not_merge_touching_opposite_arms():
    frames = np.zeros((2, 100, 200, 3), np.uint8)
    masks = np.zeros((2, 100, 200), np.uint8)
    masks[:, 35:55, :] = 1
    with pytest.raises(ValueError, match="entry-anchored"):
        robot_prompt(frames, dict(prompt_support=masks > 0, masks=masks), 0)


def test_robot_mask_audit_rejects_partial_and_missing_arms():
    from real_robot_data_retime.interaction.robot_discovery import robot_mask_audit

    masks = np.zeros((30, 100, 200), np.uint8)
    masks[:, 35:55, :110] = 1
    masks[:, 70:90, 150:] = 2
    robots = np.stack([masks == 1, masks == 2], axis=1)
    geometry = dict(prompt_support=masks > 0, masks=masks)
    assert robot_mask_audit(np.packbits(robots, axis=-1), geometry, 30)["passed"]
    robots[:, 0, :, 30:] = False
    result = robot_mask_audit(np.packbits(robots, axis=-1), geometry, 30)
    assert not result["passed"]
    assert not result["arms"][0]["passed"]
    assert result["arms"][1]["passed"]


def test_background_person_is_not_required_in_robot_mask():
    from real_robot_data_retime.interaction.robot_discovery import robot_mask_audit

    masks = np.zeros((30, 100, 200), np.uint8)
    masks[:, 60:80, :80] = 1
    masks[:, 70:90, 150:] = 2
    robots = np.stack([masks == 1, masks == 2], axis=1)
    masks[:, 10:35, :90] = 1  # Unrelated motion behind the tabletop.
    assert robot_mask_audit(
        np.packbits(robots, axis=-1), dict(prompt_support=masks > 0, masks=masks), 30
    )["passed"]


def test_robot_prompt_does_not_place_positive_points_on_ambiguous_shadow():
    from real_robot_data_retime.interaction.robot_discovery import (
        supported_robot_prompt,
    )

    region = np.zeros((100, 200), bool)
    region[50:80, :140] = True
    support = np.zeros_like(region)
    support[50:80, :60] = True
    p = supported_robot_prompt(region, support)
    assert p["bbox"][2] > 140  # Context still covers the full candidate region.
    for x, y in p["positive_points"]:
        assert support[round(y), round(x)]
    with pytest.raises(ValueError, match="insufficient reliable"):
        supported_robot_prompt(region, np.zeros_like(region))


def test_released_object_does_not_pull_fingertip_away_from_robot():
    from real_robot_data_retime.interaction.gripper_geometry import end_effector
    from real_robot_data_retime.interaction.neural_tracks import object_free_robot_masks

    robot = np.zeros((100, 160), bool)
    robot[25:45, :95] = True
    robot[25:65, 80:105] = True
    letter = np.zeros_like(robot)
    letter[60:80, 100:125] = True
    contaminated = robot | letter
    packed = np.packbits(contaminated[None, None], axis=-1)
    tracks = [{"packed_masks": np.packbits(letter[None], axis=-1)}]
    clean = np.unpackbits(
        object_free_robot_masks(packed, tracks)[0, 0], axis=-1, count=160
    ).astype(bool)
    measured, _, local = end_effector(clean, 0)
    wrong, _, _ = end_effector(contaminated, 0)
    assert np.linalg.norm(measured - wrong) > 10
    assert measured[0] < 106 and measured[1] < 66
    assert not (local & letter).any()
    assert np.array_equal(packed, np.packbits(contaminated[None, None], axis=-1))
