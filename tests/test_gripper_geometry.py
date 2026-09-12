import cv2
import numpy as np

from real_robot_data_retime.interaction.gripper_geometry import end_effector


def test_distal_hand_is_not_thin_cable_or_rightmost_wrist():
    mask = np.zeros((180, 240), np.uint8)
    cv2.rectangle(mask, (0, 15), (160, 40), 1, -1)
    cv2.rectangle(mask, (110, 30), (140, 125), 1, -1)
    cv2.line(mask, (80, 30), (210, 170), 1, 2)
    center, _spread, local = end_effector(mask.astype(bool), 0)
    assert center[1] > 70
    assert center[0] < 160
    assert local.sum() > 0


def test_reflective_collar_gap_does_not_make_wrist_the_end_effector():
    mask = np.zeros((120, 200), bool)
    mask[30:65, :140] = True
    mask[40:95, 144:164] = True
    mask[85:103, 151:173] = True
    center, _spread, local = end_effector(mask, 0)
    assert center[0] > 145 and center[1] > 75
    assert not local[~mask].any()


def test_static_background_rail_cannot_become_a_motion_supported_tip():
    from real_robot_data_retime.interaction.gripper_geometry import (
        motion_supported_grippers,
    )

    h, w = 100, 160
    robot = np.zeros((h, w), bool)
    robot[20:27, :] = True
    robot[20:65, 150:] = True
    robot[50:65, 70:] = True
    robots = np.zeros((1, 2, h, w), bool)
    robots[0, 1] = robot
    motion = np.zeros((1, h, w), np.uint8)
    motion[0, 50:65, 70:] = 2
    grippers = {
        "centers": np.array([[[np.nan, np.nan], [8.0, 23.0]]]),
        "apertures": np.zeros((1, 2)),
        "masks": robots.copy(),
        "robot_masks": np.packbits(robots, axis=-1),
    }
    result = motion_supported_grippers(grippers, {"masks": motion}, (h, w))
    assert result["centers"][0, 1, 0] > 50
    assert result["centers"][0, 1, 1] > 40
    assert not np.any(result["masks"] & ~robots)
