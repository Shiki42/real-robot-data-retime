import cv2, numpy as np
from real_robot_data_retime.interaction.gripper_geometry import end_effector


def test_distal_hand_is_not_thin_cable_or_rightmost_wrist():
    mask = np.zeros((180, 240), np.uint8)
    cv2.rectangle(mask, (0, 15), (160, 40), 1, -1)
    cv2.rectangle(mask, (110, 30), (140, 125), 1, -1)
    cv2.line(mask, (80, 30), (210, 170), 1, 2)
    center, spread, local = end_effector(mask.astype(bool), 0)
    assert center[1] > 70
    assert center[0] < 160
    assert local.sum() > 0
