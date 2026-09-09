import cv2
import numpy as np
from real_robot_data_retime.interaction.discovery import task_object_proposals


def test_reflected_cube_color_is_not_discarded_by_extra_saturation_threshold():
    hsv = np.zeros((220, 400, 3), np.uint8)
    hsv[10:90, 210:310] = [0, 255, 150]
    hsv[155:175, 130:150] = [97, 105, 151]
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    proposals = task_object_proposals(np.repeat(frame[None], 8, axis=0), "drawer")
    assert len(proposals) == 1
    assert 85 < proposals[0]["color"][1] < 110
    assert np.linalg.norm(proposals[0]["origin"] - [139.5, 164.5]) < 1
