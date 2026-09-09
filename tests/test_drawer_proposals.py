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


def test_muted_colored_cube_is_not_retracked_as_dark_robot():
    import cv2
    import numpy as np
    from real_robot_data_retime.interaction.discovery import (
        object_proposals,
        track_candidates,
        InteractionConfig,
    )

    frames = np.full((20, 120, 200, 3), 200, np.uint8)
    color = cv2.cvtColor(np.uint8([[[96, 94, 157]]]), cv2.COLOR_HSV2BGR)[0, 0]
    for t in range(20):
        x = 50 + max(0, t - 8)
        frames[t, 75:95, x : x + 20] = color
        frames[t, 75:95, 90:110] = 25
    proposals = object_proposals(
        frames,
        "saturated",
        InteractionConfig(minimum_object_area=50, maximum_object_area=1000),
    )
    assert len(proposals) == 1 and proposals[0]["color"][1] < 95
    tracks = track_candidates(frames, proposals)
    assert np.linalg.norm(tracks[0]["centers"][-1] - [70.5, 84.5]) < 2


def test_drawer_proposals_exclude_colored_robot_entry_pixels():
    import cv2
    import numpy as np
    from real_robot_data_retime.interaction.discovery import (
        task_object_proposals,
        InteractionConfig,
    )

    frames = np.full((8, 120, 200, 3), 200, np.uint8)
    red = cv2.cvtColor(np.uint8([[[0, 220, 200]]]), cv2.COLOR_HSV2BGR)[0, 0]
    blue = cv2.cvtColor(np.uint8([[[96, 110, 200]]]), cv2.COLOR_HSV2BGR)[0, 0]
    frames[:, 10:40, 130:180] = red
    frames[:, 85:97, 75:87] = blue
    frames[:, 85:97, 15:27] = blue
    proposals = task_object_proposals(
        frames,
        "drawer",
        InteractionConfig(minimum_object_area=50, maximum_object_area=1000),
    )
    assert len(proposals) == 1
    assert 75 <= proposals[0]["origin"][0] <= 87
