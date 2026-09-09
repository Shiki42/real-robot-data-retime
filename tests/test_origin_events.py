import numpy as np
from real_robot_data_retime.interaction.origin_events import origin_departure_interval


def test_transient_occlusion_is_not_a_pickup():
    frames = np.full((60, 60, 80, 3), 180, np.uint8)
    mask = np.zeros((60, 80), bool)
    mask[25:35, 30:40] = True
    frames[:35, mask] = 30
    robots = np.zeros((60, 60, 80), bool)
    robots[10:20, 20:40, 25:45] = True
    robots[30:37, 20:40, 25:45] = True
    frames[10:20, 20:40, 25:45] = 20
    frames[30:37, 20:40, 25:45] = 20
    result, signal = origin_departure_interval(frames, dict(mask=mask), robots, 30)
    assert result["last_observed_at_origin"] == 29
    assert result["first_observed_empty"] >= 37
    assert result["pickup_interval"][0] > 20
