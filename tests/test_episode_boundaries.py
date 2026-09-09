import numpy as np
from real_robot_data_retime.timeline.episode import assign_boundaries


def test_multiple_actions_have_distinct_approaches():
    xy = np.zeros((140, 2, 2))
    xy[10:40, 0, 0] = np.arange(30)
    xy[40:60, 0, 0] = 29
    xy[60:100, 0, 0] = 29 + np.arange(40)
    xy[100:, 0, 0] = 68
    events = [
        dict(robot_id="left", grasp_start=20, pickup_frame=25, release_frame=35),
        dict(robot_id="left", grasp_start=70, pickup_frame=75, release_frame=90),
    ]
    a, b = assign_boundaries(events, xy, 30, 100)
    assert a["approach_start"] < 20
    assert a["retract_end"] < b["approach_start"]
    assert b["approach_start"] >= 58
