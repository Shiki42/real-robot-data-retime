import numpy as np
import pytest

from real_robot_data_retime.tasks.workpiece import alternating_pickups
from real_robot_data_retime.timeline.visual import plan_visual


def test_alternating_pickups_allow_left_wait_for_slow_right_exit():
    n, h, w = 85, 24, 64
    robots = np.zeros((n, 2, h, w), bool)
    robots[:, 0, 8:12, 2:6] = True
    robots[:, 1, 8:12, 56:60] = True
    # Left visits twice; the first right visit clears slowly. A later left
    # visit must not force the first right arm to wait outside both visits.
    for side, spans in [(0, [(8, 16), (23, 30)]), (1, [(45, 66), (72, 79)])]:
        for start, stop in spans:
            robots[start:stop, side] = False
            robots[start:stop, side, 8:12, 29:33] = True
    events = [
        dict(
            robot_id=side,
            object_id=k,
            approach_start=start,
            pickup_frame=pickup,
            release_frame=pickup + 2,
            retract_end=end,
        )
        for k, (side, start, pickup, end) in enumerate(
            [
                ("left", 0, 12, 22),
                ("left", 17, 26, 38),
                ("right", 40, 49, 70),
                ("right", 67, 75, 84),
            ]
        )
    ]
    segmentation = dict(
        robots=np.packbits(robots, axis=-1),
        objects=np.packbits(np.zeros((4, n, h, w), bool), axis=-1),
    )
    left, right, report = plan_visual(
        dict(task="workpiece", fps=30, episodes=events),
        np.zeros((n, h, w, 3), np.uint8),
        {},
        segmentation,
    )
    assert [x["arm"] for x in report["pickup_order"]] == [
        "left",
        "right",
        "left",
        "right",
    ]
    assert np.all(np.diff([x["output_frame"] for x in report["pickup_order"]]) > 0)
    assert np.any((np.diff(left) == 0) & (left[:-1] < 26) & (right[:-1] >= 49))
    assert np.any((np.diff(right) == 0) & (right[:-1] < 49))
    assert np.all((left < 23) | (right >= 66))
    assert np.all(np.diff(left) >= 0) and np.all(np.diff(right) >= 0)
    assert (left[0], left[-1], right[0], right[-1]) == (0, 38, 40, 84)


def test_incomplete_workpiece_fails_explicitly():
    with pytest.raises(ValueError, match="two pickups per arm"):
        alternating_pickups([dict(robot_id="left", pickup_frame=5)])
