import numpy as np
import pytest

from real_robot_data_retime.tasks.workpiece import alternating_pickups
from real_robot_data_retime.timeline.visual import plan_visual
from real_robot_data_retime.timeline.workpiece import approach_clock


def test_execution_owner_cannot_be_slowed_by_the_other_arm():
    n, h, w = 260, 24, 64
    robots = np.zeros((n, 2, h, w), bool)
    robots[:, 0, 8:12, 2:6] = True
    robots[:, 1, 8:12, 56:60] = True
    for side, spans in [(0, [(10, 28), (60, 72)]), (1, [(115, 168), (220, 235)])]:
        for start, stop in spans:
            robots[start:stop, side] = False
            robots[start:stop, side, 8:12, 29:33] = True
    events = [
        dict(
            robot_id=side,
            object_id=k,
            approach_start=start,
            pickup_frame=pickup,
            release_frame=release,
            retract_end=end,
            release_evidence=dict(clearance_frame=release + 1),
        )
        for k, (side, start, pickup, release, end) in enumerate(
            [
                ("left", 0, 20, 40, 60),
                ("left", 41, 65, 85, 95),
                ("right", 100, 125, 175, 220),
                ("right", 176, 225, 245, 259),
            ]
        )
    ]
    left, right, report = plan_visual(
        dict(task="workpiece", fps=30, episodes=events),
        np.zeros((n, h, w, 3), np.uint8),
        {},
        dict(
            robots=np.packbits(robots, axis=-1),
            objects=np.packbits(np.zeros((4, n, h, w), bool), axis=-1),
        ),
    )
    assert [x["arm"] for x in report["pickup_order"]] == [
        "left",
        "right",
        "left",
        "right",
    ]
    assert np.all(np.diff([x["output_frame"] for x in report["pickup_order"]]) > 0)
    # Right's entry and any waits have no effect on the complete first left action.
    np.testing.assert_array_equal(left[:41], np.arange(41))
    assert np.any((np.diff(left) == 0) & (left[:-1] < 65) & (right[:-1] >= 125))
    for clock, start, end in [(left, 65, 95), (right, 125, 175), (right, 225, 259)]:
        active = (clock[:-1] >= start) & (clock[:-1] < end) & (clock[1:] <= end)
        np.testing.assert_allclose(np.diff(clock)[active], 1, atol=1e-9)
    assert np.all((left < 60) | (right >= 168))
    assert report["smoothing"]["method"] == "independent_approach_clocks"


def test_approach_ramps_do_not_retime_an_earlier_execution():
    clock, holds, ramps = approach_clock(0, 100, [60], 30)
    assert clock[holds[0]] == 60
    np.testing.assert_array_equal(clock[:53], np.arange(53))
    assert np.max(np.diff(clock)) <= 1.000000001
    assert np.min(np.diff(clock)) > 0
    assert ramps[0]["brake_source_start"] == 52
    assert clock[-1] == 100


def test_incomplete_workpiece_fails_explicitly():
    with pytest.raises(ValueError, match="two pickups per arm"):
        alternating_pickups([dict(robot_id="left", pickup_frame=5)])
