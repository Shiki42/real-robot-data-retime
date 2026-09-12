import numpy as np
import pytest

from real_robot_data_retime.timeline.workpiece_workspace import (
    DEFAULT_WORKSPACE,
    workspace_events,
)


def fixture():
    tcp = np.zeros((2, 100, 3))
    tcp[:, :, 0] = 0.35
    tcp[:, :, 2] = 0.05
    events = []
    for side, sign in [(0, 1), (1, -1)]:
        tcp[side, :, 1] = sign * 0.25
        for cycle, start in enumerate([0, 50]):
            tcp[side, start + 10 : start + 20, 1] = 0
            tcp[side, start + 20 : start + 25, 1] = sign * 0.03
            tcp[side, start + 25 : start + 30, 1] = sign * 0.16
            events.append(
                {
                    "robot_id": ["left", "right"][side],
                    "approach_start": start,
                    "pickup_frame": start + 15,
                    "release_frame": start + 35,
                    "retract_end": start + 45,
                }
            )
    return tcp, events


def test_fixed_box_entry_and_observed_retreat():
    tcp, events = fixture()
    r = workspace_events(tcp, events, DEFAULT_WORKSPACE)
    assert r[0][0]["wait_frame"] == 9
    # Confirm at 3 cm while still inside, then backdate to the last stationary pose.
    assert r[0][0]["withdrawal_frame"] == 19
    assert r[0][0]["withdrawal_confirmed_frame"] == 20
    assert r[1][1]["withdrawal_frame"] == 69
    assert r[1][1]["withdrawal_confirmed_frame"] == 70


def test_waiting_pose_does_not_depend_on_other_arms_future():
    tcp, events = fixture()
    before = workspace_events(tcp, events, DEFAULT_WORKSPACE)
    tcp[0, 31:45, 0] = 0.8
    after = workspace_events(tcp, events, DEFAULT_WORKSPACE)
    assert after == before


def test_invalid_workspace_and_missing_withdrawal_fail():
    tcp, events = fixture()
    config = dict(DEFAULT_WORKSPACE, maximum_m=[0, 0, 0])
    with pytest.raises(ValueError, match="geometry"):
        workspace_events(tcp, events, config)
    tcp[0, 15:46, 1] = 0
    with pytest.raises(ValueError, match="withdrawal"):
        workspace_events(tcp, events, DEFAULT_WORKSPACE)


def test_inward_jitter_resets_the_backdated_origin():
    tcp, events = fixture()
    tcp[0, 15:20, 1] = [0.0, 0.01, -0.01, -0.005, 0.005]
    tcp[0, 20, 1] = 0.025
    record = workspace_events(tcp, events, DEFAULT_WORKSPACE)[0][0]
    assert record["withdrawal_frame"] == 17
    assert record["withdrawal_confirmed_frame"] == 20


def test_nominal_release_starts_at_onset_not_confirmation():
    from real_robot_data_retime.timeline.workpiece_workspace import nominal_schedule

    clocks = [np.arange(12), np.arange(12)]
    events = [
        [
            {"withdrawal_frame": 3, "withdrawal_confirmed_frame": 5},
            {"withdrawal_frame": 8},
        ],
        [{"withdrawal_frame": 5}, {"withdrawal_frame": 10}],
    ]
    left, right = nominal_schedule(clocks, [[6], [1, 8]], events)
    index = np.flatnonzero(right > 1)[0]
    assert left[index - 1] == 3


def test_short_brake_preserves_every_frame_through_placement():
    from real_robot_data_retime.timeline.workpiece import approach_clock

    clock, holds, ramps = approach_clock(0, 60, [31], 30, brake_after={31: 30})
    np.testing.assert_array_equal(clock[:31], np.arange(31))
    assert ramps[0]["brake_intervals"] == 2
    assert clock[holds[0]] == 31
    assert ramps[0]["brake_source_start"] == 30


def test_joint_onsets_restore_preparation_before_late_visual_annotations():
    from real_robot_data_retime.timeline.workpiece_workspace import (
        preparation_onsets,
        source_clocks,
    )

    state = np.zeros((220, 14))
    state[:60, :6] = np.arange(60)[:, None]
    state[60:, :6] = 59
    state[90:, 7:13] = np.arange(130)[:, None]
    own = [
        [
            {"approach_start": 20, "pickup_frame": 40, "release_frame": 55},
            {"retract_end": 80},
        ],
        [
            {"approach_start": 125, "pickup_frame": 150, "release_frame": 155},
            {"retract_end": 195},
        ],
    ]
    starts = preparation_onsets(state, state.copy(), own)
    assert starts[0] == 0
    assert 60 < starts[1] <= 90
    assert own[1][0]["approach_start"] == 125
    own[1][0]["approach_start"] = 70
    assert preparation_onsets(state, state.copy(), own) == starts

    clocks, _, _ = source_clocks(own, [[65], [135, 175]], 30, starts=starts)
    assert clocks[1][0] == starts[1]
    assert np.max(np.diff(clocks[1])) <= 1 + 1e-8
    assert np.all(np.diff(clocks[1][: 125 - starts[1]]) > 0)


def test_second_wait_uses_full_return_and_approach_before_late_visual_label():
    from real_robot_data_retime.timeline.workpiece_workspace import (
        waiting_source_frames,
    )

    inside = np.zeros((3, 60), bool)
    inside[0, 11:15] = True
    inside[0, 20:26] = True
    inside[1, 15:19] = True
    inside[2, 31:36] = True
    inside[2, 45:51] = True
    own = [
        [{"release_frame": 10}, {"approach_start": 23, "pickup_frame": 25}],
        [
            {"approach_start": 10, "pickup_frame": 18, "release_frame": 30},
            {"approach_start": 48, "pickup_frame": 50},
        ],
    ]
    assert waiting_source_frames(inside, own, [0, 0]) == [[19], [14, 44]]


def test_explicit_waits_must_remain_outside_before_first_entry():
    from real_robot_data_retime.timeline.workpiece_workspace import (
        workspace_waiting_stops,
    )

    tcp = np.zeros((2, 60, 3))
    tcp[0, 20:26] = [0.35, 0, 0.05]
    tcp[1, 15:19] = [0.35, 0, 0.05]
    tcp[1, 45:51] = [0.35, 0, 0.05]
    own = [
        [{"release_frame": 10}, {"pickup_frame": 25}],
        [{"pickup_frame": 18, "release_frame": 30}, {"pickup_frame": 50}],
    ]
    cache = {"tcp": tcp, "own": own, "starts": [0, 0]}
    config = dict(DEFAULT_WORKSPACE, waiting_source_frames=[18, 12, 40])
    assert workspace_waiting_stops(cache, config) == [[18], [12, 40]]
    with pytest.raises(ValueError, match="outside preparation"):
        workspace_waiting_stops(cache, dict(config, waiting_source_frames=[21, 12, 40]))


def test_disabled_waits_preserve_uninterrupted_source_preparation():
    from real_robot_data_retime.timeline.workpiece_workspace import source_clocks

    own = [
        [{'release_frame': 30}, {'retract_end': 90}],
        [{'release_frame': 50}, {'retract_end': 110}],
    ]
    clocks, holds, ramps = source_clocks(own, [[], []], 30, starts=[0, 10])
    np.testing.assert_array_equal(clocks[0], np.arange(91))
    np.testing.assert_array_equal(clocks[1], np.arange(10, 111))
    assert holds == [[], []]
    assert ramps == [[], []]
    clocks, holds, ramps = source_clocks(own, [[], [75]], 30, starts=[0, 10])
    assert len(holds[1]) == 1
    assert clocks[1][holds[1][0]] == 75
    np.testing.assert_array_equal(clocks[1][:41], np.arange(10, 51))
