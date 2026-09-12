import numpy as np
import pytest
from real_robot_data_retime.timeline.workpiece_workspace import (
    workspace_events,
    DEFAULT_WORKSPACE,
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
                dict(
                    robot_id=["left", "right"][side],
                    approach_start=start,
                    pickup_frame=start + 15,
                    release_frame=start + 35,
                    retract_end=start + 45,
                )
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
            dict(withdrawal_frame=3, withdrawal_confirmed_frame=5),
            dict(withdrawal_frame=8),
        ],
        [dict(withdrawal_frame=5), dict(withdrawal_frame=10)],
    ]
    left, right = nominal_schedule(clocks, [[6], [1, 8]], events)
    index = np.flatnonzero(right > 1)[0]
    assert left[index - 1] == 3


def test_admission_bound_matches_clear_shortest_path():
    from real_robot_data_retime.timeline.workpiece_workspace import (
        admission_remaining_bound,
    )
    from real_robot_data_retime.timeline.scheduler import schedule_sources

    gates = [(1, 2, 0, 5), (0, 7, 1, 6), (1, 9, 0, 10)]

    def dependency(i, j):
        clocks = (i, j)
        return all(
            clocks[s] <= stop or clocks[o] >= release for s, stop, o, release in gates
        )

    bound = admission_remaining_bound([14, 15], gates)
    for i, j in [(0, 0), (3, 2), (7, 4), (8, 9), (12, 11)]:
        if not dependency(i, j):
            continue

        def safe(a, b, na, nb):
            current = (a + i, b + j)
            following = (na + i, nb + j)
            return all(
                not (current[s] <= stop < following[s] and current[o] < release)
                for s, stop, o, release in gates
            )

        path = schedule_sources(
            14 - i,
            15 - j,
            safe,
            dependency=lambda a, b: dependency(a + i, b + j),
            left_priority=False,
        )
        assert bound(i, j) <= len(path.left) - 1


def test_short_brake_preserves_every_frame_through_placement():
    from real_robot_data_retime.timeline.workpiece import approach_clock

    clock, holds, ramps = approach_clock(0, 60, [31], 30, brake_after={31: 30})
    np.testing.assert_array_equal(clock[:31], np.arange(31))
    assert ramps[0]["brake_intervals"] == 2
    assert clock[holds[0]] == 31
    assert ramps[0]["brake_source_start"] == 30
