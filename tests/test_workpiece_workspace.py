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
    # Four centimetres alone is insufficient until the EE envelope clears the box.
    assert r[0][0]["withdrawal_frame"] == 25
    assert r[1][1]["withdrawal_frame"] == 75


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
