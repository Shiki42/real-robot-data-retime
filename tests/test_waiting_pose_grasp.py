import numpy as np
from real_robot_data_retime.timeline import planner


class SafeRig:
    def __init__(self, left, right, *args, **kwargs):
        self.max_reach_m = 0.1
        self.poses = [
            [[None, None, None, 0.0, np.array([0.0, 0.0, 0.4])] for _ in rows]
            for rows in [left, right]
        ]

    def __call__(self, *args):
        return True

    def configuration_safe(self, *args):
        return True

    def arm_clears_volume(self, *args, **kwargs):
        return True

    def pose_clears_volume(self, *args, **kwargs):
        return True

    def _interpolated_pose(self, side, start, *args):
        return self.poses[side][start]


def inputs(monkeypatch, held_clear):
    monkeypatch.setattr(planner, "PiperXClearance", SafeRig)
    monkeypatch.setattr(
        planner, "drawer_sweep", lambda *a: (np.eye(3), np.zeros(3), np.ones(3))
    )
    monkeypatch.setattr(planner, "outside_box", lambda *a, **k: held_clear)
    state = np.zeros((20, 14))
    state[:, 0] = np.arange(20)
    state[:, 7] = np.arange(20)
    state[:12, 6] = 30
    state[12:, 6] = 70
    action = state.copy()
    action[11, 6] = 70
    timeline = dict(
        source_frames=20,
        task="drawer",
        fps=10,
        episodes=[
            dict(
                robot_id="left",
                grasp_start=3,
                grasp_frame=3,
                pickup_frame=3,
                release_frame=14,
                release_confirmation_frame=14,
                last_attached_frame=10,
            )
        ],
        drawer_motion=dict(open_frame=2, pull_start=1, close_start=17),
    )
    return state, action, timeline


def test_wait_does_not_replay_an_opening_gripper_command(monkeypatch):
    state, action, timeline = inputs(monkeypatch, True)
    left, right, plan = planner.plan_joints(
        state, action, timeline, "unused", "unused", terminal_seconds=0
    )
    assert plan["dependencies"]["safe_wait_frame"] == 10
    assert plan["dependencies"]["withdrawal_frame"] > 14


def test_no_held_wait_uses_recorded_pre_pickup_pose(monkeypatch):
    state, action, timeline = inputs(monkeypatch, False)
    left, right, plan = planner.plan_joints(
        state, action, timeline, "unused", "unused", terminal_seconds=0
    )
    assert (
        plan["dependencies"]["wait_method"]
        == "empty_pre_pickup_pose_clear_of_future_drawer_sweep"
    )
    assert plan["dependencies"]["safe_wait_frame"] == 2
    assert np.all((left < 3) | (right >= 2))
