from real_robot_data_retime.timeline.planner import original_pair_edge


def test_original_contact_replay_cannot_admit_new_pairs_or_prolong_contact():
    edges = {50, 51}
    assert original_pair_edge(50, 50, 51, 51, edges)
    assert not original_pair_edge(50, 49, 51, 51, edges)
    assert not original_pair_edge(50, 50, 52, 52, edges)
    assert not original_pair_edge(50, 50, 50, 50, edges)
    assert not original_pair_edge(49, 49, 50, 50, edges)


def test_planner_replays_only_original_paired_edges_when_strict_path_is_impossible(
    monkeypatch,
):
    import numpy as np
    import real_robot_data_retime.timeline.planner as planner

    class RecordedContact:
        def __init__(self, left, right, *args, **kwargs):
            self.values = (left, right)
            self.max_reach_m = 0.1
            self.poses = [
                [([None], None, None, 0.0, np.array([0.0, 0.0, 0.4])) for _ in rows]
                for rows in self.values
            ]

        def __call__(self, i, j, ni, nj):
            return i != 12 and ni != 12

        def configuration_safe(self, i, j):
            return i != 12

        def arm_clears_volume(self, *args, **kwargs):
            return True

        def pose_clears_volume(self, *args, **kwargs):
            return True

        def _interpolated_pose(self, side, start, stop, *args):
            return self.poses[side][start]

    monkeypatch.setattr(planner, "PiperXClearance", RecordedContact)
    monkeypatch.setattr(
        planner, "drawer_sweep", lambda *args: (np.eye(3), np.zeros(3), np.ones(3))
    )
    monkeypatch.setattr(planner, "outside_box", lambda *args, **kwargs: True)
    state = np.zeros((20, 14))
    state[:, 0] = np.arange(20)
    state[:, 7] = np.arange(20)
    timeline = dict(
        source_frames=20,
        task="drawer",
        fps=10,
        episodes=[
            dict(robot_id="left", grasp_start=3, pickup_frame=3, release_frame=12)
        ],
        drawer_motion=dict(open_frame=2, pull_start=1, close_start=17),
    )
    left, right, receipt = planner.plan_joints(
        state, state, timeline, "unused", "unused", terminal_seconds=0
    )
    assert receipt["new_edges_collision_free"]
    assert not receipt["swept_edges_verified"]
    assert receipt["preserved_original_pair_edges"]
    for edge in receipt["preserved_original_pair_edges"]:
        k, t = edge["output_edge"], edge["source_edge"]
        assert left[k] == right[k] == t
        assert left[k + 1] == right[k + 1] == t + 1
