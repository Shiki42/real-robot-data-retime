import numpy as np
from real_robot_data_retime.timeline.smooth import smooth_wait_boundaries
from real_robot_data_retime.automatic_dataset import remap_table
from real_robot_data_retime.timeline.holds import append_terminal_hold
from real_robot_data_retime.compositing.interpolation import FlowFrames


def test_alternating_waits_preserve_pair_path_and_ease_at_each_boundary():
    left = np.r_[np.arange(21), np.full(12, 20), np.arange(21, 45), np.full(12, 44)]
    right = np.arange(len(left))
    output_left, output_right, report = smooth_wait_boundaries(left, right, 30)
    assert len(report["transitions"]) == 3
    assert len(output_left) > len(left)
    assert np.all(np.diff(output_left) >= 0) and np.all(np.diff(output_right) >= 0)
    assert np.array_equal(
        [output_left[0], output_right[0], output_left[-1], output_right[-1]],
        [left[0], right[0], left[-1], right[-1]],
    )
    # Right source time is the original paired-path coordinate in this case.
    assert np.allclose(output_left, np.interp(output_right, np.arange(len(left)), left))
    for phase in report["transitions"]:
        stop = phase["stop_output_frame"]
        begin = phase["brake_start_output_frame"]
        end = phase["restart_end_output_frame"]
        assert stop - begin == 15 and end - stop == 9
        assert output_right[stop] - output_right[stop - 1] < 0.01
        assert output_right[stop + 1] - output_right[stop] < 0.015
    holds = (output_right >= 20) & (output_right <= 32)
    assert np.all(output_left[holds] == 20)


def test_adjacent_wait_boundaries_do_not_overlap_or_reverse():
    output_left, output_right, p = smooth_wait_boundaries(
        [0, 1, 1, 2, 2, 3], [0, 0, 1, 1, 2, 2], 30
    )
    assert np.all(np.diff(output_left) >= 0) and np.all(np.diff(output_right) >= 0)
    assert max(np.diff(output_left).max(), np.diff(output_right).max()) <= 1
    stops = [x["stop_output_frame"] for x in p["transitions"]]
    assert np.all(np.diff(stops) >= 24)


def test_no_wait_does_not_change_source_path():
    output_left, output_right, p = smooth_wait_boundaries(
        np.arange(50), np.arange(50), 30
    )
    assert np.array_equal(output_left, np.arange(50)) and np.array_equal(
        output_right, output_left
    )
    assert p["transitions"] == []


def test_fractional_action_and_hold_metadata_are_not_truncated():
    import pyarrow as pa

    rows = np.arange(70, dtype=float).reshape(5, 14)
    table = pa.table(
        {
            "action": rows.tolist(),
            "observation.state": rows.tolist(),
            "timestamp": np.arange(5, dtype=float),
            "frame_index": np.arange(5),
            "index": np.arange(5),
            "episode_index": np.zeros(5, dtype=int),
            "task_index": np.zeros(5, dtype=int),
        }
    )
    output_left = np.array([0.0, 0.25, 1.5, 4.0])
    output_right = np.array([0.0, 0.5, 2.5, 4.0])
    result = remap_table(table, output_left, output_right, 0)
    a = np.array(result["action"].to_pylist())
    assert a[1, 0] == 3.5 and a[1, 7] == 14
    assert result["retime.left_source_frame"].to_pylist() == output_left.tolist()
    held_l, held_r, hold = append_terminal_hold(output_left, output_right, 30)
    assert held_l[1] == 0.25 and held_r[1] == 0.5
    assert hold["frames"] == 60


def test_depth_interpolation_preserves_millimetres_and_invalid_support():
    rgb = np.zeros((2, 48, 64, 3), np.uint8)
    flow = FlowFrames(rgb)
    zero = np.zeros((48, 64, 2), np.float32)
    flow.pair = lambda lo: (zero, zero)
    a = np.full((48, 64), 1000, np.uint16)
    b = np.full_like(a, 2000)
    image, valid = flow.sample(0.5, [a > 0, b > 0], values=[a, b])
    assert image.dtype == np.uint16 and np.all(image == 1500) and valid.all()


def test_joint_planner_audits_smoothed_state_and_action(monkeypatch):
    from types import SimpleNamespace
    from real_robot_data_retime.timeline import planner

    instances = []

    class Rig:
        def __init__(self, left, right, *args, **kwargs):
            instances.append((np.asarray(left), np.asarray(right)))

        def __call__(self, *args):
            return True

    monkeypatch.setattr(planner, "PiperXClearance", Rig)
    monkeypatch.setattr(
        planner,
        "detect_arm_segments",
        lambda *args: SimpleNamespace(
            left=SimpleNamespace(start=0, end=21),
            right=SimpleNamespace(start=0, end=41),
        ),
    )
    state = np.repeat(np.arange(50, dtype=float)[:, None], 14, axis=1)
    action = state + 0.01
    left, right, report = planner.plan_joints(
        state,
        action,
        dict(task="letters", source_frames=50, fps=30),
        "unused",
        "unused",
    )
    assert len(instances) == 3  # original geometry, output state and output action
    assert report["smoothing"]["transitions"]
    assert len(instances[1][0]) == report["synthetic_terminal_hold"]["start_frame"]
    assert np.any(left != np.floor(left)) or np.any(right != np.floor(right))
    assert np.allclose(instances[2][0] - instances[1][0], 0.01)


def test_smoothed_mesh_failure_is_not_replaced_with_hard_stop(monkeypatch):
    import pytest
    from types import SimpleNamespace
    from real_robot_data_retime.timeline import planner

    count = 0

    class Rig:
        def __init__(self, *args, **kwargs):
            nonlocal count
            count += 1
            self.original = count == 1

        def __call__(self, *args):
            return self.original

    monkeypatch.setattr(planner, "PiperXClearance", Rig)
    monkeypatch.setattr(
        planner,
        "detect_arm_segments",
        lambda *args: SimpleNamespace(
            left=SimpleNamespace(start=0, end=21),
            right=SimpleNamespace(start=0, end=41),
        ),
    )
    values = np.zeros((50, 14))
    with pytest.raises(planner.NoSafeSchedule, match="smoothed trajectory failed"):
        planner.plan_joints(
            values,
            values,
            dict(task="workpiece", source_frames=50, fps=30),
            "unused",
            "unused",
        )
