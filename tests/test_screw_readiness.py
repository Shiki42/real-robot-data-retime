import numpy as np

from real_robot_data_retime.timeline.readiness import (
    align_ready_suffix,
    sustained_ready_frame,
)
from real_robot_data_retime.timeline.screw import screw_schedule


def poses(n):
    return np.repeat(np.eye(4)[None], n, axis=0)


def test_readiness_requires_sustained_position_orientation_and_closed_gripper():
    state = np.zeros((20, 7))
    action = state.copy()
    sp = poses(20)
    ap = poses(20)
    sp[:5, 0, 3] = 0.02
    angle = np.radians(5)
    sp[8, :2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    state[12, 6] = 3
    ap[14, 0, 3] = 0.02
    ready, e = sustained_ready_frame(state, action, sp, ap, 0, 19)
    assert ready == 15
    assert e["measured_and_commanded"][0]["max_position_mm"] == 0


def test_micro_adjustments_remain_on_source_path_when_retimed():
    clock = np.arange(100, dtype=float)
    for end in [76, 130]:
        mapped = align_ready_suffix(clock, 60, end)
        np.testing.assert_array_equal(mapped[:60], clock[:60])
        assert mapped[-1] == 99
        assert np.diff(mapped).min() > 0
        assert np.diff(mapped).max() <= 3


def test_ready_arm_residual_duration_cannot_force_active_arm_to_stop():
    # Two poses are physically ready at different times; the left has a long
    # recorded micro-alignment tail. It must not become a false preparation wait.
    values = np.arange(600)[:, None] * np.ones((1, 14))
    coupled = [(100, 110), (210, 220), (320, 330), (430, 440), (540, 550)]
    ready = [(a - 4, a - 15) for a, b in coupled]
    physical = [(a - 35, a - 15) for a, b in coupled]
    retreat = [b + 4 for a, b in coupled]
    _l, r, p = screw_schedule(
        values,
        values,
        0,
        600,
        coupled,
        ready,
        retreat,
        0.5,
        30,
        preparation_frames=physical,
    )
    for s in p["stages"]:
        if s["kind"] == "independent":
            assert s["transitions"][1]["hold_frames"] == 0
            start = s["pickup_start_output_frames"][1]
            assert (np.diff(r[start : s["output_end"] + 1]) > 0).all()


def test_episode_one_second_insertion_has_no_false_right_wait():
    import json
    from pathlib import Path

    root = Path(__file__).parents[1]
    config = json.loads((root / "docs/screw-pilot/config.json").read_text())
    with np.load(root / "tests/fixtures/screw_ep001.npz") as data:
        left, right, plan = screw_schedule(
            data["state"],
            data["action"],
            int(data["start"]),
            int(data["stop"]),
            config["coupled_intervals"],
            config["ready_frames"],
            config["right_retreat_ends"],
            0.5,
            30,
            preparation_frames=config["preparation_frames"],
        )
    stage = plan["stages"][2]
    assert stage["transitions"][1]["hold_frames"] == 0
    assert np.all(
        np.diff(right[stage["pickup_start_output_frames"][1] : stage["output_end"] + 1])
        > 0
    )
    assert left[709] == right[709] == 807
