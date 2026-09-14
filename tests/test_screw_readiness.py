import json
from pathlib import Path

import numpy as np
import pytest

from real_robot_data_retime.timeline.readiness import final_left_pose
from real_robot_data_retime.timeline.screw import (
    screw_schedule,
    validate_screw_schedule,
)
from real_robot_data_retime.timeline.smooth import sample_rows


def test_final_pose_is_after_adjustment_not_the_intermediate_wait():
    state = np.zeros((60, 7))
    state[10:30, 0] = 1
    state[30:40, 0] = np.linspace(1, 2, 10)
    state[40:, 0] = 2
    action = state.copy()
    action[45, 1] = 0.2
    frame, evidence = final_left_pose(state, action, 0, 55)
    assert frame == 46
    assert evidence["final_left_source_frame"] > 40


def episode_two(position):
    root = Path(__file__).parents[1]
    config = json.loads((root / "docs/screw-cohort/ep002.json").read_text())
    with np.load(root / "tests/fixtures/screw_ep002.npz") as data:
        state, action = data["state"], data["action"]
        start, stop = int(data["start"]), int(data["stop"])
    left, right, plan = screw_schedule(
        state,
        action,
        start,
        stop,
        config["coupled_intervals"],
        config["ready_frames"],
        config["right_retreat_ends"],
        position,
        30,
    )
    return left, right, plan, state, action, start, stop


@pytest.mark.parametrize("position", [0.15, 0.5, 0.85])
def test_episode_two_only_one_waiter_and_no_left_adjustment_after_wait(position):
    left, right, plan, state, action, _, _ = episode_two(position)
    for stage in plan["stages"]:
        if stage["kind"] != "independent":
            continue
        assert sum(tr["hold_frames"] > 0 for tr in stage["transitions"]) <= 1
        final = stage["final_ready_source_frames"][0]
        for values in [state, action]:
            assert np.all(
                np.ptp(values[final : stage["source_end"] + 1, :7], axis=0)
                <= np.array([0.1] * 6 + [0.2])
            )
        tr = stage["transitions"][0]
        if tr["mode"] == "final_pose_wait":
            a, b = tr["arrival_output_frame"], stage["output_end"]
            assert np.all(left[a:b] == final)
            for values in [state, action]:
                held = sample_rows(values[:, :7], left[a : b + 1])
                assert np.all(np.ptp(held, axis=0) <= np.array([0.1] * 6 + [0.2]))
        late = 0 if stage["late_preparation_arm"] == "left" else 1
        clock = (left, right)[late]
        a, b = stage["pickup_start_output_frames"][late], stage["output_end"]
        assert np.all(np.diff(clock[a : b + 1]) > 0)
    # The earlier fourth-round attempt/correction must remain protected.
    assert plan["stages"][7]["source_start"] == 1740


def test_validator_rejects_adjustment_after_final_wait():
    left, right, plan, state, action, start, stop = episode_two(0.15)
    tr = plan["stages"][0]["transitions"][0]
    left[tr["arrival_output_frame"]] -= 0.01
    with pytest.raises(ValueError, match="left moved again"):
        validate_screw_schedule(left, right, plan["stages"], state, action, start, stop)


def test_final_pose_rejects_nonfinite_motion():
    values = np.zeros((20, 7))
    values[3, 0] = np.nan
    with pytest.raises(ValueError):
        final_left_pose(values, values, 0, 19)


def test_episode_one_preserves_single_waiter_regression():
    root = Path(__file__).parents[1]
    c = json.loads((root / "docs/screw-pilot/config.json").read_text())
    with np.load(root / "tests/fixtures/screw_ep001.npz") as d:
        _, _, p = screw_schedule(
            d["state"],
            d["action"],
            int(d["start"]),
            int(d["stop"]),
            c["coupled_intervals"],
            c["ready_frames"],
            c["right_retreat_ends"],
            0.5,
            30,
        )
    assert p["validation"]["at_most_one_preparation_waiter"]
    assert p["validation"]["no_left_adjustment_after_final_wait"]


def test_ep002_middle_has_no_long_hidden_both_arm_wait():
    left, right, plan, state, action, _, _ = episode_two(0.5)
    motion = [
        np.c_[sample_rows(v[:, :7], left), sample_rows(v[:, 7:], right)]
        for v in (state, action)
    ]
    change = np.maximum(*[np.max(np.abs(np.diff(v, axis=0)), axis=1) for v in motion])
    for stage in plan["stages"]:
        if stage["kind"] != "independent":
            continue
        a, b = max(stage["pickup_start_output_frames"]), stage["output_end"]
        edges = np.diff(np.r_[False, change[a:b] < 0.01, False].astype(int))
        lengths = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
        assert not len(lengths) or lengths.max() <= 3


def test_ep002_rejects_the_old_intermediate_ready_pose():
    _, _, _, state, action, start, stop = episode_two(0.5)
    root = Path(__file__).parents[1]
    c = json.loads((root / "docs/screw-cohort/ep002.json").read_text())
    c["ready_frames"][0][0] = 309
    with pytest.raises(ValueError, match="still contains an adjustment"):
        screw_schedule(
            state,
            action,
            start,
            stop,
            c["coupled_intervals"],
            c["ready_frames"],
            c["right_retreat_ends"],
            0.5,
            30,
        )
