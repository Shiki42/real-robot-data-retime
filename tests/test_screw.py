import numpy as np
import pytest

from real_robot_data_retime.timeline.screw import (
    screw_schedule,
    validate_screw_schedule,
)

WINDOWS = [(30, 40), (70, 80), (110, 120), (150, 160), (190, 200)]
READY = [(start - 4, start - 10) for start, _ in WINDOWS]
RETREATS = [end + 4 for _, end in WINDOWS]


@pytest.mark.parametrize("position", [0.0, 0.15, 0.5, 0.85, 1.0])
def test_five_coupled_windows_and_all_moving_frames_survive(position):
    values = np.arange(220)[:, None] * np.ones((1, 14))
    left, right, plan = screw_schedule(
        values, values, 0, 220, WINDOWS, READY, RETREATS, position, 30
    )
    for clock in (left, right):
        assert clock[0] == 0 and clock[-1] == 219
        assert np.diff(clock).max() <= 1 + 1e-9
    assert plan["validation"]["passed"]
    assert len([s for s in plan["stages"] if s["kind"] == "coupled"]) == 5
    for stage in plan["stages"]:
        if stage["kind"] == "independent":
            a, b = stage["output_start"], stage["output_end"]
            moving = (
                np.column_stack([np.diff(left[a : b + 1]), np.diff(right[a : b + 1])])
                > 0
            )
            assert moving.any(axis=1).all()
            for side, tr in enumerate(stage["transitions"]):
                if tr["mode"] == "continuous":
                    onset = stage["pickup_start_output_frames"][side]
                    assert (np.diff((left, right)[side][onset : b + 1]) > 0).all()


def test_state_motion_cannot_be_discarded_by_static_commands():
    state = np.arange(220)[:, None] * np.ones((1, 14))
    left, right, _ = screw_schedule(
        state, np.zeros_like(state), 0, 220, WINDOWS, READY, RETREATS, 0.5, 30
    )
    assert np.diff(left).max() <= 1 + 1e-9
    assert np.diff(right).max() <= 1 + 1e-9


def test_validator_rejects_changed_insertion_clock():
    state = np.arange(220)[:, None] * np.ones((1, 14))
    left, right, plan = screw_schedule(
        state, state, 0, 220, WINDOWS, READY, RETREATS, 0.5, 30
    )
    stage = plan["stages"][1]
    right[stage["output_start"] + 2] -= 1
    with pytest.raises(ValueError, match="coupled insertion"):
        validate_screw_schedule(left, right, plan["stages"], state, state, 0, 220)


@pytest.mark.parametrize(
    "windows",
    [WINDOWS[:4], [(30, 40), (39, 80), *WINDOWS[2:]], [(30.5, 40), *WINDOWS[1:]]],
)
def test_rejects_invalid_rounds(windows):
    state = np.zeros((220, 14))
    with pytest.raises(ValueError):
        screw_schedule(state, state, 0, 220, windows, READY, RETREATS, 0.5, 30)


def test_compositor_keeps_whole_crossing_arm_and_left_owned_bin():
    from real_robot_data_retime.compositing.screw import ScrewStageCompositor

    frames = np.full((2, 40, 60, 3), 210, dtype=np.uint8)
    masks = np.zeros((2, 2, 40, 60), bool)
    masks[0, 0, 25:35, :38] = True
    masks[1, 0, 25:35, :15] = True
    masks[0, 1, 5:15, 50:] = True
    masks[1, 1, 5:15, 40:] = True
    frames[0][masks[0, 0]] = 20
    frames[1][masks[1, 0]] = 30
    frames[0][masks[0, 1]] = 40
    frames[1][masks[1, 1]] = 50
    frames[0, 17:22, 32:37] = [40, 120, 180]
    frames[1, 17:22, 32:37] = [180, 120, 40]
    comp = ScrewStageCompositor(frames, masks, 30, [[31, 16, 7, 7]])
    np.testing.assert_array_equal(comp.frame(0, 0), frames[0])
    output = comp.frame(0, 1)
    np.testing.assert_array_equal(output[29, 35], frames[0, 29, 35])
    np.testing.assert_array_equal(output[9, 45], frames[1, 9, 45])
    np.testing.assert_array_equal(output[19, 34], frames[0, 19, 34])


def test_compositor_removes_wrong_source_arm_in_other_workspace():
    from real_robot_data_retime.compositing.screw import ScrewStageCompositor

    frames = np.full((3, 40, 60, 3), 210, dtype=np.uint8)
    masks = np.zeros((3, 2, 40, 60), bool)
    masks[0, 0, 25:35, :15] = True
    masks[1, 0, 25:35, :40] = True
    masks[2, 0, 25:35, :15] = True
    for t in range(3):
        frames[t][masks[t, 0]] = 20
    comp = ScrewStageCompositor(frames, masks, 30, [])
    output = comp.frame(0, 1)
    np.testing.assert_array_equal(output[30, 35], [210, 210, 210])


def test_stage_receipt_serializes_and_rejects_internal_pause():
    import json

    values = np.arange(220)[:, None] * np.ones((1, 14))
    left, right, plan = screw_schedule(
        values, values, 0, 220, WINDOWS, READY, RETREATS, 0.5, 30
    )
    json.dumps(plan, allow_nan=False)
    left[10] = left[9]
    with pytest.raises(ValueError, match="stopped|interrupted"):
        validate_screw_schedule(left, right, plan["stages"], values, values, 0, 220)


def test_right_lead_waits_near_contact_and_restarts_before_coupling():
    from real_robot_data_retime.timeline.smooth import sample_rows

    values = np.arange(220)[:, None] * np.ones((1, 14))
    left, right, plan = screw_schedule(
        values, values, 0, 220, WINDOWS, READY, RETREATS, 0.85, 30
    )
    assert np.count_nonzero(right != np.floor(right)) == 110
    assert np.count_nonzero(left != np.floor(left)) == 0
    motion = sample_rows(values[:, 7:], right)
    for stage in plan["stages"]:
        if stage["kind"] != "independent":
            continue
        transition = stage["transitions"][1]
        a, b = (
            transition["arrival_output_frame"],
            transition["restart_start_output_frame"],
        )
        assert b > a
        assert np.all(right[a : b + 1] == stage["right_ready_source_frame"])
        assert np.ptp(motion[a : b + 1], axis=0).max() == 0
        rates = np.diff(motion[transition["brake_start_output_frame"] : a + 1, 0])
        assert np.all(np.diff(rates) < 0)
        rates = np.diff(motion[b : transition["restart_end_output_frame"] + 1, 0])
        assert np.all(np.diff(rates) > 0)
        coupled = next(
            s
            for s in plan["stages"]
            if s["kind"] == "coupled" and s["cycle"] == stage["cycle"]
        )
        assert transition["restart_end_output_frame"] <= coupled["output_start"]


def test_compositor_consumes_fractional_clock_instead_of_truncating(monkeypatch):
    from real_robot_data_retime.compositing.screw import ScrewStageCompositor

    frames = np.full((2, 40, 60, 3), 210, np.uint8)
    masks = np.zeros((2, 2, 40, 60), bool)
    masks[:, 1, 5:15, 40:] = True
    frames[:, 5:15, 40:] = 20
    comp = ScrewStageCompositor(frames, masks, 30, [])
    interpolated = np.full((40, 60, 3), 77, np.uint8)

    def sample(source, endpoint_masks):
        assert source == 0.5
        assert len(endpoint_masks) == 2
        return interpolated, masks[0, 1]

    monkeypatch.setattr(comp.flow_frames, "sample", sample)
    output = comp.frame(0, 0.5)
    np.testing.assert_array_equal(output[10, 50], [77, 77, 77])
    assert comp.report()["interpolated_frames"]["right"] == 1


def test_coupled_render_preserves_native_frames_in_float_clock_arrays():
    from real_robot_data_retime.screw import render_frames

    frames = np.arange(36, dtype=np.uint8).reshape(3, 2, 2, 3)
    stages = [{"kind": "coupled", "output_end": 1}]
    clock = np.array([1.0, 2.0])
    result = np.array(list(render_frames(frames, [], clock, clock, stages)))
    np.testing.assert_array_equal(result, frames[1:])
    with pytest.raises(ValueError, match="native source"):
        list(render_frames(frames, [], clock - 0.5, clock - 0.5, stages))


def test_carried_screw_mask_does_not_claim_objects_before_pickup():
    from real_robot_data_retime.screw import segment_carried_screw

    class Model:
        def propagate(self, frames, proposals, seed_frame, reverse, stop_frame):
            for frame in range(seed_frame, stop_frame, -1 if reverse else 1):
                mask = np.zeros((1, 10, 10), bool)
                mask[0, 2, 3] = True
                yield frame, mask

    frames = np.zeros((7, 10, 10, 3), np.uint8)
    robots = np.zeros((7, 2, 10, 10), bool)
    spec = {
        "pickup_frame": 2,
        "source_frame": 4,
        "bbox": [1, 1, 4, 4],
        "positive_points": [[3, 2]],
        "negative_points": [],
    }
    segment_carried_screw(Model(), frames, robots, 0, 6, spec)
    assert not robots[:2].any()
    assert robots[2:, 1, 2, 3].all()
    assert not robots[:, 0].any()


@pytest.mark.parametrize("position", [0.15, 0.5, 0.85])
def test_right_retreat_is_immediate_native_and_precedes_new_pickup(position):
    values = np.arange(220)[:, None] * np.ones((1, 14))
    left, right, plan = screw_schedule(
        values, values, 0, 220, WINDOWS, READY, RETREATS, position, 30
    )
    for stage in plan["stages"]:
        if "mandatory_previous_right_retreat" not in stage:
            continue
        r = stage["mandatory_previous_right_retreat"]
        np.testing.assert_array_equal(
            right[r["output_start"] : r["output_end"] + 1],
            np.arange(r["source_start"], r["source_end"] + 1),
        )
        if stage["kind"] == "independent":
            assert stage["pickup_start_output_frames"][1] >= r["output_end"]
    stage = plan["stages"][2]
    r = stage["mandatory_previous_right_retreat"]
    right[r["output_start"] + 1] = right[r["output_start"]]
    with pytest.raises(ValueError, match="mandatory right retreat"):
        validate_screw_schedule(left, right, plan["stages"], values, values, 0, 220)


def test_small_lead_flows_through_without_full_stop():
    values = np.arange(220)[:, None] * np.ones((1, 14))
    left, right, plan = screw_schedule(
        values, values, 0, 220, WINDOWS, READY, RETREATS, 0.48, 30
    )
    stage = plan["stages"][0]
    assert stage["transitions"][0]["mode"] == "flow_through"
    for side, clock in enumerate((left, right)):
        onset = stage["pickup_start_output_frames"][side]
        assert (np.diff(clock[onset : stage["output_end"] + 1]) > 0).all()
    assert np.max(np.diff(left[: stage["output_end"] + 1])) <= 1 + 1e-8
