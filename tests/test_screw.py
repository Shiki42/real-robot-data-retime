import numpy as np
import pytest

from real_robot_data_retime.timeline.screw import (
    screw_schedule,
    validate_screw_schedule,
)

WINDOWS = [(30, 40), (70, 80), (110, 120), (150, 160), (190, 200)]


@pytest.mark.parametrize("position", [0.0, 0.15, 0.5, 0.85, 1.0])
def test_five_coupled_windows_and_all_moving_frames_survive(position):
    values = np.arange(220)[:, None] * np.ones((1, 14))
    left, right, plan = screw_schedule(values, values, 0, 220, WINDOWS, position, 30)
    for clock in (left, right):
        np.testing.assert_array_equal(np.unique(clock), np.arange(220))
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
            for active in moving.T:
                where = np.flatnonzero(active)
                assert np.all(active[where[0] : where[-1] + 1])


def test_state_motion_cannot_be_discarded_by_static_commands():
    state = np.arange(220)[:, None] * np.ones((1, 14))
    left, right, _ = screw_schedule(
        state, np.zeros_like(state), 0, 220, WINDOWS, 0.5, 30
    )
    np.testing.assert_array_equal(np.unique(left), np.arange(220))
    np.testing.assert_array_equal(np.unique(right), np.arange(220))


def test_validator_rejects_changed_insertion_clock():
    state = np.arange(220)[:, None] * np.ones((1, 14))
    left, right, plan = screw_schedule(state, state, 0, 220, WINDOWS, 0.5, 30)
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
        screw_schedule(state, state, 0, 220, windows, 0.5, 30)


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
    left, right, plan = screw_schedule(values, values, 0, 220, WINDOWS, 0.5, 30)
    json.dumps(plan, allow_nan=False)
    left[10] = left[9]
    with pytest.raises(ValueError, match="bundle was interrupted"):
        validate_screw_schedule(left, right, plan["stages"], values, values, 0, 220)
