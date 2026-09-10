import numpy as np
import pytest
from real_robot_data_retime.timeline.smooth import (
    lift_clock,
    select_lift_peak,
    sample_rows,
)
from real_robot_data_retime.timeline.scheduler import schedule_sources
from real_robot_data_retime.compositing.interpolation import FlowFrames


def test_brake_hold_restart_is_monotone_and_reaches_zero_speed():
    x, peak, report = lift_clock(100, 200, 150, 30)
    assert x[0] == 100 and x[-1] == 200 and x[peak] == 150
    assert np.all(np.diff(x) >= 0) and np.max(np.diff(x)) <= 1 + 1e-10
    assert x[peak] - x[peak - 1] < 0.01
    assert x[peak + 1] - x[peak] < 0.015
    assert np.all(
        x[: report["brake_start_index"]] == np.floor(x[: report["brake_start_index"]])
    )
    assert np.all(
        x[report["restart_end_index"] :] == np.floor(x[report["restart_end_index"] :])
    )
    assert peak - report["brake_start_index"] == 15
    assert report["restart_end_index"] - peak == 9
    with pytest.raises(ValueError, match="insufficient"):
        lift_clock(149, 200, 150, 30)


def test_peak_excludes_higher_unsafe_and_post_release_poses():
    tcp = np.c_[np.zeros(12), np.zeros(12), [0, 1, 2, 3, 4, 5, 6, 100, 4, 3, 200, 0]]
    eligible = np.ones(12, bool)
    eligible[7] = False
    peak, _ = select_lift_peak(tcp, eligible, 3, 9)
    assert peak == 6
    with pytest.raises(ValueError, match="no safe"):
        select_lift_peak(tcp, np.zeros(12, bool), 3, 9)


def test_scheduler_waits_at_peak_and_opens_before_restart_interval():
    left, peak, _ = lift_clock(0, 40, 15, 30)
    opening = peak + 20

    def safe(i, j, ni, nj):
        return left[ni] <= 15 or j >= opening

    plan = schedule_sources(
        len(left),
        80,
        safe,
        left_priority=False,
        can_wait=lambda side, i: side == 1 or i in (peak, len(left) - 1),
    )
    clocks = left[plan.left]
    interior_waits = np.flatnonzero((np.diff(clocks) == 0) & (clocks[:-1] < 40))
    assert len(interior_waits) == 20
    assert np.all(clocks[interior_waits] == 15)
    moving = np.flatnonzero(clocks[1:] > 15)
    assert np.all(plan.right[moving] >= opening)
    sampled = sample_rows(np.c_[np.arange(41) * 2, np.full(41, 35)], clocks)
    assert np.allclose(sampled[:, 0], clocks * 2)
    assert np.all(sampled[:, 1] == 35)


def test_flow_moves_foreground_instead_of_cross_fading_two_locations():
    frames = np.zeros((2, 48, 64, 3), np.uint8)
    masks = np.zeros((2, 48, 64), bool)
    masks[0, 10:25, 10:25] = True
    masks[1, 10:25, 20:35] = True
    frames[masks] = 200
    flow = FlowFrames(frames)
    forward = np.zeros((48, 64, 2), np.float32)
    forward[..., 0] = 10
    flow.pair = lambda lo: (forward, -forward)
    image, mask = flow.sample(0.5, masks)
    yy, xx = np.where(mask)
    assert (xx.min(), xx.max()) == (15, 29)
    assert np.all(image[mask] == 200)
    original, original_mask = flow.sample(0, masks)
    assert np.array_equal(original, frames[0])
    assert np.array_equal(original_mask, masks[0])
