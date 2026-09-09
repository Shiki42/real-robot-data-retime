import numpy as np
import pytest
from real_robot_data_retime.timeline.scheduler import schedule_sources


def test_independent_runs_at_full_speed():
    plan = schedule_sources(4, 6, lambda *edge: True)
    assert plan.left.tolist() == [0, 1, 2, 3, 3, 3]
    assert plan.right.tolist() == list(range(6))


def test_right_waits_before_entering_future_collision():
    # Right pose 1 is safe now but cannot remain there while left passes pose 2.
    forbidden = {(2, 1), (2, 2)}

    def safe(i, j, ni, nj):
        return (i, j) not in forbidden and (ni, nj) not in forbidden

    plan = schedule_sources(5, 4, safe)
    assert plan.left.tolist() == [0, 1, 2, 3, 4, 4]
    assert plan.right.tolist() == [0, 0, 0, 1, 2, 3]


def test_drawer_gate_allows_cube_to_approach_then_wait():
    plan = schedule_sources(
        6,
        7,
        lambda *e: True,
        dependency=lambda i, j: i <= 2 or j >= 5,
        left_priority=False,
    )
    assert np.all((plan.left <= 2) | (plan.right >= 5))
    assert len(plan.left) == 8
    assert 2 in plan.left[2:5]


def test_endpoint_clear_but_swept_collision_rejected():
    def safe(i, j, ni, nj):
        return not (ni > i and nj > j)

    plan = schedule_sources(3, 3, safe)
    assert len(plan.left) == 5
    assert not np.any((np.diff(plan.left) > 0) & (np.diff(plan.right) > 0))


def test_no_safe_left_priority_path_fails_explicitly():
    with pytest.raises(ValueError, match="no safe schedule"):
        schedule_sources(3, 2, lambda i, j, ni, nj: ni != 1)
