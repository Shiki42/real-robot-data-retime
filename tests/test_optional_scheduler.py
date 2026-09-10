import numpy as np

from real_robot_data_retime.optional_scheduler import optional_schedule


def test_original_path_is_retained_when_shortcuts_conflict():
    n = 6
    base = np.arange(n)
    wait = np.ones(n, bool)
    l, r, s = optional_schedule(
        n, {1: [3]}, {2: [4]}, base, lambda l, r: l == r, lambda *x: True, wait, wait
    )
    np.testing.assert_array_equal(l, base)
    np.testing.assert_array_equal(r, base)
    assert not s["node_budget_exhausted"]


def test_can_choose_or_decline_each_shortcut():
    n = 8
    base = np.arange(n)
    wait = np.ones(n, bool)
    l, r, s = optional_schedule(
        n, {1: [4]}, {2: [5]}, base, lambda *x: True, lambda *x: True, wait, wait
    )
    assert len(l) == 6 and len(l) == len(r)
    assert l[-1] == r[-1] == n - 1 and s["found_strict_improvement"]


def test_coupled_interval_keeps_clocks_identical():
    n = 10
    wait = np.ones(n, bool)
    dep = lambda l, r: l == r if (5 <= l <= 7 or 5 <= r <= 7) else True
    l, r, s = optional_schedule(
        n, {1: [3]}, {2: [4]}, np.arange(n), dep, lambda *x: True, wait, wait
    )
    assert all(dep(int(a), int(b)) for a, b in zip(l, r))


def test_invalid_or_unsafe_baseline_is_not_returned():
    import pytest

    wait = np.ones(3, bool)
    with pytest.raises(ValueError):
        optional_schedule(3, {}, {}, [], lambda *x: True, lambda *x: True, wait, wait)
    with pytest.raises(ValueError, match="transition"):
        optional_schedule(
            3, {}, {}, np.arange(3), lambda *x: True, lambda *x: False, wait, wait
        )
    with pytest.raises(ValueError, match="Wait"):
        optional_schedule(
            3, {}, {}, np.arange(3), lambda *x: True, lambda *x: True, wait[:1], wait
        )
