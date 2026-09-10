from types import SimpleNamespace

import numpy as np

from real_robot_data_retime.idle_dp import compact_idle_optimal


def run(x, tol, protected=None):
    q = np.array(x, float).reshape(-1, 1)
    c = SimpleNamespace(retained_idle_seconds=4.0, minimum_idle_seconds=10.0)
    return compact_idle_optimal(
        q,
        q,
        1,
        np.array([tol]),
        np.array([tol]),
        np.zeros(len(q), bool) if protected is None else protected,
        c,
    )[0]


def test_global_start_choice_beats_greedy_partition():
    x = [0] * 5 + [1] * 4 + [2] * 20
    index = run(x, 1)
    assert len(index) == 9
    assert index[0] == 0 and index[-1] == 28


def test_relaxing_tolerance_cannot_increase_optimal_duration():
    x = [0] * 5 + [1] * 4 + [2] * 20
    lengths = [len(run(x, t)) for t in [0, 1, 2, 3]]
    assert lengths == sorted(lengths, reverse=True)


def test_protected_interval_is_never_skipped():
    x = [0] * 40
    p = np.zeros(40, bool)
    p[10:30] = True
    index = run(x, 1, p)
    assert set(range(10, 30)) <= set(index)


def test_shortcut_does_not_increase_peak_velocity_or_acceleration():
    x = np.array([0] * 5 + [1] * 4 + [2] * 20, float)
    i = run(x, 1)
    assert np.abs(np.diff(x[i])).max() <= np.abs(np.diff(x)).max()
    assert np.abs(np.diff(x[i], n=2)).max() <= np.abs(np.diff(x, n=2)).max()
