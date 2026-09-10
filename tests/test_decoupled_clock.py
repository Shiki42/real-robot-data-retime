import numpy as np
import pytest

from real_robot_data_retime.decoupled_clock import compose_independent_sources


def test_modes_preserve_motion_and_shared_boundaries():
    l = [2, 3, 7, 8]
    r = [2, 4, 5, 6, 8]
    pre = [0, 1, 2]
    post = [8, 9, 10]
    for mode in ["parallel", "left_first", "right_first"]:
        a, b = compose_independent_sources(l, r, pre, post, mode)
        assert np.array_equal(a[:3], pre) and np.array_equal(b[:3], pre)
        assert np.array_equal(a[-3:], post) and np.array_equal(b[-3:], post)
        assert list(a[np.r_[True, np.diff(a) != 0]]) == [0, 1, 2, 3, 7, 8, 9, 10]
        assert list(b[np.r_[True, np.diff(b) != 0]]) == [0, 1, 2, 4, 5, 6, 8, 9, 10]
        assert len(a) == len(b)
        if mode == "left_first":
            assert np.max(a[b == 2]) == 8
        if mode == "right_first":
            assert np.max(b[a == 2]) == 8


def test_offsets_change_only_waits():
    a, b = compose_independent_sources([0, 1, 4], [0, 2, 3, 4], [0], [4], left_delay=3)
    assert list(a[:4]) == [0, 0, 0, 0]
    assert a[-1] == b[-1] == 4 and len(a) == 6


def test_rejects_invalid_boundaries_and_delays():
    with pytest.raises(ValueError):
        compose_independent_sources([0, 2], [0, 3], [0], [3])
    with pytest.raises(ValueError):
        compose_independent_sources([0, 2], [0, 2], [0], [2], left_delay=-1)


def test_fractional_and_negative_indices_are_rejected():
    with pytest.raises(ValueError):
        compose_independent_sources([0, 1.5], [0, 2], [0], [2])
    with pytest.raises(ValueError):
        compose_independent_sources([-1, 2], [-1, 2], [-1], [2])
