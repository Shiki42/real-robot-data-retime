import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('review_idle', Path(__file__).parents[1] / 'scripts/export_review_idle_masks.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_generated_waits_keep_departure_and_terminal_arrival_supervision():
    clock = [0, 0, 0, 1, 2, 2, 2, 3, 4, 4, 4]
    expected = [1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1]
    np.testing.assert_array_equal(module.clock_idle_mask(clock), expected)
    np.testing.assert_array_equal(np.asarray(clock)[~module.clock_idle_mask(clock)], [0, 1, 2, 3, 4])


def test_slow_ramps_and_original_source_progress_are_not_idle():
    assert not module.clock_idle_mask([0, 1e-12, 0.01, 0.1, 1, 2]).any()
    assert not module.clock_idle_mask([4]).any()
    assert module.clock_idle_mask([4, 4, 4]).all()


@pytest.mark.parametrize('clock', [[], [1, 0], [0, np.nan], [[0, 1]]])
def test_invalid_clocks_are_rejected(clock):
    with pytest.raises(ValueError):
        module.clock_idle_mask(clock)


def test_required_open_wait_is_supervised_but_other_idle_is_unchanged():
    left = np.array([0, 0, 1, 2, 2, 2, 2, 3, 4, 4])
    right = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    stages = dict(peak_source_frame=2, open_source_frame=6)
    required = module.required_open_wait(left, right, stages)
    np.testing.assert_array_equal(np.flatnonzero(required), [3, 4, 5])
    idle = module.clock_idle_mask(left)
    idle[required] = False
    np.testing.assert_array_equal(np.flatnonzero(idle), [0, 9])
    assert not module.required_open_wait(left, right, dict(peak_source_frame=2, open_source_frame=2)).any()


def test_close_wait_preserves_dependency_and_masks_only_excess_quiet():
    left = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    right = np.array([0, 1, 2, 3, 4, 5.5, 6, 7])
    stages = dict(open_source_frame=1, withdrawal_source_frame=4, close_source_frame=6)
    quiet = np.array([1, 1, 1, 1, 0, 1, 0, 1], bool)
    required, excess = module.right_wait_masks(left, right, stages, quiet)
    np.testing.assert_array_equal(np.flatnonzero(required), [1, 2])
    np.testing.assert_array_equal(np.flatnonzero(excess), [3])
    # Adjustment frame 4, interpolation into moving source 6, and closing stay supervised.
    assert not excess[4:].any()
