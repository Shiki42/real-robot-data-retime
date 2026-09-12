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
