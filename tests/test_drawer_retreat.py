import numpy as np
import pytest

from real_robot_data_retime.timeline.drawer_retreat import (
    retreat_start,
    early_closing_clear,
)
from real_robot_data_retime.uniform_drawer import source_cohort
from real_robot_data_retime.timeline.uniform import uniform_samples


def test_retreat_gate_does_not_precede_actual_motion_or_release():
    tcp = np.zeros((30, 3))
    tcp[15:, 2] = np.arange(1, 16) * 0.004
    event = dict(release_frame=10, approach_start=0)
    assert retreat_start(tcp, event, 22, 30) == 14
    event["release_frame"] = 17
    assert retreat_start(tcp, event, 22, 30) == 17


def test_quiet_or_inward_motion_does_not_trigger_early_close():
    tcp = np.zeros((30, 3))
    tcp[0, 0] = -1
    tcp[12:, 0] = np.arange(18) * 0.004
    assert retreat_start(tcp, dict(release_frame=10, approach_start=0), 22, 30) == 22


@pytest.mark.parametrize("blocked", ["drawer", "arms", None])
def test_early_closing_retains_both_geometry_checks(blocked):
    class Checker:
        margin = 0.005
        max_reach_m = 0.5

        def _pose(self, row, side):
            return row

        def pose_clears_volume(self, pose, volume, margin):
            return blocked != "drawer"

        def _clear(self, left, right, *, extra_margin=0):
            return blocked != "arms"

    values = np.zeros((4, 14))
    assert early_closing_clear(
        Checker(), values, values, None, (0.0, 1.0), (1.0, 2.0)
    ) == (blocked is None)


def test_filtered_sources_use_dense_uniform_ranks():
    indices = source_cohort(dict(source_indices=[2, 6, 9]), 10)
    points = [uniform_samples(indices.index(ep), len(indices)) for ep in indices]
    assert np.allclose(sorted(np.ravel(points)), np.arange(6) / 6)
    assert all(np.isclose(pair[1] - pair[0], 0.5) for pair in points)


@pytest.mark.parametrize("indices", [[], [1, 1], [3, 1], [-1], [10], [True]])
def test_invalid_source_cohort_is_rejected(indices):
    with pytest.raises(ValueError, match="source_indices"):
        source_cohort(dict(source_indices=indices), 10)
