import numpy as np
import pytest
from real_robot_data_retime.collision.diagnostics import violation_intervals


def test_violation_intervals_use_exact_threshold_and_sample_times():
    times = np.arange(7) / 120
    values = [0.02, 0.0155, 0.014, 0.012, 0.017, 0, 0.019]
    spans = violation_intervals(times, values, 0.0155)
    assert len(spans) == 2
    assert spans[0]["first_below_seconds"] == 2 / 120
    assert spans[0]["last_below_seconds"] == 3 / 120
    assert spans[0]["minimum_time_seconds"] == 3 / 120
    assert spans[1]["minimum_distance_m"] == 0


def test_unsorted_distance_samples_are_rejected():
    with pytest.raises(ValueError):
        violation_intervals([0, 0.1, 0.1], [0.02] * 3, 0.0155)


def test_exact_primitive_gap_and_overlap_are_reported_without_fake_depth():
    from types import SimpleNamespace
    from real_robot_data_retime.collision.diagnostics import minimum_mesh_distance

    fcl = pytest.importorskip("hppfcl")
    checker = SimpleNamespace(
        fcl=fcl, geometry=[fcl.Sphere(0.01)], radii=np.array([0.01])
    )

    def pose(x):
        matrix = np.eye(4)[None]
        matrix[0, 0, 3] = x
        return matrix, np.array([[x, 0, 0]]), np.full((1, 3), 0.01), 0, None

    distance, pair = minimum_mesh_distance(checker, pose(0), pose(0.03))
    assert distance == pytest.approx(0.01)
    assert pair == (0, 0)
    assert minimum_mesh_distance(checker, pose(0), pose(0.01))[0] == 0
