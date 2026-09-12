import numpy as np
import pytest
from real_robot_data_retime.collision import continuous


def test_reserves_between_sample_clearance_and_rejects_mid_edge(monkeypatch):
    captured = {}

    class Geometry:
        def __init__(self, left, right, urdf, meshes, margin_m):
            captured["margin"] = margin_m
            self.values = [np.array(left), np.array(right)]
            self.max_reach_m = 1
            self.margin = margin_m
            pose = (None, np.zeros((1, 3)), np.ones((1, 3)), 0, None)
            self.poses = [[pose, pose], [pose, pose]]

        def configuration_safe(self, *args):
            return True

        def _interpolated_pose(self, side, a, b, k, steps):
            return k / steps

        def _clear(self, left, right):
            return not 0.4 < left < 0.6

    monkeypatch.setattr(continuous, "PiperXClearance", Geometry)
    values = np.zeros((2, 7))
    values[1, 0] = 10
    c = continuous.ContinuousClearance(values, np.zeros_like(values), None, None)
    assert captured["margin"] > 0.021
    assert not c(0, 0, 1, 1)
    assert c.checked_samples > 1
    c.__call__.cache_clear()


def test_invalid_continuous_clearance_parameters():
    with pytest.raises(ValueError):
        continuous.ContinuousClearance(None, None, None, None, clearance_m=0)
