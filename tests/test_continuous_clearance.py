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

        def _clear(self, left, right, *, margin_m=None):
            return not 0.4 < left < 0.6

    monkeypatch.setattr(continuous, "PiperXClearance", Geometry)
    values = np.zeros((2, 7))
    values[1, 0] = 10
    c = continuous.ContinuousClearance(values, np.zeros_like(values), None, None)
    assert 0.0155 < captured["margin"] < 0.01551
    assert not c(0, 0, 1, 1)
    assert c.checked_samples >= 1
    c.__call__.cache_clear()


def test_invalid_continuous_clearance_parameters():
    with pytest.raises(ValueError):
        continuous.ContinuousClearance(None, None, None, None, clearance_m=0)


def test_adaptive_certificate_accepts_clearance_just_above_threshold(monkeypatch):
    class Geometry:
        def __init__(self, left, right, urdf, meshes, margin_m):
            self.values = [left, right]
            self.margin = margin_m
            self.max_reach_m = 1
            pose = (None, np.zeros((1, 3)), np.ones((1, 3)), 0, None)
            self.poses = [[pose, pose], [pose, pose]]

        def configuration_safe(self, *args):
            return True

        def _interpolated_pose(self, *args):
            return None

        def _clear(self, left, right, *, margin_m=None):
            return 0.01552 > (self.margin if margin_m is None else margin_m)

    monkeypatch.setattr(continuous, "PiperXClearance", Geometry)
    values = np.zeros((2, 7))
    values[1, 0] = 0.01
    c = continuous.ContinuousClearance(values, np.zeros_like(values), None, None)
    assert c(0, 0, 1, 1)
    assert c.checked_samples > 1
    c.__call__.cache_clear()


def test_source_clock_cache_keys_are_source_frames_not_output_indices():
    calls = []

    class Geometry:
        values = (np.zeros((2, 7)), np.zeros((2, 7)))
        poses = [[object(), object()], [object(), object()]]

        def _pose(self, row, side):
            calls.append(("pose", side))
            return object()

        def _clear(self, left, right):
            calls.append(("clear",))
            return True

    base = Geometry()
    poses = [{float(i): p for i, p in enumerate(arm)} for arm in base.poses]
    pairs = {}
    first = continuous.ContinuousClearance.on_source_clocks(
        base, [np.array([0, 0.5, 1])] * 2, poses, pairs, 0.0155
    )
    assert first.checker.configuration_safe(1, 1)
    second = continuous.ContinuousClearance.on_source_clocks(
        base, [np.array([0, 0, 0.5, 1])] * 2, poses, pairs, 0.0155
    )
    assert second.checker.configuration_safe(2, 2)
    assert calls.count(("clear",)) == 1
    assert len([c for c in calls if c[0] == "pose"]) == 2
