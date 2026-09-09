import numpy as np
import pytest
from real_robot_data_retime.collision.piperx import PiperXClearance


def test_swept_arc_collision_between_clear_endpoints():
    fcl = pytest.importorskip("hppfcl")
    checker = PiperXClearance.__new__(PiperXClearance)
    checker.fcl = fcl
    checker.margin = 0.01
    checker.max_reach_m = 0.52
    checker.geometry = [fcl.Sphere(0.02)]
    checker.radii = np.array([0.02])
    checker._configuration_cache = {}
    from collections import OrderedDict

    checker._interpolation_cache = OrderedDict()
    left = np.zeros((2, 7))
    left[:, 0] = [-30, 30]
    right = np.zeros((2, 7))
    checker.values = (left, right)

    def pose(row, side):
        angle = np.deg2rad(row[0])
        p = (
            np.array([0.5 * np.cos(angle), 0.5 * np.sin(angle), 0.0])
            if side == 0
            else np.array([0.5, 0.0, 0.0])
        )
        matrix = np.eye(4)
        matrix[:3, 3] = p
        return matrix[None], p[None], np.full((1, 3), 0.02), 0.0, p

    checker._pose = pose
    checker.poses = [
        [pose(row, side) for row in values]
        for side, values in enumerate(checker.values)
    ]
    assert checker.configuration_safe(0, 0) and checker.configuration_safe(1, 1)
    assert not checker(0, 0, 1, 1)
