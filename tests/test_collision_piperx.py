import os
from pathlib import Path
import numpy as np
import pytest


@pytest.mark.skipif(
    not os.environ.get("ROBOVISUALIZE_ASSETS"),
    reason="RoboVisualize mesh integration test",
)
def test_real_mesh_clearance_and_total_aperture():
    from real_robot_data_retime.collision.piperx import PiperXClearance

    x = np.zeros((2, 7))
    x[1, 6] = 70
    c = PiperXClearance(
        x,
        x,
        Path("assets/piper_x_description.urdf"),
        Path(os.environ["ROBOVISUALIZE_ASSETS"]),
    )
    assert c(0, 0, 1, 1)
    c._pose(x[1], 0)
    assert np.isclose(c.model.q[c.model.joint_q_indices["joint7"]], 0.035)
    half = x[1].copy()
    half[6] = 35
    c._pose(half, 0)
    assert np.isclose(c.model.q[c.model.joint_q_indices["joint7"]], 0.0175)
    overlapping = PiperXClearance(
        x,
        x,
        Path("assets/piper_x_description.urdf"),
        Path(os.environ["ROBOVISUALIZE_ASSETS"]),
        base_spacing_m=0.001,
    )
    assert not overlapping(0, 0, 0, 0)


@pytest.mark.skipif(
    not os.environ.get("ROBOVISUALIZE_ASSETS"),
    reason="RoboVisualize mesh integration test",
)
def test_oriented_scene_volume_broad_phase_preserves_collision_decision():
    from real_robot_data_retime.collision.piperx import PiperXClearance

    x = np.zeros((1, 7))
    checker = PiperXClearance(
        x,
        x,
        Path("assets/piper_x_description.urdf"),
        Path(os.environ["ROBOVISUALIZE_ASSETS"]),
    )
    angle = 0.7
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    center = np.array([0.0, checker.base_spacing_m / 2, 0.0]) @ rotation
    assert not checker.arm_clears_volume(0, 0, (rotation, center - 0.15, center + 0.15))
    center = np.array([3.0, 3.0, 3.0]) @ rotation
    assert checker.arm_clears_volume(0, 0, (rotation, center - 0.15, center + 0.15))
