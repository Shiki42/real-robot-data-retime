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
