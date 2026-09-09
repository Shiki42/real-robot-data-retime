import numpy as np
from real_robot_data_retime.collision.drawer import outside_box, safe_wait_before_entry


def test_wait_pose_must_clear_swept_volume_and_held_cube():
    bounds = (np.eye(3), np.array([0.0, 0.0, 0.0]), np.array([0.2, 0.2, 0.1]))
    tcp = np.array(
        [[0.1, 0.1, 0.2], [0.1, 0.1, 0.16], [0.1, 0.1, 0.12], [0.1, 0.1, 0.08]]
    )
    assert safe_wait_before_entry(tcp, 0, 3, bounds) == 1
    assert not outside_box(tcp[2], bounds)


def test_lifting_can_cross_future_sweep_before_drawer_arrives():
    from real_robot_data_retime.collision.drawer import drawer_body, drawer_sweep

    positions = np.array([[0.2, 0.0, 0.1], [0.4, 0.0, 0.1]])
    future = drawer_sweep(positions, 0, 1)
    closed = drawer_body(positions[0], future[0])
    cube = np.array([0.35, 0.0, 0.1])
    assert outside_box(cube, closed, radius=0.025)
    assert not outside_box(cube, future, radius=0.025)
    lifted = cube + np.array([0.0, 0.0, 0.2])
    assert outside_box(lifted, future, radius=0.025)
