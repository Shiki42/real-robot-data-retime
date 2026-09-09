import numpy as np
from real_robot_data_retime.timeline.holds import compress_wait


def test_wait_compression_preserves_gripper_events_and_boundaries():
    x = np.zeros((100, 7))
    x[30:50, 6] = 20
    x[50:, 6] = 10
    indices = compress_wait(x, 20, 80)
    assert set([0, 19, 20, 30, 50, 79, 80, 99]) <= set(indices)
    assert len(indices) == 44
    assert np.array_equal(x[indices[indices < 20]], x[:20])


def test_wait_does_not_delete_genuine_transport_motion():
    x = np.zeros((30, 7))
    x[:, 0] = np.arange(30)
    assert np.array_equal(compress_wait(x, 2, 28), np.arange(30))


def test_terminal_hold_is_explicit_and_uses_original_last_poses():
    from real_robot_data_retime.timeline.holds import append_terminal_hold

    a, b, receipt = append_terminal_hold([5, 6], [8, 9], 30)
    assert len(a) == 62
    assert (a[2:] == 6).all() and (b[2:] == 9).all()
    assert receipt["frames"] == 60
    assert receipt["method"] == "repeated_source_boundary"
