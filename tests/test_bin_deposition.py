import numpy as np
from real_robot_data_retime.tasks.workpiece import verify_deposit, bin_visit


def test_empty_visit_is_not_a_deposit():
    frames = np.full((80, 80, 100, 3), 180, np.uint8)
    robots = np.zeros((80, 80, 100), bool)
    visit = dict(clearance_frame=40, release_frame=35)
    empty = verify_deposit(frames, robots, 20, visit, [10, 10, 40, 40], 30)
    assert not empty["verified"]
    frames[35:, 25:33, 25:33] = 30
    assert verify_deposit(frames, robots, 20, visit, [10, 10, 40, 40], 30)["verified"]


def test_bin_visit_requires_dwell_then_exit():
    xy = np.tile([80.0, 70.0], (90, 1))
    xy[30:50] = [30, 30]
    visit = bin_visit(xy, 10, 90, [10, 10, 40, 40], 30)
    assert visit["entry_frame"] == 30 and visit["clearance_frame"] == 50
