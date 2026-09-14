import numpy as np


def test_open_drawer_uses_original_speed_without_lift_stop():
    from real_robot_data_retime.timeline.uniform import uniform_lift_profile
    for position in [0.55, 0.65, 0.9]:
        clock, peak, ramps, timing = uniform_lift_profile(100, 700, 300, 220, position, 30)
        assert ramps is None
        assert peak == 200
        assert (np.diff(clock) == 1).all()
        assert timing['right_delay_frames'] + 220 <= timing['left_delay_frames'] + peak


def test_late_drawer_retains_smooth_lift_wait():
    from real_robot_data_retime.timeline.uniform import uniform_lift_profile
    clock, peak, ramps, timing = uniform_lift_profile(100, 700, 300, 220, 0, 30)
    assert ramps is not None
    assert np.diff(clock)[peak - 1] < 0.1
    assert timing['right_delay_frames'] + 220 > timing['left_delay_frames'] + peak
