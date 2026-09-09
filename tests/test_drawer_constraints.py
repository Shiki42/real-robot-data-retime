import numpy as np
from real_robot_data_retime.tasks.drawer_constraints import (
    discover_drawer_motion,
    precedence_gate,
)


def test_visual_handle_outward_dwell_return():
    x = np.r_[
        np.zeros(20),
        np.linspace(0, 40, 30),
        np.full(40, 40),
        np.linspace(40, 0, 30),
        np.zeros(20),
    ]
    handle = np.c_[100 + np.zeros(len(x)), 100 + x]
    distractor = np.c_[np.full(len(x), 30), np.full(len(x), 80)]
    result = discover_drawer_motion([distractor, handle], handle + 3, 30, 424)
    assert result.handle_candidate == 1
    assert 48 <= result.open_frame <= 53
    assert 88 <= result.close_start <= 94


def test_both_drawer_dependencies():
    kwargs = dict(
        safe_approach_end=100, open_frame=200, withdrawal_frame=180, close_start=300
    )
    assert precedence_gate(90, 100, **kwargs)
    assert not precedence_gate(120, 100, **kwargs)
    assert precedence_gate(120, 200, **kwargs)
    assert not precedence_gate(120, 300, **kwargs)
    assert precedence_gate(180, 300, **kwargs)


def test_drawer_state_is_relative_to_moving_cabinet():
    x = np.r_[
        np.zeros(20),
        np.linspace(0, 40, 30),
        np.full(40, 40),
        np.linspace(40, 0, 30),
        np.zeros(20),
    ]
    shift = np.linspace(0, 80, len(x))
    cabinet = np.c_[100 + np.zeros(len(x)), 60 + shift]
    handle = np.c_[100 + np.zeros(len(x)), 100 + x + shift]
    result = discover_drawer_motion([cabinet, handle], handle + 3, 30, 424)
    assert result.handle_candidate == 1
    assert result.reference_candidate == 0
    assert 48 <= result.open_frame <= 53
    assert 88 <= result.close_start <= 94


def test_dwell_jitter_is_not_mistaken_for_drawer_closing():
    x = np.r_[
        np.zeros(20),
        np.linspace(0, 40, 30),
        np.full(80, 40),
        np.linspace(40, 0, 30),
        np.zeros(20),
    ]
    x[75:84] = np.linspace(40, 45, 9)
    x[84:93] = np.linspace(45, 40, 9)
    handle = np.c_[np.full(len(x), 100), 100 + x]
    fixed = np.c_[np.full(len(x), 80), np.full(len(x), 60)]
    result = discover_drawer_motion([handle, fixed], handle + 3, 30, 424)
    assert result is not None
    assert 127 <= result.close_start <= 134
    assert result.open_frame < 75


def test_moving_cube_cannot_be_the_cabinet_reference():
    x = np.r_[
        np.zeros(20),
        np.linspace(0, 40, 30),
        np.full(80, 40),
        np.linspace(40, 0, 30),
        np.zeros(20),
    ]
    handle = np.c_[np.full(len(x), 100), 100 + x]
    fixed1 = np.c_[np.full(len(x), 80), np.full(len(x), 60)]
    fixed2 = fixed1 + 20
    cube = fixed1.astype(float)
    cube[85:115, 1] += np.linspace(0, 60, 30)
    cube[115:, 1] += 60
    result = discover_drawer_motion([handle, fixed1, fixed2, cube], handle + 3, 30, 424)
    assert result.reference_candidate in [1, 2]
    assert 127 <= result.close_start <= 134


def test_quiet_interval_that_started_before_opening_still_counts():
    from real_robot_data_retime.tasks.drawer_constraints import settled_open_frame

    gripper = np.full((100, 2), 20.0)
    assert settled_open_frame(gripper, 30, 80, 30, 640) == 30
    gripper[30:80] = np.nan
    assert settled_open_frame(gripper, 30, 80, 30, 640) is None
