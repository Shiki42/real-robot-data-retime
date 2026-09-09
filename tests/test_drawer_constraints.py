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
