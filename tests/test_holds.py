import numpy as np


def test_idle_compaction_preserves_slow_accumulated_motion_and_command_changes():
    from real_robot_data_retime.timeline.holds import compress_static_spans

    state = np.zeros((100, 7))
    action = state.copy()
    state[30:70, 0] = np.arange(40) * 0.1
    state[70:, 0] = 3.9
    action[:] = state
    action[80:85, 6] = 4
    kept = compress_static_spans(state, action, [(0, 100)], 30)
    assert len(kept) < 100
    assert set(range(30, 70)).issubset(set(kept))
    assert 79 in kept and 80 in kept and 84 in kept and 85 in kept
