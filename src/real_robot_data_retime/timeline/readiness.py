"""Find completion of the last left adjustment before reviewed contact."""

import numpy as np


def final_left_pose(
    state, action, start, contact, joint_tolerance=0.05, gripper_tolerance=0.1
):
    """Require the entire suffix to match the contact-entry pose in both streams.

    A transient intermediate stop is not a final pose. If no settled suffix
    exists, return contact itself; the scheduler must not invent a stop there.
    Contact is a reviewed task boundary, not inferred from joint stillness.
    """
    state, action = np.asarray(state), np.asarray(action)
    if state.ndim != 2 or state.shape[1] != 7 or action.shape != state.shape:
        raise ValueError("expected matching N x 7 state and action")
    if not np.isfinite([state, action]).all() or not 0 <= start < contact < len(state):
        raise ValueError("invalid finite motion or contact interval")
    values = np.c_[state, action]
    tolerance = np.tile([joint_tolerance] * 6 + [gripper_tolerance], 2)
    if np.any(tolerance <= 0) or not np.isfinite(tolerance).all():
        raise ValueError("invalid final-pose tolerances")
    delta = np.abs(values[start : contact + 1] - values[contact])
    outside = np.flatnonzero(np.any(delta > tolerance, axis=1))
    local = int(outside[-1]) + 1 if len(outside) else 0
    return start + local, {
        "source_reference_frame": contact,
        "final_left_source_frame": start + local,
        "joint_tolerance_deg": joint_tolerance,
        "gripper_tolerance_mm": gripper_tolerance,
        "max_state_action_difference": delta[local:].max(axis=0).tolist(),
    }
