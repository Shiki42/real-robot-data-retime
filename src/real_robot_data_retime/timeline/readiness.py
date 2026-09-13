"""Sustained task-space readiness, independent of remaining demonstration time."""

import numpy as np


def sustained_ready_frame(
    state,
    action,
    state_poses,
    action_poses,
    search_start,
    reference,
    position_mm=5.0,
    orientation_deg=2.0,
    gripper_mm=0.5,
    command_position_mm=10.0,
    command_orientation_deg=3.0,
):
    """Earliest pose whose entire suffix remains in the reference tolerance.

    Both measured and commanded TCP translation/orientation and gripper must
    agree. No frames are removed: this separates preparation from fine alignment.
    """
    state, action = np.asarray(state), np.asarray(action)
    state_poses, action_poses = np.asarray(state_poses), np.asarray(action_poses)
    if (
        state.ndim != 2
        or state.shape[1] != 7
        or action.shape != state.shape
        or state_poses.shape != (len(state), 4, 4)
        or action_poses.shape != state_poses.shape
    ):
        raise ValueError("expected matching N x 7 values and N x 4 x 4 poses")
    if not 0 <= search_start <= reference < len(state):
        raise ValueError("invalid readiness search interval")
    valid = np.ones(reference - search_start + 1, bool)
    metrics = []
    for channel, (values, poses) in enumerate(
        [(state, state_poses), (action, action_poses)]
    ):
        interval = poses[search_start : reference + 1]
        target = poses[reference]
        distance = np.linalg.norm(interval[:, :3, 3] - target[:3, 3], axis=1) * 1000
        angle = np.degrees(
            np.arccos(
                np.clip(
                    (np.einsum("nij,ij->n", interval[:, :3, :3], target[:3, :3]) - 1)
                    / 2,
                    -1,
                    1,
                )
            )
        )
        aperture = np.abs(
            values[search_start : reference + 1, 6] - values[reference, 6]
        )
        if not np.isfinite([distance, angle, aperture]).all():
            raise ValueError("nonfinite readiness geometry")
        valid &= (
            (distance <= (position_mm if channel == 0 else command_position_mm))
            & (angle <= (orientation_deg if channel == 0 else command_orientation_deg))
            & (aperture <= gripper_mm)
        )
        metrics.append((distance, angle, aperture))
    local = int(np.flatnonzero(np.logical_and.accumulate(valid[::-1])[::-1])[0])
    return search_start + local, {
        "reference_source_frame": int(reference),
        "ready_source_frame": search_start + local,
        "position_tolerance_mm": position_mm,
        "command_position_tolerance_mm": command_position_mm,
        "command_orientation_tolerance_deg": command_orientation_deg,
        "orientation_tolerance_deg": orientation_deg,
        "gripper_tolerance_mm": gripper_mm,
        "measured_and_commanded": [
            {
                "max_position_mm": float(d[local:].max()),
                "max_orientation_deg": float(a[local:].max()),
                "max_gripper_mm": float(g[local:].max()),
            }
            for d, a, g in metrics
        ],
    }


def minimum_alignment_duration(clock, ready_index, max_rate=3.0):
    distance = len(clock) - 1 - ready_index
    ramp = max(1, min(3, distance // 3))
    return max(2, int(np.ceil((distance + ramp * (max_rate - 1)) / max_rate)))


def align_ready_suffix(clock, ready_index, finish_length):
    """Positive-speed fine alignment with original speed at both joins."""
    from .smooth import sample_rows, speed_ramp

    distance = len(clock) - 1 - ready_index
    duration = finish_length - 1 - ready_index
    if min(distance, duration) < 2:
        raise ValueError("insufficient ready suffix for timing alignment")
    ramp = max(1, min(3, distance // 3, duration // 2))
    rate = (distance - ramp) / (duration - ramp)
    if rate <= 0 or rate > 3 + 1e-8:
        raise ValueError("fine alignment rate outside (0,3]")
    down = rate * np.arange(ramp + 1) + (1 - rate) * speed_ramp(ramp, ramp / 2, False)
    plateau = down[-1] + rate * np.arange(1, duration - 2 * ramp + 1)
    at = plateau[-1] if len(plateau) else down[-1]
    up = (
        at
        + rate * np.arange(1, ramp + 1)
        + (1 - rate) * speed_ramp(ramp, ramp / 2, True)[1:]
    )
    path = np.r_[down, plateau, up]
    path[-1] = distance
    return np.r_[clock[:ready_index], sample_rows(clock, ready_index + path)]
