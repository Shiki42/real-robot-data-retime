"""Replace measured waiting jitter with a held source pose for editing."""

import numpy as np


def append_terminal_hold(left_indices, right_indices, fps, seconds=2.0):
    """Explicit synthetic still hold: repeat original final poses and RGB frames."""
    if fps <= 0 or seconds < 0:
        raise ValueError("invalid terminal hold duration")
    left = np.asarray(left_indices)
    right = np.asarray(right_indices)
    if left.ndim != 1 or right.shape != left.shape or not len(left):
        raise ValueError("source maps must be nonempty and equally sized")
    count = int(np.ceil(fps * seconds))
    return (
        np.r_[left, np.repeat(left[-1], count)],
        np.r_[right, np.repeat(right[-1], count)],
        dict(
            start_frame=len(left),
            end_frame=len(left) + count,
            frames=count,
            method="repeated_source_boundary",
        ),
    )


def compress_static_spans(
    state,
    action,
    intervals,
    fps,
    *,
    joint_tolerance=0.3,
    gripper_tolerance=0.5,
    minimum_seconds=0.5,
    guard_seconds=0.2,
):
    """Collapse only bounded stationary spans outside protected manipulation.

    Both measured and commanded pose ranges must stay inside tolerance for the
    entire span. Slow accumulated approach motion therefore remains at source
    speed; unlike a velocity threshold, bounded jitter cannot accumulate drift.
    """
    state, action = np.asarray(state, float), np.asarray(action, float)
    if (
        state.ndim != 2
        or state.shape[1] != 7
        or state.shape != action.shape
        or not np.isfinite([state, action]).all()
    ):
        raise ValueError("expected matching finite N x 7 measured and commanded poses")
    if (
        min(fps, joint_tolerance, gripper_tolerance, minimum_seconds, guard_seconds)
        <= 0
    ):
        raise ValueError("stationary-span parameters must be positive")
    values = np.concatenate([state, action], axis=1)
    limit = np.tile([joint_tolerance] * 6 + [gripper_tolerance], 2)
    kept = np.ones(len(values), bool)
    minimum = max(3, int(np.ceil(fps * minimum_seconds)))
    guard = max(1, round(fps * guard_seconds))
    for begin, end in intervals:
        if not 0 <= begin <= end <= len(values):
            raise ValueError("stationary search interval outside source")
        start = begin
        while start < end:
            low = values[start].copy()
            high = low.copy()
            stop = start + 1
            while stop < end:
                lo = np.minimum(low, values[stop])
                hi = np.maximum(high, values[stop])
                if np.any(hi - lo > limit):
                    break
                low, high = lo, hi
                stop += 1
            if stop - start >= max(minimum, 2 * guard + 1):
                kept[start + guard : stop - guard] = False
            start = stop
    return np.flatnonzero(kept)
