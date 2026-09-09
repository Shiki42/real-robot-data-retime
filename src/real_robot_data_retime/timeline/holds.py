"""Replace measured waiting jitter with a held source pose for editing."""

import numpy as np


def compress_wait(values, start, stop, joint_tolerance=0.3, gripper_tolerance=0.5):
    """Remove an explicitly discovered waiting interval only if pose stays close.

    Bounds are algorithm-discovered source frame indices, [start, stop). The
    departure pose is retained; transitions still require swept collision checks.
    Tolerances use degrees and total jaw aperture mm, respectively.
    """
    x = np.asarray(values, float)
    if x.ndim != 2 or x.shape[1] != 7 or not np.isfinite(x).all():
        raise ValueError("expected finite N x 7 arm trajectory")
    if not 0 <= start < stop <= len(x):
        raise ValueError("invalid waiting interval")
    if min(joint_tolerance, gripper_tolerance) <= 0:
        raise ValueError("positive pose tolerances required")
    threshold = np.array([joint_tolerance] * 6 + [gripper_tolerance])
    # Retain any pose that exceeds the hold tolerance from the last retained
    # pose. Thus genuine gripper opening/closing never disappears as idle.
    kept = list(range(start))
    anchor = start
    kept.append(anchor)
    for t in range(start + 1, stop):
        if np.any(np.abs(x[t] - x[anchor]) > threshold):
            kept.append(t)
            anchor = t
    if kept[-1] != stop - 1:
        kept.append(stop - 1)
    kept.extend(range(stop, len(x)))
    return np.asarray(kept, dtype=np.int64)
