"""Reject robot-attached fragments masquerading as initially separate objects."""

import numpy as np


def detached_origin(robots, objects, object_id, pickup_frame, fps):
    stop = min(max(3, round(fps * 0.5)), int(pickup_frame))
    mask = np.asarray(objects[object_id, :stop], bool)
    robot = np.asarray(robots[:stop], bool).any(axis=1)
    area = mask.sum(axis=(1, 2))
    observed = area > 0
    overlap = np.divide((mask & robot).sum(axis=(1, 2)), np.maximum(area, 1))
    fraction = float(np.median(overlap[observed])) if observed.any() else 1.0
    return dict(
        verified=bool(observed.sum() >= 3 and fraction < 0.5),
        median_robot_overlap=fraction,
        observed_frames=int(observed.sum()),
        method="initial_object_separate_from_robot",
    )
