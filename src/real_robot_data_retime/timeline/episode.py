"""Manipulation episode boundaries derived from visual motion and events."""

import numpy as np
from scipy.ndimage import median_filter
from ..interaction.evidence import stable_runs


def assign_boundaries(events, grippers, fps, image_width):
    result = []
    for side, name in enumerate(["left", "right"]):
        own = sorted(
            (dict(e) for e in events if e["robot_id"] == name),
            key=lambda e: e["pickup_frame"],
        )
        xy = grippers[:, side]
        speed = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        finite = np.isfinite(speed)
        filtered = median_filter(np.where(finite, speed, np.inf), size=5)
        stops = stable_runs(
            (filtered < image_width * 0.001) & finite, max(3, round(fps * 0.2))
        )
        moving = np.flatnonzero((filtered > image_width * 0.0015) & finite)
        for k, event in enumerate(own):
            floor = own[k - 1]["release_frame"] + 1 if k else 0
            prior = [b for a, b in stops if floor <= b < event["grasp_start"]]
            approach = max(prior) if prior else floor
            active = moving[(moving >= approach) & (moving < event["grasp_start"])]
            event["approach_start"] = (
                int(active[0]) if len(active) else event["grasp_start"]
            )
            ceiling = own[k + 1]["grasp_start"] if k + 1 < len(own) else len(xy) - 1
            after = [a for a, b in stops if event["release_frame"] < a <= ceiling]
            event["retract_end"] = int(min(after)) if after else int(ceiling)
            if (
                not event["approach_start"]
                <= event["grasp_start"]
                <= event["pickup_frame"]
                < event["release_frame"]
                <= event["retract_end"]
            ):
                raise ValueError(
                    "visually inferred episode boundaries are inconsistent"
                )
            result.append(event)
    return sorted(result, key=lambda e: e["pickup_frame"])


def object_state(frame, episode):
    """State transitions use inferred source times; pickup removes the origin."""
    if frame < episode["grasp_start"]:
        return "AT_ORIGIN"
    if frame < episode["pickup_frame"]:
        return "GRASPING"
    if frame < episode["release_frame"]:
        return "HELD"
    if frame == episode["release_frame"]:
        return "RELEASING"
    return "PLACED"
