"""Conservative image-space scheduling when metric joints are unavailable."""

import numpy as np
from functools import lru_cache
from .scheduler import schedule_sources
from ..background.clean_plate import dilate
from ..compositing.ownership import arm_foreground
from ..tasks.drawer_constraints import discover_insertion_gate, precedence_gate


def plan_visual(timeline, frames, tracks, segmentation):
    n, h, w = frames.shape[:3]
    robots = np.unpackbits(segmentation["robots"], axis=-1, count=w).astype(bool)
    objects = np.unpackbits(segmentation["objects"], axis=-1, count=w).astype(bool)
    events = timeline["episodes"]
    sources = []
    for side in ["left", "right"]:
        own = [e for e in events if e["robot_id"] == side]
        if own:
            sources.append(
                np.arange(
                    min(e["approach_start"] for e in own),
                    max(e["retract_end"] for e in own) + 1,
                )
            )
        else:
            sources.append(np.arange(n))
    dependency = lambda i, j: True
    drawer = timeline["task"] == "drawer"
    if drawer:
        (event,) = events
        motion = timeline["drawer_motion"]
        a, b = motion["open_frame"], motion["close_start"]
        gate, interior = discover_insertion_gate(
            frames[a],
            tracks["objects"][event["object_id"]],
            event["pickup_frame"],
            event["release_frame"],
        )
        # The open drawer's source dwell becomes a held right pose. The actual
        # moving parts before and after this dwell are retained at source speed.
        sources[1] = np.r_[np.arange(a + 1), np.arange(b, n)]
        outside = []
        for t in range(event["release_frame"] + 1, n):
            mask = robots[t, 0]
            if not np.any(mask & interior):
                outside.append(t)
        if not outside:
            raise ValueError("left withdrawal from drawer is not visible")
        withdrawal = outside[0]
        sources[0] = np.arange(min(e["approach_start"] for e in events), n)

        def dependency(i, j):
            return precedence_gate(
                int(sources[0][i]),
                int(sources[1][j]),
                safe_approach_end=gate,
                open_frame=a,
                withdrawal_frame=withdrawal,
                close_start=b,
            )

    @lru_cache(maxsize=4096)
    def mask(side, index):
        foreground = arm_foreground(
            robots, objects, events, side, int(sources[side][index])
        )
        return np.packbits(dilate(foreground, 2))

    @lru_cache(None)
    def clear(i, j):
        return not np.any(mask(0, i) & mask(1, j))

    def safe(i, j, ni, nj):
        # Sweep union of adjacent silhouettes; this is a projected-occlusion
        # constraint, never reported as a metric robot collision guarantee.
        return clear(i, j) and clear(ni, nj) and clear(i, nj) and clear(ni, j)

    schedule = schedule_sources(
        len(sources[0]),
        len(sources[1]),
        safe,
        dependency=dependency,
        left_priority=not drawer,
    )
    return (
        sources[0][schedule.left],
        sources[1][schedule.right],
        dict(
            collision_scope="projected silhouettes only; no joint-space verification",
            left_wait_frames=schedule.left_waits,
            right_wait_frames=schedule.right_waits,
            output_frames=len(schedule.left),
        ),
    )
