"""Adapt the existing uniform phase scheduler to non-contiguous source maps.

Object independence and shared-space safety are external preconditions.
"""

import numpy as np

from .retime import ArmSegment, EpisodeSegments, build_uniform_schedule_plan


def _phase(left_n, right_n, position):
    segments = EpisodeSegments(
        0, ArmSegment(0, left_n), ArmSegment(0, right_n), 0.0, 0.0, 0.0, 0.0
    )
    return build_uniform_schedule_plan(segments, position, left_n + right_n)


def compose_independent_sources(
    left, right, prefix, suffix, mode="parallel", left_delay=0, right_delay=0
):
    raw = [np.asarray(x) for x in (left, right, prefix, suffix)]
    if any(x.dtype.kind not in "iu" for x in raw):
        raise ValueError("Source paths must contain integer frame indices")
    left, right, prefix, suffix = [x.astype(np.int64) for x in raw]
    for x in (left, right, prefix, suffix):
        if x.ndim != 1 or not len(x) or np.any(x < 0) or np.any(np.diff(x) <= 0):
            raise ValueError("Source paths must be nonempty and strictly increasing")
    if (
        left[0] != right[0]
        or left[-1] != right[-1]
        or prefix[-1] != left[0]
        or suffix[0] != left[-1]
    ):
        raise ValueError("Shared entry/exit source frames must match")
    if any(
        not isinstance(x, (int, np.integer)) or x < 0 for x in (left_delay, right_delay)
    ):
        raise ValueError("Delays must be nonnegative integer frames")
    if mode != "parallel" and (left_delay or right_delay):
        raise ValueError("Offsets only supported in parallel mode")
    if left_delay and right_delay:
        raise ValueError("Both-idle offsets are not supported")
    if left_delay > len(right) or right_delay > len(left):
        raise ValueError("Offset would introduce a both-idle gap")
    if mode == "right_first" or (mode == "parallel" and left_delay == len(right)):
        plan = _phase(len(right), len(left), 0)
        li, ri = plan.right_source_indices, plan.left_source_indices
    elif mode in ("left_first", "parallel"):
        position = 0 if mode == "left_first" else len(left) + left_delay - right_delay
        plan = _phase(len(left), len(right), position)
        li, ri = plan.left_source_indices, plan.right_source_indices
    else:
        raise ValueError("Unknown mode")
    return np.r_[prefix[:-1], left[li], suffix[1:]], np.r_[
        prefix[:-1], right[ri], suffix[1:]
    ]
