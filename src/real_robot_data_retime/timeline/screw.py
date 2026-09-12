"""Repeated independent pickup stages separated by original coupled insertions."""

import itertools

import numpy as np

from .holds import compress_static_spans
from .uniform import stage_delays


def screw_schedule(state, action, start, stop, coupled, position, fps):
    """Bounds are inclusive, except the episode's exclusive ``stop``.

    Each left storage+pickup remains a single stage. Only bounded measured AND
    commanded stillness is collapsed; every other source frame is preserved.
    Coupled windows, including approach/release guards, replay one-for-one.
    """
    state, action = np.asarray(state), np.asarray(action)
    if state.shape != action.shape or state.ndim != 2 or state.shape[1] != 14:
        raise ValueError("expected matching N x 14 state and action")
    if not np.isfinite([state, action]).all() or not 0 <= start < stop <= len(state):
        raise ValueError("invalid source values or trim interval")
    if start != int(start) or stop != int(stop):
        raise ValueError("trim bounds must be integer frames")
    intervals = np.asarray(coupled)
    if intervals.shape != (5, 2) or not np.isfinite(intervals).all():
        raise ValueError("screw task requires five finite coupled intervals")
    if np.any(intervals != np.floor(intervals)):
        raise ValueError("coupled boundaries must be integer frames")
    if (
        (intervals[:, 0] >= intervals[:, 1]).any()
        or intervals[0, 0] <= start
        or intervals[-1, 1] >= stop
    ):
        raise ValueError("invalid coupled interval bounds")
    if (intervals[1:, 0] <= intervals[:-1, 1]).any():
        raise ValueError("coupled windows must be separated and ordered")
    pairs, stages = [], []
    cursor = int(start)
    length = 0
    for cycle, (begin, end) in enumerate(intervals.astype(int)):
        clocks = []
        skipped = []
        for side in range(2):
            sl = slice(7 * side, 7 * side + 7)
            local = slice(cursor, begin + 1)
            clock = (
                compress_static_spans(
                    state[local, sl], action[local, sl], [(0, begin - cursor + 1)], fps
                )
                + cursor
            )
            clocks.append(clock)
            skipped.append(int(begin - cursor + 1 - len(clock)))
        timing = stage_delays(len(clocks[0]) - 1, len(clocks[1]) - 1, position)
        delays = [timing["left_delay_frames"], timing["right_delay_frames"]]
        count = max(len(c) + d for c, d in zip(clocks, delays))
        t = np.arange(count)
        pair = np.column_stack(
            [c[np.clip(t - d, 0, len(c) - 1)] for c, d in zip(clocks, delays)]
        )
        # Adjacent stages share a boundary pose, stored only once.
        drop = int(bool(pairs))
        stage_start = length - drop
        pairs.append(pair[drop:])
        length += count - drop
        stages.append(
            {
                "kind": "independent",
                "cycle": cycle + 1,
                "source_start": cursor,
                "source_end": int(begin),
                "output_start": stage_start,
                "output_end": length - 1,
                "uniform": timing,
                "collapsed_static_frames": skipped,
            }
        )
        sync_start = length - 1
        sync = np.arange(begin + 1, end + 1)
        pairs.append(np.column_stack([sync, sync]))
        length += len(sync)
        stages.append(
            {
                "kind": "coupled",
                "cycle": cycle + 1,
                "source_start": int(begin),
                "source_end": int(end),
                "output_start": sync_start,
                "output_end": length - 1,
            }
        )
        cursor = int(end)
    tail = np.arange(cursor + 1, stop)
    pairs.append(np.column_stack([tail, tail]))
    stages.append(
        {
            "kind": "final_storage",
            "output_start": length - 1,
            "output_end": length + len(tail) - 1,
            "source_start": cursor,
            "source_end": stop - 1,
        }
    )
    result = np.concatenate(pairs)
    validation = validate_screw_schedule(
        result[:, 0], result[:, 1], stages, state, action, start, stop
    )
    return (
        result[:, 0],
        result[:, 1],
        {
            "stages": stages,
            "validation": validation,
            "position": float(position),
            "fps": float(fps),
            "output_frames": len(result),
            "timing_method": "uniform_relative_onset_native_motion_bounded_static_compression",
        },
    )


def validate_screw_schedule(left, right, stages, state, action, start, stop):
    left, right = np.asarray(left), np.asarray(right)
    if left.ndim != 1 or right.shape != left.shape or not len(left):
        raise ValueError("invalid paired clocks")
    if not np.isfinite([left, right]).all() or np.any(
        np.diff([left, right], axis=1) < 0
    ):
        raise ValueError("clocks must be finite and monotone")
    if np.any(np.asarray([left, right]) != np.floor([left, right])):
        raise ValueError("native source clocks must be integer frames")
    if (left[0], right[0], left[-1], right[-1]) != (start, start, stop - 1, stop - 1):
        raise ValueError("episode boundary poses were lost")
    if [stage["kind"] for stage in stages] != ["independent", "coupled"] * 5 + [
        "final_storage"
    ]:
        raise ValueError("expected five independent/coupled rounds and final storage")
    checks = []
    for stage in stages:
        a, b = stage["output_start"], stage["output_end"]
        if stage["kind"] in ("coupled", "final_storage"):
            expected = np.arange(stage["source_start"], stage["source_end"] + 1)
            if not np.array_equal(left[a : b + 1], expected) or not np.array_equal(
                right[a : b + 1], expected
            ):
                raise ValueError("coupled insertion or final storage was retimed")
            checks.append(
                {
                    "kind": stage["kind"],
                    "cycle": stage.get("cycle"),
                    "frames": len(expected),
                    "passed": True,
                }
            )
        else:
            starts = [
                int(np.flatnonzero(np.diff(c[a : b + 1]) > 0)[0]) for c in (left, right)
            ]
            for clock in (left, right):
                active = np.diff(clock[a : b + 1]) > 0
                moving = np.flatnonzero(active)
                if not active[moving[0] : moving[-1] + 1].all():
                    raise ValueError("independent arm bundle was interrupted")
            if starts[1] - starts[0] != stage["uniform"]["relative_start_frames"]:
                raise ValueError("uniform stage onset changed")
            if np.any(
                (np.diff(left[a : b + 1]) == 0) & (np.diff(right[a : b + 1]) == 0)
            ):
                raise ValueError("introduced both-arms-idle gap")
    # Every skip must be bounded stillness across its full interval, not merely
    # low speed at either endpoint. Check both state and commanded action.
    tolerance = np.array([0.3] * 6 + [0.5])
    skipped = [0, 0]
    for side, clock in enumerate((left, right)):
        unique = np.unique(clock).astype(int)
        for a, b in itertools.pairwise(unique):
            if b - a <= 1:
                continue
            for values in (state, action):
                if np.any(
                    np.ptp(values[a : b + 1, side * 7 : side * 7 + 7], axis=0)
                    > tolerance + 1e-6
                ):
                    raise ValueError("nonstationary source motion was skipped")
            skipped[side] += int(b - a - 1)
    return {
        "passed": True,
        "protected_intervals": checks,
        "skipped_bounded_static_frames": skipped,
        "all_nonstationary_source_frames_retained": True,
    }
