"""Independent pickup/approach, smooth ready waits, then coupled insertion."""

import numpy as np

from .holds import compress_static_spans
from .smooth import lift_clock, sample_rows, speed_ramp
from .uniform import stage_delays


def ready_clock(
    state, action, start, coupled_start, ready, fps, brake_seconds, restart_seconds
):
    """Reuse the existing monotone stop/restart clock on a compressed prefix."""
    brake_distance = round(round(fps * brake_seconds) / 2)
    local_ready = ready - start
    kept = compress_static_spans(
        state[start : coupled_start + 1],
        action[start : coupled_start + 1],
        [(0, local_ready - brake_distance)],
        fps,
    )
    peak = int(np.flatnonzero(kept == local_ready)[0])
    path, stop, ramp = lift_clock(
        0,
        len(kept) - 1,
        peak,
        fps,
        brake_seconds=brake_seconds,
        restart_seconds=restart_seconds,
    )
    clock = start + sample_rows(kept.astype(float), path)
    ramp.update(
        ready_source_frame=int(ready),
        brake_source_start=int(kept[ramp["brake_source_start"]] + start),
        restart_source_end=int(kept[ramp["restart_source_end"]] + start),
        collapsed_static_frames=int(coupled_start - start + 1 - len(kept)),
    )
    return clock[: stop + 1], clock[stop + 1 :], ramp


def native_clock(state, action, start, end, fps):
    return start + compress_static_spans(
        state[start : end + 1], action[start : end + 1], [(0, end - start + 1)], fps
    ).astype(float)


def slow_approach(clock, extra, fps):
    """Spend a short lead without stopping, with unit speed at both joins."""
    distance = min(len(clock) - 1, max(round(fps * 0.8), extra + 1))
    if extra >= distance:
        raise ValueError("insufficient approach for a positive-speed timing adjustment")
    duration = distance + extra
    down, up = duration // 2, duration - duration // 2
    stop = np.r_[
        speed_ramp(down, down / 2, False), down / 2 + speed_ramp(up, up / 2, True)[1:]
    ]
    t = np.arange(duration + 1, dtype=float)
    progress = t - (2 * extra / duration) * (t - stop)
    progress[-1] = distance
    begin = len(clock) - 1 - distance
    return np.r_[clock[:begin], sample_rows(clock, begin + progress)], begin


def screw_schedule(
    state,
    action,
    start,
    stop,
    coupled,
    ready,
    retreat_ends,
    position,
    fps,
    *,
    brake_seconds=0.5,
    restart_seconds=0.3,
):
    state, action = np.asarray(state), np.asarray(action)
    intervals, ready, retreat_ends = (
        np.asarray(coupled),
        np.asarray(ready),
        np.asarray(retreat_ends),
    )
    if (
        state.shape != action.shape
        or state.ndim != 2
        or state.shape[1] != 14
        or not np.isfinite([state, action]).all()
    ):
        raise ValueError("expected matching finite N x 14 state and action")
    if not 0 <= start < stop <= len(state) or start != int(start) or stop != int(stop):
        raise ValueError("invalid integer trim bounds")
    if intervals.shape != (5, 2) or ready.shape != (5, 2) or retreat_ends.shape != (5,):
        raise ValueError("expected five insertion, ready and retreat annotations")
    for values in (intervals, ready, retreat_ends):
        if not np.isfinite(values).all() or np.any(values != np.floor(values)):
            raise ValueError("annotations must be finite integer frames")
    starts = np.r_[start, intervals[:-1, 1]]
    if (
        (intervals[:, 0] >= intervals[:, 1]).any()
        or (intervals[:, 0] <= starts).any()
        or intervals[-1, 1] >= stop
        or (ready <= starts[:, None]).any()
        or (ready >= intervals[:, 0, None]).any()
        or (retreat_ends < intervals[:, 1]).any()
        or retreat_ends[-1] >= stop
        or (retreat_ends[:-1] >= ready[1:, 1]).any()
    ):
        raise ValueError("invalid ordered source intervals")
    pairs, stages = [], []
    cursor, length = int(start), 0
    for cycle, ((begin, end), peaks) in enumerate(
        zip(intervals.astype(int), ready.astype(int))
    ):
        right_start = int(start if cycle == 0 else retreat_ends[cycle - 1])
        retreat_duration = right_start - cursor
        sources = [cursor, right_start]
        clocks = [
            native_clock(
                state[:, side * 7 : side * 7 + 7],
                action[:, side * 7 : side * 7 + 7],
                sources[side],
                begin,
                fps,
            )
            for side in range(2)
        ]
        preparation_durations = [
            int(np.flatnonzero(clock >= peak)[0]) for clock, peak in zip(clocks, peaks)
        ]
        timing = stage_delays(*preparation_durations, position)
        delays = [timing["left_delay_frames"], timing["right_delay_frames"]]
        shift = max(0, retreat_duration - delays[1])
        delays = [d + shift for d in delays]
        finishes = [d + len(c) - 1 for d, c in zip(delays, clocks)]
        finish = max(finishes)
        transitions = [
            {
                "arm": ["left", "right"][side],
                "mode": "continuous",
                "ready_source_frame": int(peaks[side]),
                "hold_frames": 0,
            }
            for side in range(2)
        ]
        if finishes[0] != finishes[1]:
            early = int(np.argmin(finishes))
            sl = slice(early * 7, early * 7 + 7)
            before, after, ramp = ready_clock(
                state[:, sl],
                action[:, sl],
                sources[early],
                begin,
                int(peaks[early]),
                fps,
                brake_seconds,
                restart_seconds,
            )
            hold = finish - (delays[early] + len(before) + len(after) - 1)
            if hold >= 0:
                clocks[early] = np.r_[before, np.repeat(peaks[early], hold), after]
                transitions[early].update(
                    ramp,
                    mode="ready_wait",
                    hold_frames=int(hold),
                    brake_start_output_frame=delays[early] + ramp["brake_start_index"],
                    arrival_output_frame=delays[early] + len(before) - 1,
                    restart_start_output_frame=finish - len(after),
                    restart_end_output_frame=finish
                    - len(after)
                    + round(fps * restart_seconds),
                )
            else:
                clocks[early], slow_start = slow_approach(
                    clocks[early], finish - finishes[early], fps
                )
                transitions[early].update(
                    mode="flow_through",
                    slow_start_output_frame=delays[early] + slow_start,
                )
        t = np.arange(finish + 1)
        pair = np.column_stack(
            [c[np.clip(t - d, 0, len(c) - 1)] for c, d in zip(clocks, delays)]
        )
        pair[: retreat_duration + 1, 1] = np.arange(cursor, right_start + 1)
        drop = int(bool(pairs))
        stage_start = length - drop
        for side, tr in enumerate(transitions):
            if tr["mode"] != "ready_wait":
                tr["arrival_output_frame"] = int(
                    np.flatnonzero(pair[:, side] >= peaks[side])[0]
                )
            for key in list(tr):
                if key.endswith("_output_frame"):
                    tr[key] += stage_start
        pairs.append(pair[drop:])
        length += len(pair) - drop
        stages.append(
            {
                "kind": "independent",
                "cycle": cycle + 1,
                "source_start": cursor,
                "source_end": int(begin),
                "output_start": stage_start,
                "output_end": length - 1,
                "uniform": timing,
                "pickup_start_output_frames": [stage_start + d for d in delays],
                "left_ready_source_frame": int(peaks[0]),
                "right_ready_source_frame": int(peaks[1]),
                "transitions": transitions,
                "mandatory_previous_right_retreat": {
                    "previous_cycle": cycle,
                    "source_start": cursor,
                    "source_end": right_start,
                    "output_start": stage_start,
                    "output_end": stage_start + retreat_duration,
                },
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
            "source_start": cursor,
            "source_end": stop - 1,
            "output_start": length - 1,
            "output_end": length + len(tail) - 1,
            "mandatory_previous_right_retreat": {
                "previous_cycle": 5,
                "source_start": cursor,
                "source_end": int(retreat_ends[-1]),
                "output_start": length - 1,
                "output_end": length - 1 + int(retreat_ends[-1]) - cursor,
            },
        }
    )
    for index, stage in enumerate(stages):
        if stage["kind"] == "coupled":
            stage["right_retreat"] = stages[index + 1][
                "mandatory_previous_right_retreat"
            ]
    pair = np.concatenate(pairs)
    validation = validate_screw_schedule(
        pair[:, 0], pair[:, 1], stages, state, action, start, stop
    )
    return (
        pair[:, 0],
        pair[:, 1],
        {
            "stages": stages,
            "validation": validation,
            "position": float(position),
            "fps": float(fps),
            "output_frames": len(pair),
            "timing_method": "continuous_late_arm_with_mandatory_right_retreat",
        },
    )


def validate_screw_schedule(left, right, stages, state, action, start, stop):
    left, right = np.asarray(left), np.asarray(right)
    if (
        left.ndim != 1
        or left.shape != right.shape
        or not len(left)
        or not np.isfinite([left, right]).all()
        or np.any(np.diff([left, right], axis=1) < 0)
    ):
        raise ValueError("invalid monotone paired clocks")
    if (left[0], right[0], left[-1], right[-1]) != (start, start, stop - 1, stop - 1):
        raise ValueError("episode boundary poses were lost")
    if [s["kind"] for s in stages] != ["independent", "coupled"] * 5 + [
        "final_storage"
    ]:
        raise ValueError("expected five preparation/insertion rounds")
    protected, smooth, retreats = [], [], []
    for s in stages:
        a, b = s["output_start"], s["output_end"]
        if "mandatory_previous_right_retreat" in s:
            r = s["mandatory_previous_right_retreat"]
            ra, rb = r["output_start"], r["output_end"]
            if not np.array_equal(
                right[ra : rb + 1], np.arange(r["source_start"], r["source_end"] + 1)
            ):
                raise ValueError("mandatory right retreat was delayed or retimed")
            retreats.append(dict(**r, passed=True))
        if s["kind"] in ("coupled", "final_storage"):
            expected = np.arange(s["source_start"], s["source_end"] + 1)
            if not np.array_equal(left[a : b + 1], expected) or not np.array_equal(
                right[a : b + 1], expected
            ):
                raise ValueError("coupled insertion or final storage was retimed")
            protected.append(
                {
                    "kind": s["kind"],
                    "cycle": s.get("cycle"),
                    "frames": len(expected),
                    "passed": True,
                }
            )
        else:
            if np.any(
                (np.diff(left[a : b + 1]) == 0) & (np.diff(right[a : b + 1]) == 0)
            ):
                raise ValueError("introduced both-arms-idle gap")
            onsets = s["pickup_start_output_frames"]
            if onsets[1] - onsets[0] != s["uniform"]["relative_start_frames"]:
                raise ValueError("uniform pickup onset changed")
            for side, tr in enumerate(s["transitions"]):
                clock = (left, right)[side]
                onset = onsets[side]
                if tr["mode"] == "ready_wait":
                    brake, arrival, restart, end = (
                        tr[k]
                        for k in [
                            "brake_start_output_frame",
                            "arrival_output_frame",
                            "restart_start_output_frame",
                            "restart_end_output_frame",
                        ]
                    )
                    down = np.diff(clock[brake : arrival + 1])
                    up = np.diff(clock[restart : end + 1])
                    if not (
                        len(down) >= 2
                        and len(up) >= 2
                        and np.all(np.diff(down) <= 1e-8)
                        and np.all(np.diff(up) >= -1e-8)
                        and down[-1] < down[0]
                        and up[0] < up[-1]
                    ):
                        raise ValueError(
                            "wait does not decelerate and accelerate smoothly"
                        )
                    if not np.all(
                        clock[arrival : restart + 1] == tr["ready_source_frame"]
                    ):
                        raise ValueError("ready pose drifted while waiting")
                    if (
                        not (np.diff(clock[onset : arrival + 1]) > 0).all()
                        or not (np.diff(clock[restart : b + 1]) > 0).all()
                    ):
                        raise ValueError("independent arm bundle was interrupted")
                    smooth.append(
                        {
                            "cycle": s["cycle"],
                            "arm": tr["arm"],
                            "hold_frames": restart - arrival,
                            "brake_first_rate": float(down[0]),
                            "brake_last_rate": float(down[-1]),
                            "restart_first_rate": float(up[0]),
                            "restart_last_rate": float(up[-1]),
                            "passed": True,
                        }
                    )
                elif not (np.diff(clock[onset : b + 1]) > 0).all():
                    raise ValueError("late arm stopped before insertion")
    for stage in stages:
        if stage["kind"] == "coupled":
            retreat = stage["right_retreat"]
            expected = np.arange(stage["source_start"], retreat["source_end"] + 1)
            if not np.array_equal(
                right[stage["output_start"] : retreat["output_end"] + 1], expected
            ):
                raise ValueError(
                    "right insertion and retreat are not one continuous task"
                )
    tolerance = np.array([0.3] * 6 + [0.5])
    skips = [0, 0]
    for side, clock in enumerate((left, right)):
        for k in np.flatnonzero(np.diff(clock) > 1 + 1e-8):
            a, b = int(np.floor(clock[k])), int(np.ceil(clock[k + 1]))
            for values in (state, action):
                if np.any(
                    np.ptp(values[a : b + 1, side * 7 : side * 7 + 7], axis=0)
                    > tolerance + 1e-6
                ):
                    raise ValueError("nonstationary source motion was skipped")
            skips[side] += b - a - 1
    return {
        "passed": True,
        "protected_intervals": protected,
        "smooth_ready_waits": smooth,
        "mandatory_right_retreats": retreats,
        "late_arm_continuous_through_insertion": True,
        "skipped_bounded_static_frames": skips,
        "all_nonstationary_source_motion_retained": True,
        "interpolated_frames": {
            "left": int(np.count_nonzero(left != np.floor(left))),
            "right": int(np.count_nonzero(right != np.floor(right))),
        },
    }
