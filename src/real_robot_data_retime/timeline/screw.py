"""Independent pickup/approach, smooth ready waits, then coupled insertion."""

import numpy as np

from .holds import compress_static_spans
from .smooth import lift_clock, sample_rows
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


def screw_schedule(
    state,
    action,
    start,
    stop,
    coupled,
    ready,
    position,
    fps,
    *,
    brake_seconds=0.5,
    restart_seconds=0.3,
):
    state, action = np.asarray(state), np.asarray(action)
    if state.shape != action.shape or state.ndim != 2 or state.shape[1] != 14:
        raise ValueError("expected matching N x 14 state and action")
    if not np.isfinite([state, action]).all() or not 0 <= start < stop <= len(state):
        raise ValueError("invalid source values or trim interval")
    if start != int(start) or stop != int(stop):
        raise ValueError("trim bounds must be integer frames")
    intervals, ready = np.asarray(coupled), np.asarray(ready)
    if (
        intervals.shape != (5, 2)
        or ready.shape != (5, 2)
        or not np.isfinite([intervals, ready]).all()
    ):
        raise ValueError(
            "screw task requires five finite coupled and ready frame pairs"
        )
    if np.any(intervals != np.floor(intervals)) or np.any(ready != np.floor(ready)):
        raise ValueError("source boundaries must be integer frames")
    if (
        (intervals[:, 0] >= intervals[:, 1]).any()
        or intervals[0, 0] <= start
        or intervals[-1, 1] >= stop
        or (intervals[1:, 0] <= intervals[:-1, 1]).any()
    ):
        raise ValueError("invalid ordered coupled interval bounds")
    starts = np.r_[start, intervals[:-1, 1]]
    if (ready <= starts[:, None]).any() or (ready >= intervals[:, 0, None]).any():
        raise ValueError("ready poses must precede coupled contact in each round")
    pairs, stages = [], []
    cursor, length = int(start), 0
    for cycle, ((begin, end), peaks) in enumerate(
        zip(intervals.astype(int), ready.astype(int))
    ):
        prepared, resumed, ramps = [], [], []
        for side in range(2):
            sl = slice(side * 7, side * 7 + 7)
            before, after, ramp = ready_clock(
                state[:, sl],
                action[:, sl],
                cursor,
                begin,
                int(peaks[side]),
                fps,
                brake_seconds,
                restart_seconds,
            )
            prepared.append(before)
            resumed.append(after)
            ramps.append(ramp)
        timing = stage_delays(len(prepared[0]) - 1, len(prepared[1]) - 1, position)
        delays = [timing["left_delay_frames"], timing["right_delay_frames"]]
        count = max(len(c) + d for c, d in zip(prepared, delays))
        t = np.arange(count)
        pair = np.column_stack(
            [c[np.clip(t - d, 0, len(c) - 1)] for c, d in zip(prepared, delays)]
        )
        drop = int(bool(pairs))
        stage_start = length - drop
        pairs.append(pair[drop:])
        length += count - drop
        common = {
            "cycle": cycle + 1,
            "source_start": cursor,
            "source_end": int(begin),
            "left_ready_source_frame": int(peaks[0]),
            "right_ready_source_frame": int(peaks[1]),
        }
        independent = dict(
            **common,
            kind="independent",
            output_start=stage_start,
            output_end=length - 1,
            uniform=timing,
        )
        stages.append(independent)
        # Both arms are ready. Offset the shorter final approach so both enter
        # the original coupled interval together at original speed.
        count = max(map(len, resumed))
        tail_delays = [count - len(c) for c in resumed]
        approach_start = length - 1
        tail = np.column_stack(
            [
                np.r_[np.repeat(peak, d), c]
                for peak, d, c in zip(peaks, tail_delays, resumed)
            ]
        )
        pairs.append(tail)
        length += count
        stages.append(
            dict(
                **common,
                kind="approach",
                output_start=approach_start,
                output_end=length - 1,
            )
        )
        transitions = []
        for side in range(2):
            arrival = stage_start + delays[side] + len(prepared[side]) - 1
            restart = approach_start + tail_delays[side]
            transitions.append(
                dict(
                    arm=["left", "right"][side],
                    **ramps[side],
                    brake_start_output_frame=stage_start
                    + delays[side]
                    + ramps[side]["brake_start_index"],
                    arrival_output_frame=arrival,
                    restart_start_output_frame=restart,
                    restart_end_output_frame=restart + round(fps * restart_seconds),
                    hold_frames=restart - arrival,
                )
            )
        independent["transitions"] = transitions
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
        }
    )
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
            "timing_method": "uniform_ready_poses_independent_monotone_stop_restart",
        },
    )


def validate_screw_schedule(left, right, stages, state, action, start, stop):
    left, right = np.asarray(left), np.asarray(right)
    if left.ndim != 1 or left.shape != right.shape or not len(left):
        raise ValueError("invalid paired clocks")
    if not np.isfinite([left, right]).all() or np.any(
        np.diff([left, right], axis=1) < 0
    ):
        raise ValueError("clocks must be finite and monotone")
    if (left[0], right[0], left[-1], right[-1]) != (start, start, stop - 1, stop - 1):
        raise ValueError("episode boundary poses were lost")
    if [s["kind"] for s in stages] != ["independent", "approach", "coupled"] * 5 + [
        "final_storage"
    ]:
        raise ValueError("expected five prepared approaches and coupled insertions")
    checks, smooth_checks = [], []
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
            continue
        active = [np.diff(c[a : b + 1]) > 1e-9 for c in (left, right)]
        if np.any(~active[0] & ~active[1]):
            raise ValueError("introduced both-arms-idle gap")
        for moving in active:
            ids = np.flatnonzero(moving)
            if not len(ids) or not moving[ids[0] : ids[-1] + 1].all():
                raise ValueError("independent arm bundle was interrupted")
        if stage["kind"] == "approach":
            if left[b] != stage["source_end"] or right[b] != stage["source_end"]:
                raise ValueError("approach did not rejoin coupled clocks")
            continue
        onsets = [int(np.flatnonzero(m)[0]) for m in active]
        if onsets[1] - onsets[0] != stage["uniform"]["relative_start_frames"]:
            raise ValueError("uniform stage onset changed")
        for side, transition in enumerate(stage["transitions"]):
            clock = (left, right)[side]
            brake = transition["brake_start_output_frame"]
            arrival = transition["arrival_output_frame"]
            restart = transition["restart_start_output_frame"]
            end = transition["restart_end_output_frame"]
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
                    "ready wait does not decelerate and accelerate smoothly"
                )
            if not np.all(
                clock[arrival : restart + 1] == transition["ready_source_frame"]
            ):
                raise ValueError("ready pose drifted while waiting")
            smooth_checks.append(
                {
                    "cycle": stage["cycle"],
                    "arm": transition["arm"],
                    "hold_frames": int(restart - arrival),
                    "brake_first_rate": float(down[0]),
                    "brake_last_rate": float(down[-1]),
                    "restart_first_rate": float(up[0]),
                    "restart_last_rate": float(up[-1]),
                    "passed": True,
                }
            )
    # Fractional samples cover continuous motion. A source-clock jump exceeding
    # one frame is legal only across a wholly bounded stationary interval.
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
        "protected_intervals": checks,
        "smooth_ready_waits": smooth_checks,
        "skipped_bounded_static_frames": skips,
        "all_nonstationary_source_motion_retained": True,
        "interpolated_frames": {
            "left": int(np.count_nonzero(left != np.floor(left))),
            "right": int(np.count_nonzero(right != np.floor(right))),
        },
    }
