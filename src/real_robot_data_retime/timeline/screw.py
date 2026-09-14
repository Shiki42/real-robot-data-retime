"""Independent pickup/approach, smooth ready waits, then coupled insertion."""

import numpy as np

from .holds import compress_static_spans
from .readiness import final_left_pose
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
        minimum_seconds=0.2,
        guard_seconds=1 / fps,
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
        state[start : end + 1],
        action[start : end + 1],
        [(0, end - start + 1)],
        fps,
        minimum_seconds=0.2,
        guard_seconds=1 / fps,
    ).astype(float)


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
        or (ready[:, 0] > intervals[:, 0]).any()
        or (ready[:, 1] >= intervals[:, 0]).any()
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
        # Finish every left adjustment BEFORE its only possible wait. The
        # remaining source tail must be stationary in both command and state.
        final_left = int(peaks[0])
        detected, _ = final_left_pose(state[:, :7], action[:, :7], cursor, begin)
        if final_left < detected:
            raise ValueError("left final pose still contains an adjustment")
        left_native = native_clock(state[:, :7], action[:, :7], cursor, final_left, fps)
        if final_left < begin:
            left_native = np.r_[left_native, float(begin)]
        clocks = [
            left_native,
            native_clock(state[:, 7:], action[:, 7:], right_start, begin, fps),
        ]
        native_sources = [clock.tolist() for clock in clocks]
        nominal_durations = [
            len(clocks[0]) - 1,
            int(np.flatnonzero(clocks[1] >= peaks[1])[0]),
        ]
        timing = stage_delays(*nominal_durations, position)
        delays = [timing["left_delay_frames"], timing["right_delay_frames"]]
        shift = max(0, retreat_duration - delays[1])
        delays = [d + shift for d in delays]
        finishes = [d + len(c) - 1 for d, c in zip(delays, clocks)]
        late = int(np.argmax(finishes))
        finish = max(finishes)
        preparation_arrivals = [
            delays[0] + len(clocks[0]) - 1 - int(final_left < begin),
            delays[1] + nominal_durations[1],
        ]
        transitions = [
            {
                "arm": arm,
                "mode": "continuous",
                "ready_source_frame": int(peaks[i]),
                "hold_frames": 0,
            }
            for i, arm in enumerate(["left", "right"])
        ]
        early = 1 - late
        if early == 0 and finishes[0] < finish:
            # Left already reached its final insertion pose. No restart motion
            # or residual alignment is permitted between its wait and insertion.
            native = clocks[0][:-1]
            brake = round(fps * brake_seconds)
            distance = round(brake / 2)
            extra = brake - distance
            hold = finish - finishes[0] - extra
            if final_left < begin and hold >= 1 and len(native) > distance:
                onset = len(native) - 1 - distance
                ramp = onset + speed_ramp(brake, distance, False)
                before = np.r_[native[:onset], sample_rows(native, ramp)]
                arrival = delays[0] + len(before) - 1
                clocks[0] = np.r_[
                    before, np.repeat(final_left, finish - arrival - 1), float(begin)
                ]
                transitions[0].update(
                    mode="final_pose_wait",
                    ready_source_frame=final_left,
                    hold_frames=finish - arrival - 1,
                    brake_start_output_frame=delays[0] + onset,
                    arrival_output_frame=arrival,
                    restart_start_output_frame=finish,
                    restart_end_output_frame=finish,
                )
            else:
                clocks[0] = sample_rows(
                    clocks[0],
                    np.linspace(0, len(clocks[0]) - 1, finish - delays[0] + 1),
                )
                transitions[0].update(mode="continuous", ready_source_frame=final_left)
        elif early == 1 and finishes[1] < finish:
            before, after, ramp = ready_clock(
                state[:, 7:],
                action[:, 7:],
                sources[1],
                begin,
                int(peaks[1]),
                fps,
                brake_seconds,
                restart_seconds,
            )
            hold = finish - delays[1] - (len(before) + len(after) - 1)
            if hold >= round(fps * restart_seconds):
                arrival = delays[1] + len(before) - 1
                restart = arrival + hold
                clocks[1] = np.r_[before, np.repeat(peaks[1], hold), after]
                transitions[1].update(
                    ramp,
                    mode="ready_wait",
                    hold_frames=hold,
                    brake_start_output_frame=delays[1] + ramp["brake_start_index"],
                    arrival_output_frame=arrival,
                    restart_start_output_frame=restart,
                    restart_end_output_frame=restart + round(fps * restart_seconds),
                )
            else:
                clocks[1] = sample_rows(
                    clocks[1],
                    np.linspace(0, len(clocks[1]) - 1, finish - delays[1] + 1),
                )
        t = np.arange(finish + 1)
        pair = np.column_stack(
            [c[np.clip(t - d, 0, len(c) - 1)] for c, d in zip(clocks, delays)]
        )
        pair[: retreat_duration + 1, 1] = np.arange(cursor, right_start + 1)
        drop = int(bool(pairs))
        stage_start = length - drop
        for side, tr in enumerate(transitions):
            if tr["mode"] not in ("ready_wait", "final_pose_wait"):
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
                "native_source_clocks": native_sources,
                "pickup_start_output_frames": [stage_start + d for d in delays],
                "left_ready_source_frame": int(peaks[0]),
                "right_ready_source_frame": int(peaks[1]),
                "transitions": transitions,
                "final_ready_source_frames": [int(x) for x in peaks],
                "native_ready_arrival_frames": [
                    stage_start + x for x in preparation_arrivals
                ],
                "late_preparation_arm": ["left", "right"][late],
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
            "timing_method": "final_left_pose_single_wait_with_mandatory_right_retreat",
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
                elif tr["mode"] == "final_pose_wait":
                    arrival = tr["arrival_output_frame"]
                    if not (np.diff(clock[onset : arrival + 1]) > 0).all():
                        raise ValueError("left preparation stopped before final pose")
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
    for stage in stages:
        if stage["kind"] != "independent":
            continue
        if sum(tr["hold_frames"] > 0 for tr in stage["transitions"]) > 1:
            raise ValueError("both arms were scheduled to wait in one preparation")
        final_left = stage["final_ready_source_frames"][0]
        tr = stage["transitions"][0]
        if tr["mode"] == "final_pose_wait":
            a, b = tr["arrival_output_frame"], stage["output_end"]
            if not np.all(left[a:b] == final_left):
                raise ValueError("left moved again after final-pose waiting")
        if tr["mode"] == "final_pose_wait":
            onset = tr["brake_start_output_frame"]
            arrival = tr["arrival_output_frame"]
            native = np.asarray(stage["native_source_clocks"][0])
            progress = np.interp(
                left[onset : arrival + 1], native, np.arange(len(native))
            )
            speed = np.diff(progress)
            if len(speed) < 2 or np.any(np.diff(speed) > 1e-8) or speed[-1] >= speed[0]:
                raise ValueError("left final arrival did not decelerate smoothly")
            for values in (state, action):
                if np.any(
                    np.abs(
                        values[final_left : stage["source_end"] + 1, :7]
                        - values[stage["source_end"], :7]
                    )
                    > np.array([0.05] * 6 + [0.1]) + 1e-6
                ):
                    raise ValueError("left final wait precedes a real adjustment")
    tolerance = np.array([0.3] * 6 + [0.5])
    skips = [0, 0]
    for stage in stages:
        if stage["kind"] != "independent":
            continue
        for side, source in enumerate(stage["native_source_clocks"]):
            source = np.asarray(source)
            if (
                source.ndim != 1
                or not np.isfinite(source).all()
                or not (np.diff(source) > 0).all()
            ):
                raise ValueError("invalid native source clock")
            clock = (left, right)[side]
            onset = stage["pickup_start_output_frames"][side]
            mapped = clock[onset : stage["output_end"] + 1]
            if (mapped[0], mapped[-1]) != (source[0], source[-1]):
                raise ValueError("preparation lost its source boundaries")
            progress = np.interp(mapped, source, np.arange(len(source)))
            if np.any(np.diff(progress) > 1 + 1e-7):
                raise ValueError("nonstationary source motion was skipped")
            for k in np.flatnonzero(np.diff(source) > 1):
                a, b = int(source[k]), int(source[k + 1])
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
        "at_most_one_preparation_waiter": True,
        "no_left_adjustment_after_final_wait": True,
        "skipped_bounded_static_frames": skips,
        "all_nonstationary_source_motion_retained": True,
        "interpolated_frames": {
            "left": int(np.count_nonzero(left != np.floor(left))),
            "right": int(np.count_nonzero(right != np.floor(right))),
        },
    }
