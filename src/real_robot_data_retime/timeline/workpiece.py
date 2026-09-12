"""Non-preemptive workpiece execution with independent approach ramps."""

import numpy as np

from .smooth import speed_ramp
from ..background.clean_plate import dilate
from ..compositing.ownership import arm_foreground
from ..tasks.workpiece import workpiece_events


def approach_clock(start, end, stops, fps):
    """Only approach/return segments brake; an admitted execution stays at 1x."""
    down, up = round(fps * 0.5), round(fps * 0.3)
    if min(down, up) < 2:
        raise ValueError("independent ramps require at least two frame intervals")
    bounds = [start, *stops, end]
    clock = [float(start)]
    holds = []
    transitions = []
    for segment, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        if b < a:
            raise ValueError("approach stops must preserve source order")
        accelerating = segment > 0
        braking = segment < len(stops)
        distance = b - a
        if not distance:
            if braking:
                holds.append(len(clock) - 1)
            continue
        # A short initial approach must not be stretched into a slow crawl.
        # Start at source speed and shorten only its own stopping ramp.
        brake_duration = min(down, max(2, 2 * distance)) if segment == 0 else down
        brake_distance = round(brake_duration / 2)
        nominal = (round(up / 2) if accelerating else 0) + (
            brake_distance if braking else 0
        )
        plateau = max(0, int(np.floor(distance - nominal)))
        rate = distance / (nominal + plateau)
        local = [0.0]
        if accelerating:
            local.extend((rate * speed_ramp(up, round(up / 2), True)[1:]).tolist())
        if plateau:
            local.extend((local[-1] + rate * np.arange(1, plateau + 1)).tolist())
        if braking:
            begin = len(clock) - 1 + len(local) - 1
            source_begin = a + local[-1]
            local.extend(
                (
                    local[-1]
                    + rate * speed_ramp(brake_duration, brake_distance, False)[1:]
                ).tolist()
            )
        local[-1] = distance
        clock.extend((a + np.asarray(local[1:])).tolist())
        if braking:
            holds.append(len(clock) - 1)
            transitions.append(
                dict(
                    source_frame=b,
                    brake_source_start=source_begin,
                    brake_start_index=begin,
                    brake_intervals=brake_duration,
                    stop_index=holds[-1],
                )
            )
    return np.asarray(clock), holds, transitions


WAIT_CLEARANCE_WIDTH_FRACTION = 0.02


def staging_candidates(events, robots, objects, fps):
    own = workpiece_events(events)
    down_distance, up_distance = (
        round(round(fps * 0.5) / 2),
        round(round(fps * 0.3) / 2),
    )

    def mask(side, t):
        return dilate(arm_foreground(robots, objects, events, side, t), 2)

    def safe_stops(side, begin, end, owner, first, last):
        sweep = np.zeros(robots.shape[-2:], bool)
        for t in range(first, last + 1):
            sweep |= mask(owner, t)
        # Reserve additional image-space distance only at waiting poses.
        sweep = dilate(
            sweep, max(1, round(robots.shape[-1] * WAIT_CLEARANCE_WIDTH_FRACTION))
        )
        candidates = [
            t for t in range(end, begin - 1, -1) if not np.any(mask(side, t) & sweep)
        ]
        if not candidates:
            raise ValueError("no staging pose clears the preceding workpiece execution")
        return candidates

    left1, left2 = own[0]
    right1, right2 = own[1]
    right_stop1 = safe_stops(
        1,
        right1["approach_start"],
        right1["pickup_frame"] - up_distance,
        0,
        left1["approach_start"],
        left1["release_evidence"]["clearance_frame"],
    )
    left_stop = safe_stops(
        0,
        left1["release_frame"] + down_distance,
        left2["pickup_frame"] - up_distance,
        1,
        right1["pickup_frame"],
        right1["release_frame"],
    )
    right_stop2 = safe_stops(
        1,
        right1["release_frame"] + down_distance,
        right2["pickup_frame"] - up_distance,
        0,
        left2["pickup_frame"],
        left2["retract_end"],
    )
    return [left_stop, right_stop1, right_stop2]


def workpiece_sources(events, fps, stops):
    own = workpiece_events(events)
    left1, left2 = own[0]
    right1, right2 = own[1]
    left_stop, right_stop1, right_stop2 = stops
    up_distance = round(round(fps * 0.3) / 2)
    left, lh, lt = approach_clock(
        left1["approach_start"], left2["retract_end"], [left_stop], fps
    )
    right, rh, rt = approach_clock(
        right1["approach_start"],
        right2["retract_end"],
        [right_stop1, right_stop2],
        fps,
    )
    stages = dict(
        wait_source_frames=dict(left=[left_stop], right=[right_stop1, right_stop2]),
        protected_source_intervals=dict(
            left=[
                [left1["approach_start"], left1["release_frame"]],
                [left_stop + up_distance, left2["retract_end"]],
            ],
            right=[
                [right_stop1 + up_distance, right1["release_frame"]],
                [right_stop2 + up_distance, right2["retract_end"]],
            ],
        ),
        policy="admitted_execution_cannot_be_preempted",
        waiting_clearance_width_fraction=WAIT_CLEARANCE_WIDTH_FRACTION,
        right_preparation=dict(
            source_start_frame=right1["approach_start"],
            source_end_frame=right_stop1,
        ),
        start_policy="synchronous_original_approaches",
        initial_source_frames=[left1["approach_start"], right1["approach_start"]],
        smoothing="independent_approach_clocks",
        onset_delay_frames=[0, 0],
        admission_gates=[
            [1, right_stop1, 0, left1["pickup_frame"]],
            [0, left_stop, 1, right1["pickup_frame"]],
            [1, right_stop2, 0, left2["pickup_frame"]],
        ],
        ramps=dict(left=lt, right=rt),
    )
    return (
        [left, right],
        [set(lh) | {len(left) - 1}, set(rh) | {len(right) - 1}],
        stages,
    )


def admission_allowed(left, right, stages):
    clocks = (left, right)
    return all(
        clocks[side] <= stop or clocks[owner] > pickup
        for side, stop, owner, pickup in stages["admission_gates"]
    )


def verify_uninterrupted(left, right, stages):
    for index, (side, clock) in enumerate([("left", left), ("right", right)]):
        clock = clock[stages["onset_delay_frames"][index] :]
        if clock[0] != stages["initial_source_frames"][index]:
            raise ValueError("an arm's original approach was omitted")
        has_approach = (
            index == 0 or clock[0] < stages["right_preparation"]["source_end_frame"]
        )
        if has_approach and not clock[1] > clock[0]:
            raise ValueError("an arm was delayed instead of starting its approach")
        for start, end in stages["protected_source_intervals"][side]:
            selected = (
                (clock[:-1] >= start - 1e-9)
                & (clock[1:] <= end + 1e-9)
                & (clock[:-1] < end - 1e-9)
            )
            if not np.allclose(np.diff(clock)[selected], 1, atol=1e-8, rtol=0):
                raise ValueError(f"{side} execution was interrupted after admission")
