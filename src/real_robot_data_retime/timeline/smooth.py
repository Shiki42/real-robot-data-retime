"""Continuous source clocks for a gated pick/lift, open, and insert task."""

import numpy as np


def sample_rows(values, clock):
    values, clock = np.asarray(values), np.asarray(clock, float)
    if (
        clock.ndim != 1
        or not len(clock)
        or not np.isfinite(clock).all()
        or clock.min() < 0
        or clock.max() > len(values) - 1
    ):
        raise ValueError("invalid continuous source clock")
    lo = np.floor(clock).astype(int)
    hi = np.ceil(clock).astype(int)
    weight = (clock - lo).reshape((-1,) + (1,) * (values.ndim - 1))
    return values[lo] + weight * (values[hi] - values[lo])


def speed_ramp(duration, distance, up):
    u = np.arange(duration + 1, dtype=float) / duration
    correction = 30 * (distance / duration - 0.5)
    if abs(correction) > 3 + 1e-10:
        raise ValueError("ramp cannot remain monotone at this frame rate")
    integral = u**3 - 0.5 * u**4 if up else u - u**3 + 0.5 * u**4
    integral += correction * (u**3 / 3 - u**4 / 2 + u**5 / 5)
    result = duration * integral
    result[0], result[-1] = 0, distance
    return result


def lift_clock(start, stop, peak, fps, *, brake_seconds=0.5, restart_seconds=0.3):
    """Monotone speed ramps with zero speed and acceleration at the stop.

    A bounded quartic speed correction makes each ramp cover an integer number
    of source frames. Thus only the ramps need interpolation; the rest of the
    recorded path stays on its original frame grid.
    """
    if not np.isfinite([start, stop, peak, fps, brake_seconds, restart_seconds]).all():
        raise ValueError("nonfinite timing parameter")
    if fps <= 0 or min(brake_seconds, restart_seconds) <= 0:
        raise ValueError("fps and ramp durations must be positive")
    if any(x != int(x) for x in (start, stop, peak)):
        raise ValueError("source stage boundaries must be integer frames")
    brake, restart = round(fps * brake_seconds), round(fps * restart_seconds)
    if min(brake, restart) < 2:
        raise ValueError("ramps require at least two frame intervals")
    distance_down, distance_up = round(brake / 2), round(restart / 2)
    begin, end = peak - distance_down, peak + distance_up
    if not start <= begin < peak < end <= stop:
        raise ValueError("insufficient source trajectory around lift peak")

    lead = int(begin - start)
    x = np.r_[
        np.arange(start, begin),
        begin + speed_ramp(brake, distance_down, False),
        peak + speed_ramp(restart, distance_up, True)[1:],
        np.arange(end + 1, stop + 1),
    ]
    return (
        x,
        lead + brake,
        dict(
            brake_seconds=brake / fps,
            restart_seconds=restart / fps,
            brake_source_start=begin,
            restart_source_end=end,
            brake_start_index=lead,
            stop_index=lead + brake,
            restart_end_index=lead + brake + restart,
            timing_method="integrated_monotone_speed_with_integer_source_joins",
        ),
    )


def select_lift_peak(tcp, eligible, pickup, gate, *, height_band_m=0.002):
    """Prefer low 3-D speed within 2 mm of the safe pre-insertion height peak."""
    tcp, eligible = np.asarray(tcp, float), np.asarray(eligible, bool)
    if tcp.ndim != 2 or tcp.shape[1] != 3 or eligible.shape != (len(tcp),):
        raise ValueError("invalid TCP trajectory or eligibility mask")
    ids = np.arange(len(tcp))
    candidates = ids[
        eligible & (ids >= pickup) & (ids <= gate) & np.isfinite(tcp).all(axis=1)
    ]
    if not len(candidates):
        raise ValueError("no safe held lift peak before drawer entry")
    highest = float(tcp[candidates, 2].max())
    near = candidates[tcp[candidates, 2] >= highest - height_band_m]
    speed = np.linalg.norm(np.gradient(tcp, axis=0), axis=1)
    peak = int(near[np.argmin(speed[near])])
    return peak, dict(
        peak_source_frame=peak,
        tcp_height_m=float(tcp[peak, 2]),
        maximum_safe_height_m=highest,
        height_band_m=height_band_m,
    )


def smooth_wait_boundaries(
    left, right, fps, *, brake_seconds=0.5, restart_seconds=0.3, stop_indices=None
):
    """Retime a paired path at every change between moving and waiting.

    Both clocks share one path parameter: this does not independently drag an
    arm into a different section of the other arm's path. A moving partner also
    eases at the boundary, then continues while the waiting arm stays still.
    Short adjacent segments lower their peak rate instead of overlapping ramps.
    Callers must audit interpolated configurations before consuming the result.
    """
    points = np.column_stack([left, right]).astype(float)
    if (
        len(points) < 1
        or not np.isfinite(points).all()
        or np.any(np.diff(points, axis=0) < 0)
    ):
        raise ValueError("expected finite monotone paired source clocks")
    down, up = round(fps * brake_seconds), round(fps * restart_seconds)
    if fps <= 0 or min(down, up) < 2:
        raise ValueError("invalid smooth waiting duration")
    moving = np.diff(points, axis=0) > 0
    if stop_indices is None:
        corners = np.flatnonzero(np.any(moving[:-1] != moving[1:], axis=1)) + 1
    else:
        requested = np.asarray(stop_indices, float)
        if (
            requested.ndim != 1
            or not np.isfinite(requested).all()
            or np.any(requested != np.floor(requested))
        ):
            raise ValueError("stop indices must be finite integer path vertices")
        if np.any(requested <= 0) or np.any(requested >= len(points) - 1):
            raise ValueError("stop indices must be interior path vertices")
        corners = np.unique(requested.astype(int))
    if np.any(corners <= 0) or np.any(corners >= len(points) - 1):
        raise ValueError("stop indices must be interior path vertices")
    if not len(corners):
        return (
            points[:, 0],
            points[:, 1],
            dict(transitions=[], output_frames=len(points)),
        )
    bounds = np.r_[0, corners, len(points) - 1]
    path = [0.0]
    transitions = []
    for segment, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        accelerating = segment > 0
        braking = segment < len(bounds) - 2
        distance = float(b - a)
        nominal = (round(up / 2) if accelerating else 0) + (
            round(down / 2) if braking else 0
        )
        plateau = max(0, int(np.floor(distance - nominal)))
        rate = distance / (nominal + plateau)
        local = [0.0]
        if accelerating:
            local.extend((rate * speed_ramp(up, round(up / 2), True)[1:]).tolist())
        if plateau:
            local.extend((local[-1] + rate * np.arange(1, plateau + 1)).tolist())
        if braking:
            begin = len(path) - 1 + len(local) - 1
            local.extend(
                (
                    local[-1] + rate * speed_ramp(down, round(down / 2), False)[1:]
                ).tolist()
            )
            transitions.append(
                dict(
                    source_path_frame=int(b),
                    brake_start_output_frame=begin,
                    stop_output_frame=len(path) - 1 + len(local) - 1,
                    restart_end_output_frame=len(path) - 1 + len(local) - 1 + up,
                    waiting_arms=np.flatnonzero(~moving[b]).tolist(),
                )
            )
        local[-1] = distance
        path.extend((a + np.array(local[1:])).tolist())
    result = sample_rows(points, np.array(path))
    return (
        result[:, 0],
        result[:, 1],
        dict(
            transitions=transitions,
            output_frames=len(result),
            brake_seconds=down / fps,
            restart_seconds=up / fps,
            method="coordinated_source_path_wait_boundaries",
        ),
    )


def held_grasp_interval(state, action, event, fps):
    """A settled jaw closure followed by the first reopening, within visual pickup.

    Later post-release jaw closure cannot re-enter this interval. Visual
    attachment confirmation may lag the actual pickup and must not discard
    the recorded lift peak.
    """
    aperture = np.maximum(np.asarray(state)[:, 6], np.asarray(action)[:, 6])
    start, stop = event["pickup_frame"], event["release_frame"]
    window = max(3, round(fps * 0.1))
    opened = float(aperture[event["approach_start"] : stop + 1].max())
    settled = []
    floor = np.inf
    for frame in range(start, stop - window + 1):
        values = aperture[frame : frame + window]
        # Stop at the first sustained reopening. Do not cross an empty grasp
        # and quietly relabel a later grasp as the selected object's pickup.
        if np.isfinite(floor) and values.min() > floor + 2:
            break
        if np.ptp(values) <= 0.5 and values.max() < opened - 2:
            level = float(values.max())
            settled.append((frame, level))
            floor = min(floor, level)
    if not settled:
        raise ValueError("no settled recorded jaw closure after visual pickup")
    if floor <= 2:
        raise ValueError("settled closure is empty, not evidence of holding the target")
    limit = float(floor + 0.5)
    grasp = next(frame for frame, level in settled if level <= limit)
    reopening = [
        frame
        for frame in range(grasp + window, stop - window + 2)
        if np.min(aperture[frame : frame + window]) > limit
    ]
    end = reopening[0] if reopening else stop
    if end - grasp < window:
        raise ValueError("recorded grasp has no settled holding interval")
    return grasp, int(end), limit
