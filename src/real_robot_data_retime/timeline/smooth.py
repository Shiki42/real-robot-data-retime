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

    def ramp(duration, distance, up):
        u = np.arange(duration + 1, dtype=float) / duration
        correction = 30 * (distance / duration - 0.5)
        if abs(correction) > 3 + 1e-10:
            raise ValueError("ramp cannot remain monotone at this frame rate")
        integral = u**3 - 0.5 * u**4 if up else u - u**3 + 0.5 * u**4
        integral += correction * (u**3 / 3 - u**4 / 2 + u**5 / 5)
        result = duration * integral
        result[0], result[-1] = 0, distance
        return result

    lead = int(begin - start)
    x = np.r_[
        np.arange(start, begin),
        begin + ramp(brake, distance_down, False),
        peak + ramp(restart, distance_up, True)[1:],
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
