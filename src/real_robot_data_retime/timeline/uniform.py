"""Uniform relative onset sampling for two prerequisite stages."""

import numpy as np


def uniform_samples(source_index, source_count):
    if source_count < 1 or not 0 <= source_index < source_count:
        raise ValueError("invalid source episode index/count")
    # One point in each half. Unlike parity filtering this works for even N too.
    return [
        (source_index + source_count * variant) / (2 * source_count)
        for variant in range(2)
    ]


def stage_delays(a_frames, b_frames, position):
    if not np.isfinite([a_frames, b_frames, position]).all():
        raise ValueError("nonfinite stage timing")
    if (
        a_frames != int(a_frames)
        or b_frames != int(b_frames)
        or min(a_frames, b_frames) <= 0
        or not 0 <= position <= 1
    ):
        raise ValueError("invalid stage duration or uniform position")
    width = a_frames + b_frames
    requested = a_frames - position * width
    delta = int(np.floor(requested + 0.5))
    return dict(
        position=float(position),
        a_frames=int(a_frames),
        b_frames=int(b_frames),
        width_frames=int(width),
        requested_relative_start_frames=float(requested),
        relative_start_frames=delta,
        rounding_error_frames=float(delta - requested),
        left_delay_frames=max(0, -delta),
        right_delay_frames=max(0, delta),
    )


def validate_stage_schedule(left, right, stages):
    left, right = np.asarray(left), np.asarray(right)
    if (
        left.ndim != 1
        or left.shape != right.shape
        or len(left) < 2
        or not np.isfinite([left, right]).all()
        or np.any(np.diff(left) < 0)
        or np.any(np.diff(right) < 0)
    ):
        raise ValueError("invalid monotone paired source clocks")
    u = stages["uniform"]
    # The first nonzero clock interval starts at the preceding output frame.
    starts = [int(np.flatnonzero(np.diff(x) > 1e-9)[0]) for x in (left, right)]
    if starts[1] - starts[0] != u["relative_start_frames"]:
        raise ValueError("scheduler changed the sampled relative onset")
    peak, opened = stages["peak_source_frame"], stages["open_source_frame"]
    if np.any((left[1:] > peak) & (right[:-1] < opened)):
        raise ValueError("insertion interval starts before drawer opening")
    if np.any(
        (right[1:] >= stages["close_source_frame"])
        & (left[:-1] < stages["withdrawal_source_frame"])
    ):
        raise ValueError("drawer closing starts before withdrawal")
    arrival = int(np.flatnonzero(left == peak)[0])
    opening = int(np.flatnonzero(right == opened)[0])
    if arrival - starts[0] != u["a_frames"] or opening - starts[1] != u["b_frames"]:
        raise ValueError("scheduler changed a sampled prerequisite duration")
    return dict(
        left_start_frame=starts[0],
        right_start_frame=starts[1],
        lift_peak_frame=arrival,
        drawer_open_frame=opening,
        insertion_start_frame=int(np.flatnonzero(left > peak)[0]) - 1,
        passed=True,
    )
