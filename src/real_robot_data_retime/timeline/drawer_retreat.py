"""Permit closing when a released left arm starts a sustained retreat."""

import numpy as np
from .smooth import sample_rows


def retreat_start(tcp, event, full_exit, fps):
    """First sustained upward/outward motion after release, or observed full exit."""
    release = event["release_frame"]
    width = max(2, round(fps * 0.1))
    home = tcp[event["approach_start"], :2] - tcp[release, :2]
    length = np.linalg.norm(home)
    direction = home / length if length > 1e-9 else np.zeros(2)
    for frame in range(release, min(full_exit, len(tcp) - width)):
        steps = np.diff(tcp[frame : frame + width + 1], axis=0)
        upward = steps[:, 2]
        outward = steps[:, :2] @ direction
        if any(
            values[0] > 0
            and values.sum() >= 0.002
            and np.count_nonzero(values > 0) >= width - 1
            for values in (upward, outward)
        ):
            return frame
    return full_exit


def early_closing_clear(checker, state, action, volume, left, right):
    """Conservative continuous arm/arm and arm/drawer sweep checks on one edge."""
    bounds = []
    for values in (state, action):
        bound = 0.0
        for side, times in enumerate((left, right)):
            knots = np.unique(
                np.r_[
                    times[0],
                    np.arange(np.ceil(times[0]), np.floor(times[1]) + 1),
                    times[1],
                ]
            )
            travel = np.abs(
                np.diff(
                    sample_rows(values[:, side * 7 : (side + 1) * 7], knots), axis=0
                )
            ).sum(axis=0)
            bound += (
                travel[:6].sum() * np.pi / 180 * checker.max_reach_m + travel[6] / 2000
            )
        bounds.append(bound)
    bound = max(bounds)
    steps = max(1, int(np.ceil(bound / (checker.margin * 0.5))))
    slack = bound / (2 * steps)
    for fraction in np.linspace(0, 1, steps + 1):
        clocks = [
            times[0] + fraction * (times[1] - times[0]) for times in (left, right)
        ]
        for values in (state, action):
            poses = [
                checker._pose(
                    sample_rows(
                        values[:, side * 7 : (side + 1) * 7], np.array([clock])
                    )[0],
                    side,
                )
                for side, clock in enumerate(clocks)
            ]
            if not checker.pose_clears_volume(poses[0], volume, checker.margin + slack):
                return False
            clear = checker._clear(*poses, extra_margin=slack)
            if not clear:
                return False
    return True
