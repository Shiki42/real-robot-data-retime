"""Time-aware drawer-body validation during lift braking and airborne waiting."""

from functools import lru_cache
import numpy as np
from .smooth import sample_rows
from .scheduler import NoSafeSchedule
from ..collision.drawer import drawer_body, DrawerGeometry


def audit_braking(
    checker,
    state,
    action,
    tcp,
    volume,
    motion,
    event,
    left,
    right,
    start,
    stop,
    *,
    geometry=None,
):
    geometry = DrawerGeometry() if geometry is None else geometry
    pull, opened = motion["pull_start"], motion["open_frame"]

    @lru_cache(maxsize=8192)
    def pose(kind, source):
        values = state if kind == 0 else action
        return checker._pose(sample_rows(values[:, :7], np.array([source]))[0], 0)

    @lru_cache(maxsize=8192)
    def handle_pose(source):
        return checker._pose(sample_rows(state[:, 7:], np.array([source]))[0], 1)[4]

    checked = 0
    for output in range(start, stop):
        # drawer_body is a SOLID exclusion proxy for a closed/pulling drawer.
        # Once open, its interior is free space; the original joint planner
        # likewise stops applying this solid exclusion at the opening gate.
        if right[output] >= opened:
            continue
        deltas = [
            np.abs(
                np.diff(sample_rows(values[:, :7], left[output : output + 2]), axis=0)[
                    0
                ]
            )
            for values in (state, action)
        ]
        left_bound = max(
            delta[:6].sum() * np.pi / 180 * checker.max_reach_m + delta[6] / 2000
            for delta in deltas
        )
        right_times = np.clip(right[output : output + 2], pull, opened)
        delta_right = np.abs(
            sample_rows(state[:, 7:], right_times)[1]
            - sample_rows(state[:, 7:], right_times)[0]
        )[:6].sum()
        bound = left_bound + delta_right * np.pi / 180 * checker.max_reach_m
        steps = max(1, int(np.ceil(bound / (checker.margin * 0.5))))
        slack = bound / (2 * steps)
        for fraction in np.linspace(0, 1, steps + 1):
            right_source = right[output] + fraction * (
                right[output + 1] - right[output]
            )
            if right_source >= opened:
                continue
            left_source = float(
                left[output] + fraction * (left[output + 1] - left[output])
            )
            handle_source = float(
                np.clip(
                    right[output] + fraction * (right[output + 1] - right[output]),
                    pull,
                    opened,
                )
            )
            handle = handle_pose(handle_source)
            body = drawer_body(handle, volume[0], **geometry.box_kwargs)
            for kind in (0, 1):
                current = pose(kind, left_source)
                if not checker.pose_clears_volume(
                    current, body, checker.margin + slack
                ):
                    raise NoSafeSchedule(
                        f"braking arm intersects instantaneous drawer body at output {output}"
                    )
                checked += 1
    return dict(
        method="closed_or_pulling_drawer_body_during_brake_and_wait",
        checked_poses=checked,
        brake_start_output_frame=int(start),
        restart_start_output_frame=int(stop),
        passed=True,
    )
