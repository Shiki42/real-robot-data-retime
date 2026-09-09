"""Build source maps from automatically inferred events and recorded joints."""

from functools import lru_cache
from pathlib import Path
import numpy as np
from .scheduler import schedule_sources
from .holds import append_terminal_hold, compress_static_spans
from ..retime import detect_arm_segments
from ..collision.piperx import PiperXClearance
from ..collision.drawer import drawer_sweep, outside_box, drawer_body
from ..tasks.drawer_constraints import precedence_gate


def plan_joints(
    state,
    action,
    timeline,
    urdf,
    mesh_root,
    *,
    terminal_seconds=2,
):
    state, action = np.asarray(state, float), np.asarray(action, float)
    if (
        state.shape != (timeline["source_frames"], 14)
        or action.shape != state.shape
        or not np.isfinite([state, action]).all()
    ):
        raise ValueError("joint rows must match every source video frame")
    task = timeline["task"]
    if task == "drawer":
        sources = [np.arange(len(state)), np.arange(len(state))]
    else:
        detected = detect_arm_segments(state, action)
        sources = [np.arange(s.start, s.end) for s in [detected.left, detected.right]]
    margin = 0.005 if task == "drawer" else 0.02
    receipt = dict(
        task=task,
        margin_m=margin,
        base_spacing_m=0.49,
        constraint_model="RoboVisualize cross-arm meshes",
        source_frames=len(state),
        dependencies={},
    )
    if task == "drawer":
        (event,) = timeline["episodes"]
        motion = timeline["drawer_motion"]
        if event["robot_id"] != "left" or motion is None:
            raise ValueError(
                "drawer requires verified left cube and right drawer events"
            )
        opening, closing = motion["open_frame"], motion["close_start"]
        guard = round(timeline["fps"] * 0.5)
        left = compress_static_spans(
            state[:, :7],
            np.asarray(action)[:, :7],
            [
                (0, max(0, event["grasp_start"] - guard)),
                (min(len(state), event["release_frame"] + guard), len(state)),
            ],
            timeline["fps"],
        )
        sources[0] = left
        sources[1] = compress_static_spans(
            state[:, 7:],
            action[:, 7:],
            [
                (0, max(0, motion["pull_start"] - guard)),
                (min(len(state), closing + guard), len(state)),
            ],
            timeline["fps"],
        )
        held_clock = compress_static_spans(
            state[:, 7:],
            action[:, 7:],
            [(opening, closing)],
            timeline["fps"],
            minimum_seconds=3 / timeline["fps"],
            guard_seconds=1 / timeline["fps"],
        )
        sources[1] = np.intersect1d(sources[1], held_clock, assume_unique=True)
        receipt["compressed_left_idle_frames"] = len(state) - len(sources[0])
        receipt["compressed_right_idle_frames"] = len(state) - len(sources[1])
        receipt["idle_pose_range_limits"] = dict(
            joint_degrees=0.3,
            gripper_mm=0.5,
            minimum_seconds=0.5,
            known_drawer_hold_minimum_frames=3,
            known_drawer_hold_guard_frames=1,
        )
    checker = PiperXClearance(
        state[:, :7],
        state[:, 7:],
        Path(urdf),
        Path(mesh_root),
        margin_m=margin,
    )
    dependency = lambda i, j: True
    if task == "drawer":
        # Derive handle motion in the same base frame used for mesh clearance.
        tcp = [np.array([pose[4] for pose in poses]) for poses in checker.poses]
        volume = drawer_sweep(tcp[1], motion["pull_start"], opening)
        clear = np.array(
            [
                outside_box(tcp[0][t], volume)
                and checker.arm_clears_volume(0, int(t), volume, margin=margin)
                for i, t in enumerate(sources[0])
            ]
        )
        waits = sources[0][
            clear
            & (sources[0] >= event["pickup_frame"])
            & (sources[0] < event["release_frame"])
        ]
        withdrawals = sources[0][clear & (sources[0] > event["release_frame"])]
        if not len(waits) or not len(withdrawals):
            raise ValueError(
                "no recorded held waiting pose or withdrawal clears drawer sweep"
            )
        wait, withdrawal = int(waits[-1]), int(withdrawals[0])

        @lru_cache(None)
        def dependency(i, j):
            left, right = int(sources[0][i]), int(sources[1][j])
            return precedence_gate(
                left,
                right,
                safe_approach_end=wait,
                open_frame=opening,
                withdrawal_frame=withdrawal,
                close_start=closing,
            )

        receipt["dependencies"] = dict(
            open_frame=opening,
            close_start=closing,
            safe_wait_frame=wait,
            wait_method="held_pose_clear_of_future_drawer_sweep",
            moving_clearance="instantaneous_drawer_body_with_motion_bounds",
            withdrawal_frame=withdrawal,
            swept_volume=[x.tolist() for x in volume],
            held_cube_radius_m=0.035,
            scene_clearance_m=margin,
        )

    def safe(i, j, ni, nj):
        li, ri, lni, rni = (
            int(sources[0][i]),
            int(sources[1][j]),
            int(sources[0][ni]),
            int(sources[1][nj]),
        )
        if not checker(li, ri, lni, rni):
            return False
        if task != "drawer" or ri >= opening:
            return True
        if (
            ni == i
            and event["pickup_frame"] <= li <= event["release_frame"]
            and not clear[i]
        ):
            return False
        differences = [state[lni, :7] - state[li, :7], state[rni, 7:] - state[ri, 7:]]
        bounds = [
            np.abs(d[:6]).sum() * np.pi / 180 * checker.max_reach_m + abs(d[6]) / 2000
            for d in differences
        ]
        motion_bound = bounds[0] + (bounds[1] if rni > motion["pull_start"] else 0.0)
        steps = max(1, int(np.ceil(motion_bound / (margin * 0.5))))
        slack = motion_bound / (2 * steps)
        for k in range(steps + 1):
            fraction = k / steps
            left_pose = checker._interpolated_pose(0, li, lni, k, steps)
            if ri + fraction * (rni - ri) <= motion["pull_start"]:
                handle = tcp[1][motion["pull_start"]]
            else:
                handle = checker._interpolated_pose(1, ri, rni, k, steps)[4]
            body = drawer_body(handle, volume[0])
            if not checker.pose_clears_volume(left_pose, body, margin + slack):
                return False
            if (
                event["pickup_frame"]
                <= li + fraction * (lni - li)
                <= event["release_frame"]
            ):
                if not outside_box(left_pose[4], body, radius=0.035 + slack):
                    return False
        return True

    phase_priority = lambda i, j: 0.0
    if task == "drawer":
        pickup_index = int(np.searchsorted(sources[0], event["pickup_frame"]))
        pull_index = int(np.searchsorted(sources[1], motion["pull_start"]))
        open_index = int(np.searchsorted(sources[1], opening))
        wait_index = int(np.searchsorted(sources[0], wait))
        target = (pull_index + open_index) / 2

        def phase_priority(i, j):
            return (
                abs((i - pickup_index) - (j - target))
                if i <= wait_index and j <= open_index
                else 0.0
            )

        receipt["alignment"] = (
            "pickup during drawer pull, subordinate to minimum duration"
        )
    schedule = schedule_sources(
        len(sources[0]),
        len(sources[1]),
        safe,
        dependency=dependency,
        left_priority=task != "drawer",
        tie_break=phase_priority,
    )
    edges = zip(
        schedule.left[:-1], schedule.right[:-1], schedule.left[1:], schedule.right[1:]
    )
    if not all(
        safe(int(i), int(j), int(ni), int(nj)) and dependency(int(ni), int(nj))
        for i, j, ni, nj in edges
    ):
        raise RuntimeError("final schedule failed swept collision or dependency audit")
    left, right, hold = append_terminal_hold(
        sources[0][schedule.left],
        sources[1][schedule.right],
        timeline["fps"],
        terminal_seconds,
    )
    receipt.update(
        output_frames=len(left),
        left_wait_frames=schedule.left_waits,
        right_wait_frames=schedule.right_waits,
        swept_edges_verified=True,
        synthetic_terminal_hold=hold,
        optimality="minimum frames under retained source poses and declared constraints",
    )
    return left, right, receipt
