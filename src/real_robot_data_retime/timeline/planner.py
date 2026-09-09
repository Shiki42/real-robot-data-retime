"""Build source maps from automatically inferred events and recorded joints."""

from functools import lru_cache
from pathlib import Path
import numpy as np
from .scheduler import schedule_sources
from .holds import compress_wait, append_terminal_hold
from ..retime import detect_motion_interval, detect_arm_segments
from ..collision.piperx import PiperXClearance
from ..collision.drawer import drawer_sweep, outside_box
from ..tasks.drawer_constraints import discover_insertion_gate, precedence_gate


def plan_joints(
    state,
    action,
    timeline,
    frames,
    object_tracks,
    urdf,
    mesh_root,
    *,
    terminal_seconds=2,
):
    state = np.asarray(state, float)
    if state.shape != (timeline["source_frames"], 14) or not np.isfinite(state).all():
        raise ValueError("joint rows must match every source video frame")
    task = timeline["task"]
    if task == "drawer":
        segments = [detect_motion_interval(state, side) for side in ["left", "right"]]
    else:
        detected = detect_arm_segments(state, action)
        segments = [detected.left, detected.right]
    sources = [
        np.arange(s.start, len(state) if task == "drawer" else s.end) for s in segments
    ]
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
        right = compress_wait(state[:, 7:], opening, closing)
        sources[1] = right[right >= segments[1].start]
    checker = PiperXClearance(
        state[sources[0], :7],
        state[sources[1], 7:],
        Path(urdf),
        Path(mesh_root),
        margin_m=margin,
    )
    dependency = lambda i, j: True
    if task == "drawer":
        # Derive handle motion in the same base frame used for mesh clearance.
        tcp = []
        for side in [0, 1]:
            positions = []
            for row in state:
                checker._pose(row[side * 7 : side * 7 + 7], side)
                positions.append(checker.model.tcp_transform().translation.copy())
            tcp.append(np.asarray(positions))
        volume = drawer_sweep(tcp[1], motion["pull_start"], opening)
        image_gate, _ = discover_insertion_gate(
            frames[opening],
            object_tracks[event["object_id"]],
            event["pickup_frame"],
            event["release_frame"],
        )
        clear = np.array(
            [
                outside_box(tcp[0][t], volume)
                and checker.arm_clears_volume(0, i, volume)
                for i, t in enumerate(sources[0])
            ]
        )
        waits = sources[0][
            clear & (sources[0] >= event["pickup_frame"]) & (sources[0] <= image_gate)
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
            ) and (right >= opening or left < event["pickup_frame"] or bool(clear[i]))

        receipt["dependencies"] = dict(
            open_frame=opening,
            close_start=closing,
            safe_wait_frame=wait,
            image_entry_gate=image_gate,
            withdrawal_frame=withdrawal,
            swept_volume=[x.tolist() for x in volume],
            held_cube_radius_m=0.035,
        )

    def safe(i, j, ni, nj):
        if not checker(i, j, ni, nj):
            return False
        if task != "drawer":
            return True
        li, ri = int(sources[0][i]), int(sources[1][j])
        differences = [
            checker.values[side][stop] - checker.values[side][start]
            for side, start, stop in [(0, i, ni), (1, j, nj)]
        ]
        bounds = [
            np.abs(d[:6]).sum() * np.pi / 180 * checker.max_reach_m + abs(d[6]) / 2000
            for d in differences
        ]
        if event["pickup_frame"] <= li <= event["release_frame"]:
            if ri < opening:
                if not outside_box(tcp[0][li], volume, radius=0.035 + bounds[0]):
                    return False
                if not checker.arm_clears_volume(0, i, volume, margin=0.02 + bounds[0]):
                    return False
        return True

    schedule = schedule_sources(
        len(sources[0]),
        len(sources[1]),
        safe,
        dependency=dependency,
        left_priority=task != "drawer",
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
