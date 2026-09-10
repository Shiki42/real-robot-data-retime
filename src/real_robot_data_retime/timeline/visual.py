"""Conservative image-space scheduling when metric joints are unavailable."""

import numpy as np
from functools import lru_cache
from .scheduler import schedule_sources
from .smooth import lift_clock, select_lift_peak, sample_rows
from ..background.clean_plate import dilate
from ..compositing.ownership import arm_foreground
from ..tasks.drawer_constraints import discover_insertion_gate, precedence_gate


def plan_visual(
    timeline,
    frames,
    tracks,
    segmentation,
    *,
    joints=None,
    right_delay_seconds=0,
    left_delay_seconds=0,
):
    n, h, w = frames.shape[:3]
    robots = np.unpackbits(segmentation["robots"], axis=-1, count=w).astype(bool)
    objects = np.unpackbits(segmentation["objects"], axis=-1, count=w).astype(bool)
    events = timeline["episodes"]
    sources = []
    for side in ["left", "right"]:
        own = [e for e in events if e["robot_id"] == side]
        if own:
            sources.append(
                np.arange(
                    min(e["approach_start"] for e in own),
                    max(e["retract_end"] for e in own) + 1,
                )
            )
        else:
            sources.append(np.arange(n))

    def dependency(i, j):
        return True

    drawer = timeline["task"] == "drawer"
    stages = {}
    stop_index = None
    if (
        not np.isfinite([right_delay_seconds, left_delay_seconds]).all()
        or min(right_delay_seconds, left_delay_seconds) < 0
        or (joints is not None and not drawer)
        or (joints is None and (right_delay_seconds or left_delay_seconds))
    ):
        raise ValueError("staged joint timing requires drawer and nonnegative delay")
    if drawer:
        (event,) = events
        motion = timeline["drawer_motion"]
        a, b = motion["open_frame"], motion["close_start"]
        gate, interior = discover_insertion_gate(
            frames[a],
            tracks["objects"][event["object_id"]],
            event["pickup_frame"],
            event["release_frame"],
        )
        # The open drawer's source dwell becomes a held right pose. The actual
        # moving parts before and after this dwell are retained at source speed.
        sources[1] = np.r_[np.arange(a + 1), np.arange(b, n)]
        outside = []
        for t in range(event["release_frame"] + 1, n):
            mask = robots[t, 0]
            if not np.any(mask & interior):
                outside.append(t)
        if not outside:
            raise ValueError("left withdrawal from drawer is not visible")
        withdrawal = outside[0]
        sources[0] = np.arange(min(e["approach_start"] for e in events), n)

        if joints is not None:
            from ..collision.piperx import PiperXClearance
            from ..collision.drawer import drawer_sweep, outside_box
            from pathlib import Path

            state, action, urdf, mesh_root = joints
            state, action = np.asarray(state, float), np.asarray(action, float)
            if (
                state.shape != (n, 14)
                or action.shape != state.shape
                or not np.isfinite([state, action]).all()
            ):
                raise ValueError("joint rows must match source video")
            checker = PiperXClearance(
                state[:, :7], state[:, 7:], Path(urdf), Path(mesh_root), margin_m=0.005
            )
            tcp = np.array([[pose[4] for pose in arm] for arm in checker.poses])
            volume = drawer_sweep(tcp[1], motion["pull_start"], a)
            grasp = event["grasp_frame"]
            aperture = max(state[grasp, 6], action[grasp, 6]) + 0.5
            eligible = np.zeros(n, bool)
            for t in range(event["pickup_frame"], gate + 1):
                eligible[t] = (
                    max(state[t, 6], action[t, 6]) <= aperture
                    and outside_box(tcp[0, t], volume)
                    and checker.arm_clears_volume(0, t, volume, margin=0.005)
                )
            gate, stages = select_lift_peak(tcp[0], eligible, grasp, gate)
            stages["holding_aperture_limit_mm"] = float(aperture)
            stages["wait_geometry"] = (
                "arm_mesh_and_35mm_held_object_clear_of_drawer_sweep"
            )
            delay = round(right_delay_seconds * timeline["fps"])
            sources[1] = np.r_[np.zeros(delay, dtype=int), sources[1]]
            stop_index = gate - sources[0][0]

        def dependency(i, j):
            return precedence_gate(
                float(sources[0][i]),
                float(sources[1][j]),
                safe_approach_end=gate,
                open_frame=a,
                withdrawal_frame=withdrawal,
                close_start=b,
            )

    @lru_cache(maxsize=4096)
    def mask(side, index):
        source = sources[side][index]
        foreground = np.zeros((h, w), bool)
        for t in {int(np.floor(source)), int(np.ceil(source))}:
            foreground |= arm_foreground(robots, objects, events, side, t)
        return np.packbits(dilate(foreground, 2))

    @lru_cache(None)
    def clear(i, j):
        return not np.any(mask(0, i) & mask(1, j))

    def safe(i, j, ni, nj):
        # Sweep union of adjacent silhouettes; this is a projected-occlusion
        # constraint, never reported as a metric robot collision guarantee.
        if joints is not None:
            if sources[0][ni] > gate and sources[1][j] < a:
                return False
            if sources[1][nj] >= b and sources[0][i] < withdrawal:
                return False
        return clear(i, j) and clear(ni, nj) and clear(i, nj) and clear(ni, j)

    left_delay = (
        round(left_delay_seconds * timeline["fps"]) if joints is not None else 0
    )
    if joints is not None and left_delay:
        sources[0] = np.r_[np.full(left_delay, sources[0][0]), sources[0]]
        stop_index += left_delay

    def search():
        return schedule_sources(
            len(sources[0]),
            len(sources[1]),
            safe,
            dependency=dependency,
            left_priority=not drawer,
            can_wait=lambda side, i: (
                joints is None
                or (side == 1 and (sources[1][i] == a or i == len(sources[1]) - 1))
                or (side == 0 and i in (stop_index, len(sources[0]) - 1))
            ),
        )

    schedule = search()
    if joints is not None:
        waiting = np.flatnonzero(
            (np.diff(schedule.left) == 0) & (schedule.left[:-1] == stop_index)
        )
        stages["smooth_stop_required"] = bool(len(waiting))
        if len(waiting):
            clock, stop_index, ramps = lift_clock(
                sources[0][0], sources[0][-1], gate, timeline["fps"]
            )
            begin = int(np.floor(ramps["brake_source_start"]))
            if begin < event["pickup_frame"] or not eligible[begin : gate + 1].all():
                raise ValueError("0.5 second braking path leaves safe held interval")
            # Recheck the interpolated braking poses, not only recorded endpoints.
            samples = np.linspace(
                ramps["brake_source_start"],
                gate,
                max(2, int(np.ceil((gate - ramps["brake_source_start"]) * 4)) + 1),
            )
            for row in np.concatenate(
                [sample_rows(values[:, :7], samples) for values in (state, action)]
            ):
                pose = checker._pose(row, 0)
                if not (
                    outside_box(pose[4], volume)
                    and checker.pose_clears_volume(pose, volume, margin=0.005)
                ):
                    raise ValueError("interpolated braking pose enters drawer sweep")
            end = int(ramps["restart_source_end"])
            if (
                end >= event["release_frame"]
                or np.maximum(state[gate : end + 1, 6], action[gate : end + 1, 6]).max()
                > aperture
            ):
                raise ValueError("restart ramp must retain the held object")
            sources[0] = np.r_[np.full(left_delay, clock[0]), clock]
            stop_index += left_delay
            mask.cache_clear()
            clear.cache_clear()
            schedule = search()
            stages.update(ramps)
        stages["left_delay_seconds"] = left_delay_seconds
        stages["right_delay_seconds"] = right_delay_seconds
        stages["stop_output_frames"] = np.flatnonzero(
            sources[0][schedule.left] == gate
        ).tolist()
        if stages["smooth_stop_required"]:
            clock = sources[0][schedule.left]
            stages["brake_start_output_frame"] = int(
                np.flatnonzero(clock == stages["brake_source_start"])[-1]
            )
            stages["restart_start_output_frame"] = stages["stop_output_frames"][-1]
            stages["restart_end_output_frame"] = int(
                np.flatnonzero(clock == stages["restart_source_end"])[0]
            )
        stages["held_output_intervals"] = int(
            np.sum((np.diff(schedule.left) == 0) & (schedule.left[:-1] == stop_index))
        )
        stages["open_source_frame"] = a
        stages["withdrawal_source_frame"] = withdrawal
        stages["close_source_frame"] = b
    return (
        sources[0][schedule.left],
        sources[1][schedule.right],
        dict(
            collision_scope="projected silhouettes only; no joint-space verification",
            left_wait_frames=schedule.left_waits,
            right_wait_frames=schedule.right_waits,
            output_frames=len(schedule.left),
            stages=stages,
        ),
    )
