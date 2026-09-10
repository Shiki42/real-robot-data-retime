"""Conservative image-space scheduling when metric joints are unavailable."""

import numpy as np
from functools import lru_cache
from .scheduler import schedule_sources
from .smooth import (
    lift_clock,
    select_lift_peak,
    sample_rows,
    smooth_wait_boundaries,
    held_grasp_interval,
)
from ..background.clean_plate import dilate
from ..compositing.ownership import arm_foreground
from ..tasks.drawer_constraints import (
    discover_insertion_gate,
    precedence_gate,
    discover_drawer_interior,
)


def plan_visual(
    timeline,
    frames,
    tracks,
    segmentation,
    *,
    joints=None,
    right_delay_seconds=0,
    left_delay_seconds=0,
    uniform_position=None,
):
    if uniform_position is not None and (
        timeline["task"] != "drawer" or joints is None
    ):
        raise ValueError("uniform staging requires drawer joint data")
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

    workpiece = timeline["task"] == "workpiece"
    milestones = []
    stages = {}
    if workpiece:
        from ..tasks.workpiece import alternating_pickups, pickup_precedence
        from .workpiece import (
            workpiece_sources,
            staging_candidates,
            admission_allowed,
            verify_uninterrupted,
            prepend_right_preparation,
        )

        milestones = alternating_pickups(events)
        candidates = staging_candidates(events, robots, objects, timeline["fps"])

        def dependency(i, j):
            return pickup_precedence(
                sources[0][i], sources[1][j], milestones
            ) and admission_allowed(sources[0][i], sources[1][j], stages)

    drawer = timeline["task"] == "drawer"
    stop_index = None
    if (
        not np.isfinite([right_delay_seconds, left_delay_seconds]).all()
        or min(right_delay_seconds, left_delay_seconds) < 0
    ):
        raise ValueError("onset delays must be finite and nonnegative")
    if drawer:
        (event,) = events
        motion = timeline["drawer_motion"]
        a, b = motion["open_frame"], motion["close_start"]
        if uniform_position is None:
            gate, interior = discover_insertion_gate(
                frames[a],
                tracks["objects"][event["object_id"]],
                event["pickup_frame"],
                event["release_frame"],
            )
        else:
            interior = discover_drawer_interior(frames[a])
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

        if joints is not None and drawer:
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
            if uniform_position is not None:
                grasp, held_end, aperture = held_grasp_interval(
                    state, action, event, timeline["fps"]
                )
                candidate_start, candidate_end = grasp, held_end - 1
            else:
                candidate_start, candidate_end = event["pickup_frame"], gate
            eligible = np.zeros(n, bool)
            for t in range(candidate_start, candidate_end + 1):
                eligible[t] = (
                    max(state[t, 6], action[t, 6]) <= aperture
                    and outside_box(tcp[0, t], volume)
                    and checker.arm_clears_volume(0, t, volume, margin=0.005)
                )
            gate, stages = select_lift_peak(tcp[0], eligible, grasp, candidate_end)
            if uniform_position is not None:
                stages["recorded_grasp_frame"] = grasp
                stages["recorded_reopening_frame"] = held_end
            stages["holding_aperture_limit_mm"] = float(aperture)
            stages["wait_geometry"] = (
                "arm_mesh_and_35mm_held_object_clear_of_drawer_sweep"
            )
            if uniform_position is not None:
                from .uniform import stage_delays

                if left_delay_seconds or right_delay_seconds:
                    raise ValueError(
                        "uniform position and explicit delays are mutually exclusive"
                    )
                _, peak_index, _ = lift_clock(
                    sources[0][0], sources[0][-1], gate, timeline["fps"]
                )
                uniform = stage_delays(peak_index, a, uniform_position)
                left_delay_seconds = uniform["left_delay_frames"] / timeline["fps"]
                right_delay_seconds = uniform["right_delay_frames"] / timeline["fps"]
                stages["uniform"] = uniform
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
        if drawer:
            if sources[0][ni] > gate and sources[1][j] < a:
                return False
            if sources[1][nj] >= b and sources[0][i] < withdrawal:
                return False
        return clear(i, j) and clear(ni, nj) and clear(i, nj) and clear(ni, j)

    left_delay = (
        round(left_delay_seconds * timeline["fps"]) if joints is not None else 0
    )
    if joints is not None and drawer and left_delay:
        sources[0] = np.r_[np.full(left_delay, sources[0][0]), sources[0]]
        stop_index += left_delay

    if not workpiece and not (joints is not None and drawer):
        for side, seconds in enumerate((left_delay_seconds, right_delay_seconds)):
            count = round(seconds * timeline["fps"])
            sources[side] = np.r_[np.full(count, sources[side][0]), sources[side]]

    def search():
        return schedule_sources(
            len(sources[0]),
            len(sources[1]),
            safe,
            dependency=dependency,
            left_priority=not (drawer or workpiece),
            can_wait=lambda side, i: (
                i in workpiece_holds[side]
                if workpiece
                else joints is None
                or not drawer
                or (side == 1 and (sources[1][i] == a or i == len(sources[1]) - 1))
                or (side == 0 and i in (stop_index, len(sources[0]) - 1))
            ),
        )

    if workpiece:
        from heapq import heappop, heappush
        from .scheduler import NoSafeSchedule

        frontier = [(0, (0, 0, 0))]
        visited = {(0, 0, 0)}
        attempts = 0
        while frontier:
            _, selection = heappop(frontier)
            stops = [values[index] for values, index in zip(candidates, selection)]
            sources, workpiece_holds, stages = workpiece_sources(
                events, timeline["fps"], stops
            )
            for side, seconds in enumerate((left_delay_seconds, right_delay_seconds)):
                count = round(seconds * timeline["fps"])
                stages["onset_delay_frames"][side] = count
                sources[side] = np.r_[np.full(count, sources[side][0]), sources[side]]
                workpiece_holds[side] = {
                    index + count for index in workpiece_holds[side]
                }
            mask.cache_clear()
            clear.cache_clear()
            attempts += 1
            try:
                schedule = search()
                stages["staging_search_attempts"] = attempts
                break
            except NoSafeSchedule:
                for axis in range(3):
                    neighbor = list(selection)
                    neighbor[axis] += 1
                    neighbor = tuple(neighbor)
                    if (
                        neighbor[axis] < len(candidates[axis])
                        and neighbor not in visited
                    ):
                        visited.add(neighbor)
                        distance = sum(
                            values[0] - values[index]
                            for values, index in zip(candidates, neighbor)
                        )
                        heappush(frontier, (distance, neighbor))
        else:
            raise NoSafeSchedule(
                "no uninterrupted workpiece execution fits the observed staging poses"
            )
    else:
        schedule = search() if uniform_position is None else None
    if joints is not None and drawer:
        smooth_stop = uniform_position is not None or bool(
            np.any((np.diff(schedule.left) == 0) & (schedule.left[:-1] == stop_index))
        )
        stages["smooth_stop_required"] = smooth_stop
        if smooth_stop:
            clock, stop_index, ramps = lift_clock(
                sources[0][0], sources[0][-1], gate, timeline["fps"]
            )
            begin = int(np.floor(ramps["brake_source_start"]))
            if begin < candidate_start or not eligible[begin : gate + 1].all():
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
                end
                >= (
                    held_end if uniform_position is not None else event["release_frame"]
                )
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
    left, right = sources[0][schedule.left], sources[1][schedule.right]
    smoothing = dict(transitions=[])
    if not (joints is not None and drawer):
        if workpiece:
            left, right = prepend_right_preparation(
                left, right, stages, timeline["fps"]
            )
            verify_uninterrupted(left, right, stages)
            smoothing = dict(
                method="independent_approach_clocks",
                transitions=[
                    dict(arm=side, **ramp)
                    for side, ramps in stages["ramps"].items()
                    for ramp in ramps
                ],
            )
        else:
            left, right, smoothing = smooth_wait_boundaries(
                left, right, timeline["fps"]
            )

        @lru_cache(maxsize=4096)
        def support(side, start, end):
            result = np.zeros((h, w), bool)
            for t in range(int(np.floor(start)), int(np.ceil(end)) + 1):
                result |= arm_foreground(robots, objects, events, side, t)
            return dilate(result, 2)

        if smoothing["transitions"]:
            for left_frame, right_frame, nl, nr in zip(
                left[:-1], right[:-1], left[1:], right[1:]
            ):
                if drawer and (
                    (nl > gate and right_frame < a)
                    or (nr >= b and left_frame < withdrawal)
                ):
                    raise ValueError("smoothed path violates task precedence")
                if workpiece and not pickup_precedence(nl, nr, milestones):
                    raise ValueError("smoothed path violates alternating pickup order")
                cuts = {0.0, 1.0}
                for start, end in [(left_frame, nl), (right_frame, nr)]:
                    if end > start:
                        cuts.update(
                            (t - start) / (end - start)
                            for t in range(int(np.floor(start)) + 1, int(np.ceil(end)))
                            if start < t < end
                        )
                cuts = sorted(cuts)
                for u, v in zip(cuts[:-1], cuts[1:]):
                    la, lb = np.round(
                        [
                            left_frame + u * (nl - left_frame),
                            left_frame + v * (nl - left_frame),
                        ],
                        10,
                    )
                    ra, rb = np.round(
                        [
                            right_frame + u * (nr - right_frame),
                            right_frame + v * (nr - right_frame),
                        ],
                        10,
                    )
                    if np.any(support(0, la, lb) & support(1, ra, rb)):
                        raise ValueError(
                            f"smoothed wait transition fails silhouette clearance at {la},{ra} -> {lb},{rb}"
                        )
        if joints is not None:
            from ..collision.piperx import PiperXClearance
            from pathlib import Path

            for values in joints[:2]:
                checker = PiperXClearance(
                    sample_rows(values[:, :7], left),
                    sample_rows(values[:, 7:], right),
                    Path(joints[2]),
                    Path(joints[3]),
                )
                if not all(checker(i, i, i + 1, i + 1) for i in range(len(left) - 1)):
                    raise ValueError(
                        "smoothed wait transition fails joint swept clearance"
                    )
    return (
        left,
        right,
        dict(
            collision_scope=(
                "projected silhouettes plus metric wait/brake clearance"
                if joints is not None and drawer
                else "projected silhouettes plus state/action swept meshes"
                if joints is not None
                else "projected silhouettes only"
            ),
            left_wait_frames=int(np.sum(np.diff(left) == 0)),
            right_wait_frames=int(np.sum(np.diff(right) == 0)),
            output_frames=len(left),
            smoothing=smoothing,
            stages=stages,
            pickup_order=[
                dict(
                    arm=("left", "right")[side],
                    source_frame=frame,
                    output_frame=int(np.flatnonzero((left, right)[side] >= frame)[0]),
                )
                for side, frame in milestones
            ],
        ),
    )
