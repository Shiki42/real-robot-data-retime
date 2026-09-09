"""Build source maps from automatically inferred events and recorded joints."""

from functools import lru_cache
from pathlib import Path
import numpy as np
from .scheduler import schedule_sources, NoSafeSchedule
from .bounds import drawer_remaining_bound
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
        confirmed_release = event.get(
            "release_confirmation_frame", event["release_frame"]
        )
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
    required_contacts = []
    dependency = lambda i, j: True
    if task == "drawer":
        # Derive handle motion in the same base frame used for mesh clearance.
        tcp = [np.array([pose[4] for pose in poses]) for poses in checker.poses]
        volume = drawer_sweep(tcp[1], motion["pull_start"], opening)
        arm_clear = np.array(
            [
                checker.arm_clears_volume(0, t, volume, margin=margin)
                for t in range(len(state))
            ]
        )
        clear = arm_clear & np.array([outside_box(point, volume) for point in tcp[0]])
        # A held wait must retain the visually verified grasp configuration;
        # a later open-jaw withdrawal is not a valid cube-holding pose.
        grasp = event.get("grasp_frame", event["pickup_frame"])
        holding_aperture = max(state[grasp, 6], action[grasp, 6]) + 0.5
        holding = np.maximum(state[:, 6], action[:, 6]) <= holding_aperture
        last_attached = event.get("last_attached_frame", event["release_frame"] - 1)
        opens_by_confirmation = (
            max(state[confirmed_release, 6], action[confirmed_release, 6])
            > holding_aperture
        )
        holding_limit = (
            event["release_frame"] - 1
            if opens_by_confirmation
            else (event["pickup_frame"] - 1 if last_attached is None else last_attached)
        )
        eligible = (
            clear
            & holding
            & (np.arange(len(state)) >= event["pickup_frame"])
            & (np.arange(len(state)) < event["release_frame"])
            & (np.arange(len(state)) <= holding_limit)
        )
        waits = sources[0][eligible[sources[0]]]
        if len(waits):
            # Stop at the first safe held interval, before insertion/retraction.
            first = int(waits[0])
            blocked_after = np.flatnonzero(~eligible[first:])
            stop = first + int(blocked_after[0]) if len(blocked_after) else len(state)
            wait = int(waits[waits < stop][-1])
            wait_method = "held_pose_clear_of_future_drawer_sweep"
        else:
            empty = sources[0][
                arm_clear[sources[0]] & (sources[0] < event["pickup_frame"])
            ]
            if not len(empty):
                raise ValueError("no recorded waiting pose clears drawer sweep")
            wait = int(empty[-1])
            wait_method = "empty_pre_pickup_pose_clear_of_future_drawer_sweep"
        withdrawals = sources[0][
            arm_clear[sources[0]] & (sources[0] > confirmed_release)
        ]
        if not len(withdrawals):
            raise ValueError(
                "no recorded withdrawal clears drawer sweep after confirmed release"
            )
        withdrawal = int(withdrawals[0])

        @lru_cache(None)
        def dependency(i, j):
            left, right = int(sources[0][i]), int(sources[1][j])
            for frame in required_contacts:
                if (
                    (left < frame and right > frame)
                    or (left > frame and right < frame)
                    or (left == frame and right != frame)
                ):
                    return False
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
            wait_method=wait_method,
            release_confirmation_frame=confirmed_release,
            held_wait_maximum_aperture_mm=float(holding_aperture),
            holding_bound="closed_grasp_configuration"
            if opens_by_confirmation
            else "last_observed_attachment",
            moving_clearance="instantaneous_drawer_body_with_motion_bounds",
            withdrawal_frame=withdrawal,
            withdrawal_geometry="empty_gripper_and_arm_meshes",
            swept_volume=[x.tolist() for x in volume],
            held_cube_radius_m=0.035,
            scene_clearance_m=margin,
        )

    preserved_edges = set()

    def safe(i, j, ni, nj):
        li, ri, lni, rni = (
            int(sources[0][i]),
            int(sources[1][j]),
            int(sources[0][ni]),
            int(sources[1][nj]),
        )
        if not checker(li, ri, lni, rni) and not original_pair_edge(
            li, ri, lni, rni, preserved_edges
        ):
            return False
        if task != "drawer" or ri >= opening:
            return True
        if (
            ni == i
            and event["pickup_frame"] <= li <= event["release_frame"]
            and not clear[li]
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
            "pre-pickup wait until drawer opening"
            if wait_method == "empty_pre_pickup_pose_clear_of_future_drawer_sweep"
            else "pickup during drawer pull, subordinate to minimum duration"
        )

    def search():
        n, m = len(sources[0]), len(sources[1])
        if task == "drawer":
            gate_index = int(np.searchsorted(sources[0], wait, side="right"))
            withdrawal_index = int(np.searchsorted(sources[0], withdrawal))
            opening_index = int(np.searchsorted(sources[1], opening))
            closing_index = int(np.searchsorted(sources[1], closing))
            markers = [
                (
                    int(np.searchsorted(sources[0], t)),
                    int(np.searchsorted(sources[1], t)),
                )
                for t in required_contacts
            ]

            remaining = drawer_remaining_bound(
                n,
                m,
                gate_index,
                opening_index,
                withdrawal_index,
                closing_index,
                markers,
            )
        else:
            remaining = None
        return schedule_sources(
            n,
            m,
            safe,
            dependency=dependency,
            left_priority=task != "drawer",
            tie_break=phase_priority,
            remaining_lower_bound=remaining,
        )

    blocked = []
    if task == "drawer":
        # A required left pose with no admissible open-drawer partner proves
        # strict retiming impossible, without exhaustively searching the grid.
        for frame in sources[0][(sources[0] > wait) & (sources[0] < withdrawal)]:
            if not any(
                checker.configuration_safe(int(frame), j)
                for j in range(opening, closing)
            ):
                blocked.append(int(frame))
    try:
        if blocked:
            raise NoSafeSchedule(
                "required left poses have no collision-free open-drawer partner"
            )
        schedule = search()
    except NoSafeSchedule:
        if task != "drawer":
            raise
        # Cooperative drawer recordings can already contain modeled fingertip
        # contact. Only an exact replay of those paired source edges is admissible;
        # a new pose combination or an extended contact hold remains forbidden.
        preserved_edges = {
            t
            for t in range(
                max(opening, event["pickup_frame"]),
                min(closing, len(state) - 1),
            )
            if not checker(t, t, t + 1, t + 1)
        }
        if not preserved_edges:
            raise
        restored = np.unique(
            [
                t
                for edge in preserved_edges
                for t in range(max(opening, edge - 1), min(closing, edge + 3))
            ]
        )
        sources = [np.union1d(clock, restored) for clock in sources]
        required_contacts = blocked
        dependency.cache_clear()
        receipt["strict_infeasibility_witness_frames"] = blocked
        schedule = search()
    copied = []
    for k, (i, j, ni, nj) in enumerate(
        zip(
            schedule.left[:-1],
            schedule.right[:-1],
            schedule.left[1:],
            schedule.right[1:],
        )
    ):
        li, ri, lni, rni = (
            int(sources[0][i]),
            int(sources[1][j]),
            int(sources[0][ni]),
            int(sources[1][nj]),
        )
        if not checker(li, ri, lni, rni):
            if not original_pair_edge(li, ri, lni, rni, preserved_edges):
                raise RuntimeError("unrecorded collision entered the schedule")
            copied.append(dict(output_edge=k, source_edge=li))
    edges = zip(
        schedule.left[:-1], schedule.right[:-1], schedule.left[1:], schedule.right[1:]
    )
    if not all(
        safe(int(i), int(j), int(ni), int(nj)) and dependency(int(ni), int(nj))
        for i, j, ni, nj in edges
    ):
        raise RuntimeError("final schedule failed swept collision or dependency audit")
    if task == "drawer":
        receipt["compressed_left_idle_frames"] = len(state) - len(sources[0])
        receipt["compressed_right_idle_frames"] = len(state) - len(sources[1])
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
        swept_edges_verified=not copied,
        new_edges_collision_free=True,
        preserved_original_pair_edges=copied,
        collision_policy="strict_clearance"
        if not copied
        else "strict_new_edges_with_exact_original_contact_replay",
        synthetic_terminal_hold=hold,
        optimality="minimum frames under retained source poses and declared constraints",
    )
    return left, right, receipt


def original_pair_edge(left, right, next_left, next_right, preserved):
    """No time offsets, frame skipping, or prolonged contact are allowed here."""
    return (
        left == right
        and next_left == next_right
        and next_left == left + 1
        and left in preserved
    )
