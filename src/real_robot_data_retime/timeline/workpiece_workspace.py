"""Fixed-volume waiting poses and earliest continuously safe right admissions."""

from pathlib import Path

import numpy as np

from ..collision.piperx import PiperXClearance
from ..tasks.workpiece import workpiece_events
from .workpiece import approach_clock, verify_uninterrupted

DEFAULT_WORKSPACE = {
    "minimum_m": [0.27, -0.12, -0.04],
    "maximum_m": [0.46, 0.12, 0.18],
    "ee_radius_m": 0.025,
    "waiting_padding_m": 0.0,
    "waiting_padding_extra_m": [0.0, 0.0, 0.0],
    "waiting_source_frames": None,
    "waiting_enabled": [True, True, True],
    "retreat_distance_m": 0.03,
    "minimum_clearance_m": 0.05,
}


def workspace_events(tcp, events, config):
    """Each admission boundary depends on a fixed box, not the opposite trajectory."""
    lo = np.asarray(config["minimum_m"], float)
    hi = np.asarray(config["maximum_m"], float)
    radius = float(config["ee_radius_m"] + config["waiting_padding_m"])
    retreat = float(config["retreat_distance_m"])
    if (
        lo.shape != (3,)
        or hi.shape != (3,)
        or not np.isfinite([lo, hi]).all()
        or np.any(hi <= lo)
        or not np.isfinite([radius, retreat]).all()
        or radius < 0
        or retreat <= 0
    ):
        raise ValueError("invalid fixed workspace geometry")
    inside = np.all((tcp >= lo - radius) & (tcp <= hi + radius), axis=-1)
    own = workpiece_events(events)
    records = [[], []]
    for side in range(2):
        previous_release = 0
        for event in own[side]:
            begin = max(previous_release, event["approach_start"])
            pickup = event["pickup_frame"]
            hits = np.flatnonzero(inside[side, begin : pickup + 1]) + begin
            if not len(hits) or hits[0] == begin:
                raise ValueError("no observed outside approach before workspace entry")
            entry = int(hits[0])
            direction = 1 if side == 0 else -1
            # A running extremum follows the actual lateral turn, including
            # continued inward travel after grasp. It never reads a future pose.
            inward_extreme = direction * tcp[side, pickup, 1]
            onset = pickup
            release = None
            for t in range(pickup, event["retract_end"] + 1):
                current = direction * tcp[side, t, 1]
                if current <= inward_extreme:
                    inward_extreme = current
                    onset = t
                if current - inward_extreme >= retreat:
                    release = onset
                    confirmed = t
                    break
            if release is None:
                raise ValueError("no three-centimetre lateral withdrawal observed")
            records[side].append(
                {
                    "entry_frame": entry,
                    "wait_frame": entry - 1,
                    "pickup_frame": pickup,
                    "withdrawal_frame": release,
                    "withdrawal_confirmed_frame": confirmed,
                    "withdrawal_direction": ("+Y" if side == 0 else "-Y"),
                }
            )
            previous_release = event["release_frame"]
    return records


def preparation_onsets(state, action, own):
    """Recover motion before visual work-area approach annotations."""
    from ..retime import MotionHeuristic, arm_motion_energy, detect_arm_segments

    heuristic = MotionHeuristic()
    segments = detect_arm_segments(state, action, heuristic)
    starts = []
    for side, phase_start in enumerate((0, segments.split)):
        anchor = own[side][0]["pickup_frame"]
        energy = arm_motion_energy(state, ("left", "right")[side], heuristic)
        active = np.flatnonzero(energy[phase_start : anchor + 1] > 0)
        if not len(active):
            raise ValueError("no measured preparation motion before first pickup")
        starts.append(
            max(
                phase_start,
                phase_start + int(active[0]) - heuristic.boundary_padding_frames,
            )
        )
    return starts


def source_clocks(own, stops, fps, *, starts):
    clocks, holds, ramps = [], [], []
    for side in range(2):
        clock, hold, ramp = approach_clock(
            starts[side],
            own[side][-1]["retract_end"],
            stops[side],
            fps,
            brake_after={
                stop: own[side][0]["release_frame"]
                for stop in stops[side]
                if stop > own[side][0]["release_frame"]
            },
        )
        if any(
            r["brake_source_start"] < own[side][0]["release_frame"]
            for r in ramp
            if r["source_frame"] > own[side][0]["release_frame"]
        ):
            raise ValueError("outside stopping ramp interrupts preceding placement")
        clocks.append(clock)
        holds.append(hold)
        ramps.append(ramp)
    return clocks, holds, ramps


def nominal_schedule(clocks, holds, events):
    gates = [
        (1, holds[1][0], 0, events[0][0]["withdrawal_frame"]),
        (0, holds[0][0], 1, events[1][0]["withdrawal_frame"]),
        (1, holds[1][1], 0, events[0][1]["withdrawal_frame"]),
    ]
    indices = [0, 0]
    output = [[], []]
    for _ in range(sum(map(len, clocks)) + 1):
        for side in range(2):
            output[side].append(clocks[side][indices[side]])
        if all(indices[s] == len(clocks[s]) - 1 for s in range(2)):
            break
        advancing = [indices[s] < len(clocks[s]) - 1 for s in range(2)]
        for side, stop, owner, withdrawal in gates:
            if indices[side] == stop and clocks[owner][indices[owner]] < withdrawal:
                advancing[side] = False
        if not any(advancing):
            raise ValueError("fixed workspace admission deadlock")
        indices = [i + int(move) for i, move in zip(indices, advancing)]
    else:
        raise ValueError("fixed workspace scheduler exceeded monotone path bound")
    return tuple(map(np.asarray, output))


def waiting_source_frames(inside, own, starts):
    """Locate entry after the previous completed placement, not a late visual label."""
    stops = [[], []]
    for stage, (side, cycle) in enumerate([(0, 1), (1, 0), (1, 1)]):
        begin = (
            starts[side] if cycle == 0 else own[side][cycle - 1]["release_frame"] + 1
        )
        pickup = own[side][cycle]["pickup_frame"]
        if begin >= pickup:
            raise ValueError("placement overlaps the following pickup")
        entries = (
            np.flatnonzero(
                ~inside[stage, begin:pickup] & inside[stage, begin + 1 : pickup + 1]
            )
            + begin
            + 1
        )
        if not len(entries):
            raise ValueError("no outside approach for fixed-volume waiting pose")
        stops[side].append(int(entries[0]) - 1)
    return stops


def workspace_planning_cache(timeline, joints, clearance_m):
    """Share FK and source-pair decisions across waiting-position candidates."""
    state, action, urdf, meshes = joints
    fk = PiperXClearance(
        state[:, :7], state[:, 7:], Path(urdf), Path(meshes), margin_m=clearance_m
    )
    own = workpiece_events(timeline["episodes"])
    return {
        "timeline": timeline,
        "joints": joints,
        "clearance_m": clearance_m,
        "fk": fk,
        "own": own,
        "starts": preparation_onsets(state, action, own),
        "tcp": np.array([[p[4] for p in arm] for arm in fk.poses]),
        "pose_cache": [{float(i): p for i, p in enumerate(arm)} for arm in fk.poses],
        "configuration_cache": {},
    }


def workspace_waiting_stops(cache, config):
    # The box defines waiting poses only. Withdrawal is not an admission gate.
    lo, hi = np.asarray(config["minimum_m"]), np.asarray(config["maximum_m"])
    radius = config["ee_radius_m"] + config["waiting_padding_m"]
    if (
        lo.shape != (3,)
        or hi.shape != (3,)
        or not np.isfinite([lo, hi]).all()
        or np.any(hi <= lo)
        or not np.isfinite(radius)
        or radius < 0
    ):
        raise ValueError("invalid waiting volume geometry")
    extra = np.asarray(config["waiting_padding_extra_m"], float)
    if extra.shape != (3,) or not np.isfinite(extra).all() or np.any(extra < 0):
        raise ValueError("invalid per-stage waiting padding")
    radii = (radius + extra)[:, None, None]
    paths = cache["tcp"][[0, 1, 1]]
    inside = np.all((paths >= lo - radii) & (paths <= hi + radii), axis=-1)
    automatic = waiting_source_frames(inside, cache["own"], cache["starts"])
    explicit = config["waiting_source_frames"]
    values = np.asarray(
        [automatic[0][0], *automatic[1]] if explicit is None else explicit, float
    )
    if (
        values.shape != (3,)
        or not np.isfinite(values).all()
        or np.any(values != np.floor(values))
    ):
        raise ValueError("invalid explicit waiting source frames")
    limits = [automatic[0][0], *automatic[1]]
    for stage, (side, cycle) in enumerate([(0, 1), (1, 0), (1, 1)]):
        begin = (
            cache["starts"][side]
            if cycle == 0
            else cache["own"][side][cycle - 1]["release_frame"] + 1
        )
        stop = int(values[stage])
        if not begin <= stop <= limits[stage] or inside[stage, stop]:
            raise ValueError("explicit wait is not on the outside preparation path")
    enabled = config["waiting_enabled"]
    if len(enabled) != 3 or any(type(x) is not bool for x in enabled):
        raise ValueError("invalid enabled waiting stages")
    return [
        [int(values[0])] if enabled[0] else [],
        [int(values[i]) for i in [1, 2] if enabled[i]],
    ]


def plan_workspace(timeline, joints, config=DEFAULT_WORKSPACE, *, cache=None):
    """Earliest non-preemptive right admissions, constrained by swept mesh gaps."""
    from time import perf_counter

    from ..collision.continuous import ContinuousClearance
    from ..collision.piperx import DEFAULT_BASE_SPACING_M
    from .scheduler import schedule_sources

    began = perf_counter()
    clearance = config["minimum_clearance_m"]
    cache = (
        workspace_planning_cache(timeline, joints, clearance)
        if cache is None
        else cache
    )
    if (
        cache["joints"] is not joints
        or cache["timeline"] is not timeline
        or cache["clearance_m"] != clearance
    ):
        raise ValueError("workspace cache belongs to different immutable inputs")
    fk, own, starts = (cache[k] for k in ["fk", "own", "starts"])
    stops = workspace_waiting_stops(cache, config)
    selected = workspace_waiting_stops(
        cache, dict(config, waiting_enabled=[True, True, True])
    )
    boundary = workspace_waiting_stops(
        cache,
        dict(config, waiting_enabled=[True, True, True], waiting_source_frames=None),
    )
    entry_sources = [boundary[1][0] + 1, boundary[1][1] + 1]
    enabled = config["waiting_enabled"]
    clocks, holds, ramps = source_clocks(own, stops, timeline["fps"], starts=starts)
    pose_cache = cache["pose_cache"]
    configuration_cache = cache["configuration_cache"]
    check = ContinuousClearance.on_source_clocks(
        fk, clocks, pose_cache, configuration_cache, clearance
    )
    pickups = [
        [int(np.searchsorted(clocks[s], e["pickup_frame"])) for e in own[s]]
        for s in range(2)
    ]
    gates = [
        (1, pickups[1][0], 0, pickups[0][0]),
        (0, pickups[0][1], 1, pickups[1][0]),
        (1, pickups[1][1], 0, pickups[0][1]),
    ]
    rejected = set()

    def safe(i, j, ni, nj):
        current, following = (i, j), (ni, nj)
        if any(
            current[s] < crossing <= following[s] and current[o] < prior
            for s, crossing, o, prior in gates
        ):
            return False
        return (
            (i, j, ni, nj) not in rejected
            and check.checker.configuration_safe(i, j)
            and check.checker.configuration_safe(ni, nj)
        )

    attempts = 0
    while True:
        attempts += 1
        schedule = schedule_sources(
            len(clocks[0]),
            len(clocks[1]),
            safe,
            left_priority=False,
            can_wait=lambda side, index: (
                index in holds[side] or index == len(clocks[side]) - 1
            ),
            earliest_right_admissions=tuple(
                int(np.searchsorted(clocks[1], source)) - 1 for source in entry_sources
            ),
        )
        edges = [
            tuple(map(int, e))
            for e in zip(
                schedule.left[:-1],
                schedule.right[:-1],
                schedule.left[1:],
                schedule.right[1:],
            )
        ]
        failed = [e for e in edges if not check(*e)]
        print(
            f"admission search {attempts}: {len(configuration_cache)} pose pairs, {len(failed)} rejected swept edges",
            flush=True,
        )
        if not failed:
            break
        rejected.update(failed)
    left, right = clocks[0][schedule.left], clocks[1][schedule.right]
    up = round(round(timeline["fps"] * 0.3) / 2)
    stages = {
        "policy": "earliest_right_mesh_clearance_admission",
        "workspace": config,
        "base_spacing_m": DEFAULT_BASE_SPACING_M,
        "wait_source_frames": {"left": stops[0], "right": stops[1]},
        "initial_source_frames": [c[0] for c in clocks],
        "onset_delay_frames": [0, 0],
        "right_preparation": {
            "source_start_frame": starts[1],
            "source_end_frame": boundary[1][0],
            "output_start_frame": 0,
            "output_end_frame": int(np.flatnonzero(right >= boundary[1][0])[0]),
        },
        "preparation_onsets": {
            "method": "first smoothed measured-joint motion in each sequential source phase, with boundary padding",
            "event_approach_start_frames": [
                group[0]["approach_start"] for group in own
            ],
            "restored_source_frames": [
                max(0, group[0]["approach_start"] - start)
                for group, start in zip(own, starts)
            ],
        },
        "protected_source_intervals": {
            "left": [
                [starts[0], own[0][0]["release_frame"]],
                [
                    selected[0][0] + up
                    if enabled[0]
                    else own[0][0]["release_frame"] + 1,
                    own[0][1]["retract_end"],
                ],
            ],
            "right": [
                [
                    selected[1][0] + up if enabled[1] else starts[1],
                    own[1][0]["release_frame"],
                ],
                [
                    selected[1][1] + up
                    if enabled[2]
                    else own[1][0]["release_frame"] + 1,
                    own[1][1]["retract_end"],
                ],
            ],
        },
        "ramps": {"left": ramps[0], "right": ramps[1]},
        "admissions": [
            {
                "arm": "right",
                "cycle": k + 1,
                "output_frame": int(np.flatnonzero(right >= source)[0]),
                "definition": "first observed workspace entry",
            }
            for k, source in enumerate(entry_sources)
        ],
    }
    verify_uninterrupted(left, right, stages)
    ordered = sorted(
        [
            {
                "arm": ["left", "right"][side],
                "source_frame": e["pickup_frame"],
                "output_frame": int(
                    np.flatnonzero((left, right)[side] >= e["pickup_frame"])[0]
                ),
            }
            for side in range(2)
            for e in own[side]
        ],
        key=lambda e: e["output_frame"],
    )
    if [e["arm"] for e in ordered] != ["left", "right", "left", "right"]:
        raise ValueError("pickup order violated")
    samples = check.checked_samples
    check.__call__.cache_clear()
    return (
        left,
        right,
        {
            "stages": stages,
            "output_frames": len(left),
            "pickup_order": ordered,
            "collision_scope": "cross-arm URDF meshes from measured state",
            "mesh_audit": {
                "observation.state": {
                    "passed": True,
                    "checked_interior_samples": samples,
                }
            },
            "clearance_requirement_m": clearance,
            "numerical_clearance_guard_m": 1e-6,
            "continuous_audit_method": "adaptive midpoint FK with conservative motion bounds",
            "simultaneous_workspace_occupancy_allowed": True,
            "optimization_objective": "earliest right-1 admission, then right-2 admission, then duration; fixed waiting poses and 30Hz clocks",
            "search_seconds": perf_counter() - began,
            "evaluated_pose_pairs": len(configuration_cache),
            "rejected_swept_edges": len(rejected),
            "search_attempts": attempts,
        },
    )
