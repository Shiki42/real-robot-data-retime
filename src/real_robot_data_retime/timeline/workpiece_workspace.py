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
    "retreat_distance_m": 0.03,
    "minimum_clearance_m": 0.0155,
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


def source_clocks(own, stops, fps):
    clocks, holds, ramps = [], [], []
    for side in range(2):
        clock, hold, ramp = approach_clock(
            own[side][0]["approach_start"],
            own[side][-1]["retract_end"],
            stops[side],
            fps,
            brake_after={stops[side][-1]: own[side][0]["release_frame"]},
        )
        if any(
            r["brake_source_start"] < own[side][0]["release_frame"]
            for r in ramp[(1 if side == 1 else 0) :]
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


def plan_workspace(timeline, joints, config=DEFAULT_WORKSPACE):
    """Earliest non-preemptive right admissions, constrained by swept mesh gaps."""
    from time import perf_counter

    from ..collision.continuous import ContinuousClearance
    from ..collision.piperx import DEFAULT_BASE_SPACING_M
    from .scheduler import schedule_sources

    began = perf_counter()
    state, _, urdf, mesh_root = joints
    clearance = config["minimum_clearance_m"]
    fk = PiperXClearance(
        state[:, :7], state[:, 7:], Path(urdf), Path(mesh_root), margin_m=clearance
    )
    own = workpiece_events(timeline["episodes"])
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
    tcp = np.array([[p[4] for p in arm] for arm in fk.poses])
    inside = np.all((tcp >= lo - radius) & (tcp <= hi + radius), axis=-1)
    stops = [[], []]
    for side, cycle in [(0, 1), (1, 0), (1, 1)]:
        event = own[side][cycle]
        start = event["approach_start"]
        hits = np.flatnonzero(inside[side, start : event["pickup_frame"] + 1]) + start
        if not len(hits) or hits[0] <= start:
            raise ValueError("no outside approach for fixed-volume waiting pose")
        stops[side].append(int(hits[0]) - 1)
    clocks, holds, ramps = source_clocks(own, stops, timeline["fps"])
    pose_cache = [{float(i): p for i, p in enumerate(arm)} for arm in fk.poses]
    configuration_cache = {}
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
            earliest_right_admissions=tuple(holds[1]),
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
        "right_preparation": {"source_end_frame": stops[1][0]},
        "protected_source_intervals": {
            "left": [
                [own[0][0]["approach_start"], own[0][0]["release_frame"]],
                [stops[0][0] + up, own[0][1]["retract_end"]],
            ],
            "right": [
                [stops[1][0] + up, own[1][0]["release_frame"]],
                [stops[1][1] + up, own[1][1]["retract_end"]],
            ],
        },
        "ramps": {"left": ramps[0], "right": ramps[1]},
        "admissions": [
            {
                "arm": "right",
                "cycle": k + 1,
                "output_frame": int(np.flatnonzero(schedule.right > hold)[0]) - 1,
            }
            for k, hold in enumerate(holds[1])
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
