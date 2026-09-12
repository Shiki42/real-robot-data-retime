"""Fixed-volume admission, using measured withdrawal instead of future arm sweeps."""

from pathlib import Path
import numpy as np

from .workpiece import approach_clock, verify_uninterrupted
from ..collision.piperx import PiperXClearance
from ..tasks.workpiece import workpiece_events


DEFAULT_WORKSPACE = {
    "minimum_m": [0.27, -0.12, -0.04],
    "maximum_m": [0.46, 0.12, 0.18],
    "ee_radius_m": 0.025,
    "retreat_distance_m": 0.03,
    "minimum_clearance_m": 0.0155,
}


def workspace_events(tcp, events, config):
    """Each admission boundary depends on a fixed box, not the opposite trajectory."""
    lo = np.asarray(config["minimum_m"], float)
    hi = np.asarray(config["maximum_m"], float)
    radius = float(config["ee_radius_m"])
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
                dict(
                    entry_frame=entry,
                    wait_frame=entry - 1,
                    pickup_frame=pickup,
                    withdrawal_frame=release,
                    withdrawal_confirmed_frame=confirmed,
                    withdrawal_direction=("+Y" if side == 0 else "-Y"),
                )
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


def mesh_checkers(joints, clocks, clearance_m, *, include_action=False):
    from ..collision.continuous import ContinuousClearance
    from .smooth import sample_rows

    return [
        ContinuousClearance(
            sample_rows(v[:, :7], clocks[0]),
            sample_rows(v[:, 7:], clocks[1]),
            Path(joints[2]),
            Path(joints[3]),
            clearance_m=clearance_m,
        )
        for v in joints[: 2 if include_action else 1]
    ]


def admission_remaining_bound(lengths, gates):
    """Critical-path lower bound from the three alternating admission gates."""

    def remaining(i, j):
        current = (i, j)
        delay = [0, 0]
        for side, stop, owner, withdrawal in gates:
            if current[side] > stop:
                continue
            arrival = stop - current[side] + delay[side]
            release = (
                0
                if current[owner] >= withdrawal
                else withdrawal - current[owner] + delay[owner]
            )
            delay[side] += max(0, release - arrival)
        return max(lengths[side] - 1 - current[side] + delay[side] for side in range(2))

    return remaining


def plan_workspace(timeline, joints, config=DEFAULT_WORKSPACE):
    from .scheduler import schedule_sources, NoSafeSchedule
    from heapq import heappush, heappop

    state, action, urdf, mesh_root = joints
    clearance = config["minimum_clearance_m"]
    fk = PiperXClearance(
        state[:, :7], state[:, 7:], Path(urdf), Path(mesh_root), margin_m=clearance
    )
    tcp = np.array([[p[4] for p in arm] for arm in fk.poses])
    events = workspace_events(tcp, timeline["episodes"], config)
    own = workpiece_events(timeline["episodes"])
    fps = timeline["fps"]
    original_stops = [
        [events[0][1]["wait_frame"]],
        [e["wait_frame"] for e in events[1]],
    ]
    clocks, holds, ramps = source_clocks(own, original_stops, fps)
    nominal = nominal_schedule(clocks, holds, events)
    nominal_checks = mesh_checkers(joints, nominal, clearance, include_action=True)
    nominal_failures = {
        name: [i for i in range(len(nominal[0]) - 1) if not c(i, i, i + 1, i + 1)]
        for name, c in zip(("observation.state", "action"), nominal_checks)
    }
    for c in nominal_checks:
        c.__call__.cache_clear()
    del nominal_checks
    print(
        f"nominal clearance failed edges: { {k: len(v) for k, v in nominal_failures.items()} }",
        flush=True,
    )
    up = round(round(fps * 0.3) / 2)
    # Backoff candidates depend on the fixed volume and own approach only.
    candidates = [
        list(range(original_stops[0][0], own[0][0]["release_frame"], -1)),
        list(range(original_stops[1][0], own[1][0]["approach_start"], -1)),
        list(range(original_stops[1][1], own[1][0]["release_frame"], -1)),
    ]
    if any(not c for c in candidates):
        raise NoSafeSchedule("insufficient outside braking path")
    # Left 1 is uninterrupted, so these poses must occur while right 1 is
    # still gated. Prune impossible right waiting poses before full path search.
    source_geometry = [
        PiperXClearance(
            v[:, :7], v[:, 7:], Path(urdf), Path(mesh_root), margin_m=clearance + 1e-6
        )
        for v in joints[:1]
    ]
    valid_right = []
    for stop in candidates[1]:
        initial_clock, arrival, _ = approach_clock(
            own[1][0]["approach_start"], stop, [stop], fps
        )
        earliest_left = own[0][0]["approach_start"] + arrival[0]
        unavoidable = range(earliest_left, events[0][0]["withdrawal_frame"] + 1)
        if all(
            c.configuration_safe(t, stop) for t in unavoidable for c in source_geometry
        ):
            valid_right.append(stop)
    candidates[1] = valid_right
    if not valid_right:
        raise NoSafeSchedule(
            "no right outside waiting pose clears uninterrupted left 1 by required clearance"
        )
    valid_left = []
    for stop in candidates[0]:
        trial_clocks, trial_holds, _ = source_clocks(
            own, [[stop], [valid_right[0], original_stops[1][1]]], fps
        )
        early_left, early_right = nominal_schedule(trial_clocks, trial_holds, events)
        arrival = int(np.flatnonzero(early_left >= stop)[0])
        pickup = int(np.flatnonzero(early_right >= events[1][0]["pickup_frame"])[0])
        mandatory = range(
            events[1][0]["pickup_frame"], events[1][0]["withdrawal_frame"] + 1
        )
        if arrival > pickup or all(
            c.configuration_safe(stop, t) for t in mandatory for c in source_geometry
        ):
            valid_left.append(stop)
    candidates[0] = valid_left
    if not valid_left:
        raise NoSafeSchedule(
            "no outside left-2 waiting pose clears right-1 pickup-to-withdrawal by required clearance"
        )
    preferred = []
    for axis, (side, owner, begin, end) in enumerate(
        [
            (0, 1, own[1][0]["approach_start"], own[1][0]["release_frame"]),
            (1, 0, own[0][0]["approach_start"], own[0][0]["release_frame"]),
            (1, 0, own[0][0]["release_frame"] + 1, own[0][1]["retract_end"]),
        ]
    ):
        choice = 0
        for index, stop in enumerate(candidates[axis]):
            if all(
                c.configuration_safe(stop, t)
                if side == 0
                else c.configuration_safe(t, stop)
                for t in range(begin, end + 1)
                for c in source_geometry
            ):
                choice = index
                break
        preferred.append(choice)
    preferred = tuple(preferred)
    del source_geometry
    print(f"workspace clearance candidates: {[len(c) for c in candidates]}", flush=True)
    outermost = tuple(len(c) - 1 for c in candidates)
    frontier = [(0, preferred), (1, (0, 0, 0)), (2, outermost)]
    seen = {preferred, outermost, (0, 0, 0)}
    attempts = 0
    from ..collision.continuous import ContinuousClearance

    pose_cache = [{float(i): p for i, p in enumerate(arm)} for arm in fk.poses]
    configuration_cache = {}
    while frontier:
        _, choice = heappop(frontier)
        stops = [
            [candidates[0][choice[0]]],
            [candidates[1][choice[1]], candidates[2][choice[2]]],
        ]
        clocks, holds, ramps = source_clocks(own, stops, fps)
        checks = [
            ContinuousClearance.on_source_clocks(
                fk, clocks, pose_cache, configuration_cache, clearance
            )
        ]

        def dependency(i, j):
            l, r = clocks[0][i], clocks[1][j]
            return (
                (j <= holds[1][0] or l >= events[0][0]["withdrawal_frame"])
                and (i <= holds[0][0] or r >= events[1][0]["withdrawal_frame"])
                and (j <= holds[1][1] or l >= events[0][1]["withdrawal_frame"])
            )

        rejected_edges = set()

        def safe(i, j, ni, nj):
            current, following = (i, j), (ni, nj)
            if any(
                current[side] <= stop < following[side] and current[owner] < release
                for side, stop, owner, release in indexed_gates
            ):
                return False
            if (i, j, ni, nj) in rejected_edges:
                return False
            return all(
                c.checker.configuration_safe(i, j)
                and c.checker.configuration_safe(ni, nj)
                for c in checks
            )

        indexed_gates = [
            (
                1,
                holds[1][0],
                0,
                int(np.searchsorted(clocks[0], events[0][0]["withdrawal_frame"])),
            ),
            (
                0,
                holds[0][0],
                1,
                int(np.searchsorted(clocks[1], events[1][0]["withdrawal_frame"])),
            ),
            (
                1,
                holds[1][1],
                0,
                int(np.searchsorted(clocks[0], events[0][1]["withdrawal_frame"])),
            ),
        ]
        bound = admission_remaining_bound(list(map(len, clocks)), indexed_gates)
        attempts += 1
        if attempts == 1 or attempts % 25 == 0:
            print(f"workspace clearance search {attempts}: {stops}", flush=True)
        try:
            while True:
                schedule = schedule_sources(
                    len(clocks[0]),
                    len(clocks[1]),
                    safe,
                    dependency=dependency,
                    remaining_lower_bound=bound,
                    left_priority=False,
                    can_wait=lambda side, i: (
                        i in holds[side] or i == len(clocks[side]) - 1
                    ),
                )
                failed = []
                for edge in zip(
                    schedule.left[:-1],
                    schedule.right[:-1],
                    schedule.left[1:],
                    schedule.right[1:],
                ):
                    edge = tuple(map(int, edge))
                    if not all(c(*edge) for c in checks):
                        failed.append(edge)
                if not failed:
                    break
                rejected_edges.update(failed)
            left, right = clocks[0][schedule.left], clocks[1][schedule.right]
            break
        except NoSafeSchedule:
            for axis in range(3):
                neighbor = list(choice)
                neighbor[axis] += 1
                neighbor = tuple(neighbor)
                if neighbor[axis] < len(candidates[axis]) and neighbor not in seen:
                    seen.add(neighbor)
                    heappush(
                        frontier,
                        (
                            sum(c[0] - c[i] for c, i in zip(candidates, neighbor)),
                            neighbor,
                        ),
                    )
        finally:
            for c in checks:
                c.__call__.cache_clear()
    else:
        raise NoSafeSchedule(
            "no required clearance mesh-clearance schedule preserves uninterrupted execution"
        )
    stages = dict(
        policy="backdated_3cm_withdrawal_with_strict_mesh_clearance",
        workspace=config,
        events=events,
        wait_source_frames=dict(left=stops[0], right=stops[1]),
        nominal_wait_source_frames=dict(
            left=original_stops[0], right=original_stops[1]
        ),
        initial_source_frames=[c[0] for c in clocks],
        onset_delay_frames=[0, 0],
        right_preparation=dict(source_end_frame=stops[1][0]),
        protected_source_intervals=dict(
            left=[
                [own[0][0]["approach_start"], own[0][0]["release_frame"]],
                [stops[0][0] + up, own[0][1]["retract_end"]],
            ],
            right=[
                [stops[1][0] + up, own[1][0]["release_frame"]],
                [stops[1][1] + up, own[1][1]["retract_end"]],
            ],
        ),
        collision_search_attempts=attempts,
        ramps=dict(left=ramps[0], right=ramps[1]),
    )
    verify_uninterrupted(left, right, stages)
    final_checks = mesh_checkers(joints, (left, right), clearance, include_action=True)
    audit = {
        name: dict(
            passed=all(c(i, i, i + 1, i + 1) for i in range(len(left) - 1)),
            checked_interior_samples=c.checked_samples,
        )
        for name, c in zip(("observation.state", "action"), final_checks)
    }
    if not audit["observation.state"]["passed"]:
        raise ValueError("final mesh-clearance audit failed")
    stages["admissions"] = []
    for side, cycle, owner, owner_cycle in [(1, 0, 0, 0), (0, 1, 1, 0), (1, 1, 0, 1)]:
        clock = (left, right)[side]
        partner = (left, right)[owner]
        stop = stops[side][cycle if side == 1 else 0]
        started = int(np.flatnonzero(clock > stop)[0]) - 1
        release = int(
            np.flatnonzero(partner >= events[owner][owner_cycle]["withdrawal_frame"])[0]
        )
        stages["admissions"].append(
            dict(
                arm=("left", "right")[side],
                cycle=cycle + 1,
                retreat_start_output_frame=release,
                actual_start_output_frame=started,
                additional_wait_frames=started - release,
            )
        )
    pickups = sorted(
        [
            dict(
                arm=("left", "right")[side],
                source_frame=e["pickup_frame"],
                output_frame=int(
                    np.flatnonzero((left, right)[side] >= e["pickup_frame"])[0]
                ),
            )
            for side in range(2)
            for e in events[side]
        ],
        key=lambda e: e["output_frame"],
    )
    if [e["arm"] for e in pickups] != ["left", "right", "left", "right"]:
        raise ValueError("pickup order violated")
    return (
        left,
        right,
        dict(
            stages=stages,
            output_frames=len(left),
            pickup_order=pickups,
            collision_scope="cross-arm URDF meshes from measured state; action diagnostic only",
            nominal_collision_failures=nominal_failures,
            mesh_audit=audit,
            clearance_requirement_m=clearance,
            numerical_clearance_guard_m=1e-6,
            continuous_audit_method="adaptive midpoint FK with conservative motion bounds",
            commanded_action_is_acceptance_gate=False,
            simultaneous_workspace_occupancy_allowed=True,
        ),
    )
