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
    "retreat_distance_m": 0.04,
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
        or radius <= 0
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
            # An online running extremum follows the actual lateral turn, including
            # continued inward travel after grasp. It never reads a future pose.
            inward_extreme = direction * tcp[side, pickup, 1]
            release = None
            for t in range(pickup, event["retract_end"] + 1):
                current = direction * tcp[side, t, 1]
                inward_extreme = min(inward_extreme, current)
                if current - inward_extreme >= retreat and not inside[side, t]:
                    release = t
                    break
            if release is None:
                raise ValueError("withdrawal did not clear the fixed workspace")
            records[side].append(
                dict(
                    entry_frame=entry,
                    wait_frame=entry - 1,
                    pickup_frame=pickup,
                    withdrawal_frame=release,
                )
            )
            previous_release = event["release_frame"]
    return records


def plan_workspace(timeline, joints, config=DEFAULT_WORKSPACE):
    state, action, urdf, mesh_root = joints
    checker = PiperXClearance(
        state[:, :7],
        state[:, 7:],
        Path(urdf),
        Path(mesh_root),
        margin_m=0.005,
    )
    tcp = np.array([[p[4] for p in arm] for arm in checker.poses])
    events = workspace_events(tcp, timeline["episodes"], config)
    own = workpiece_events(timeline["episodes"])
    fps = timeline["fps"]
    # Left 1 owns the workspace from startup. Every later action stages outside.
    stops = [[events[0][1]["wait_frame"]], [e["wait_frame"] for e in events[1]]]
    clocks, holds, ramps = [], [], []
    for side in range(2):
        clock, hold, ramp = approach_clock(
            own[side][0]["approach_start"],
            own[side][-1]["retract_end"],
            stops[side],
            fps,
        )
        if any(
            r["brake_source_start"] < own[side][0]["release_frame"]
            for r in ramp[(1 if side == 1 else 0) :]
        ):
            raise ValueError("outside stopping ramp interrupts preceding placement")
        clocks.append(clock)
        holds.append(hold)
        ramps.append(ramp)
    # Gates use the previously observed arm withdrawal and current fixed boundary.
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
    left, right = map(np.asarray, output)
    up = round(round(fps * 0.3) / 2)
    stages = dict(
        policy="fixed_workspace_observed_withdrawal",
        workspace=config,
        events=events,
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
    )
    verify_uninterrupted(left, right, stages)
    audit = audit_ee_workspace(checker, left, right, config)
    if not audit["passed"]:
        raise ValueError(
            f"simultaneous EE workspace occupancy: {audit['failed_output_edges'][:10]}"
        )
    pickup_order = sorted(
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
    if [e["arm"] for e in pickup_order] != ["left", "right", "left", "right"]:
        raise ValueError("fixed-workspace pickup order violated")
    return (
        left,
        right,
        dict(
            stages=stages,
            output_frames=len(left),
            collision_scope="EE fixed-workspace occupancy only; links excluded",
            ee_workspace_audit=audit,
            pickup_order=pickup_order,
        ),
    )


def audit_ee_workspace(checker, left, right, config):
    """Audit actual interpolated FK at quarter-output-frame steps; ignore link meshes."""
    from .smooth import sample_rows

    lo = np.asarray(config["minimum_m"]) - config["ee_radius_m"]
    hi = np.asarray(config["maximum_m"]) + config["ee_radius_m"]
    failures = []
    minimum_distance = float("inf")
    for i in range(len(left) - 1):
        for u in (0, 0.25, 0.5, 0.75, 1):
            positions = []
            for side, clock in enumerate((left, right)):
                source = clock[i] + u * (clock[i + 1] - clock[i])
                row = sample_rows(checker.values[side], np.array([source]))[0]
                positions.append(checker._pose(row, side)[4])
            inside = [bool(np.all((p >= lo) & (p <= hi))) for p in positions]
            minimum_distance = min(
                minimum_distance, float(np.linalg.norm(positions[0] - positions[1]))
            )
            if all(inside):
                failures.append(i)
                break
    return dict(
        passed=not failures,
        failed_output_edges=failures,
        substeps_per_output_interval=4,
        method="interpolated_joint_FK_EE_envelope",
        minimum_sampled_ee_distance_m=minimum_distance,
        link_collision_check=False,
    )
