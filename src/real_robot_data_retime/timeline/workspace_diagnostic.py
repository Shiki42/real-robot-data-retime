"""Explicitly unvalidated nominal timing for inspection of clearance violations."""

from pathlib import Path

import numpy as np

from ..collision.piperx import PiperXClearance
from ..tasks.workpiece import workpiece_events
from .workpiece_workspace import (
    DEFAULT_WORKSPACE,
    nominal_schedule,
    preparation_onsets,
    source_clocks,
    workspace_events,
)


def nominal_plan(timeline, joints, config=DEFAULT_WORKSPACE):
    state, _, urdf, meshes = joints
    fk = PiperXClearance(
        state[:, :7],
        state[:, 7:],
        Path(urdf),
        Path(meshes),
        margin_m=config["minimum_clearance_m"],
    )
    tcp = np.array([[p[4] for p in arm] for arm in fk.poses])
    events = workspace_events(tcp, timeline["episodes"], config)
    own = workpiece_events(timeline["episodes"])
    stops = [[events[0][1]["wait_frame"]], [e["wait_frame"] for e in events[1]]]
    clocks, holds, ramps = source_clocks(
        own,
        stops,
        timeline["fps"],
        starts=preparation_onsets(joints[0], joints[1], own),
    )
    left, right = nominal_schedule(clocks, holds, events)
    pickups = sorted(
        [
            {
                "arm": ["left", "right"][side],
                "source_frame": e["pickup_frame"],
                "output_frame": int(
                    np.flatnonzero((left, right)[side] >= e["pickup_frame"])[0]
                ),
            }
            for side in range(2)
            for e in events[side]
        ],
        key=lambda e: e["output_frame"],
    )
    return (
        left,
        right,
        {
            "output_frames": len(left),
            "pickup_order": pickups,
            "diagnostic_only": True,
            "validated_for_rendering": False,
            "collision_scope": "not clearance validated; nominal retrospective release only",
            "stages": {
                "policy": "diagnostic_3cm_backdated_nominal",
                "events": events,
                "workspace": config,
                "wait_source_frames": {"left": stops[0], "right": stops[1]},
                "ramps": {"left": ramps[0], "right": ramps[1]},
            },
        },
    )
