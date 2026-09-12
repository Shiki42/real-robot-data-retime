"""Generate and audit fixed-workspace timing candidates from verified joint data."""

import argparse
import json
from pathlib import Path
import numpy as np

from real_robot_data_retime.staged import load_joints
from real_robot_data_retime.timeline.workpiece_workspace import (
    DEFAULT_WORKSPACE,
    plan_workspace,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("analysis", "joint-data", "urdf", "mesh-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--workspace", type=Path)
    args = parser.parse_args()
    progress = json.loads((args.analysis / "progress.json").read_text())
    analysis = json.loads((args.analysis / "report.json").read_text())
    if progress["stage"] != "complete" or not analysis["success"]:
        raise ValueError("a completed successful interaction analysis is required")
    timeline = json.loads((args.analysis / "interaction_timeline.json").read_text())
    if timeline["task"] != "workpiece":
        raise ValueError("workpiece analysis required")
    config = (
        json.loads(args.workspace.read_text()) if args.workspace else DEFAULT_WORKSPACE
    )
    joints = load_joints(args.joint_data, args.urdf, args.mesh_root, timeline)
    left, right, report = plan_workspace(timeline, joints, config)
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output / "candidate_mapping.npz", left=left, right=right)
    report.update(
        validated_for_rendering=False,
        status="ee_workspace_passed_pending_visual_review",
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(status=report["status"], output=str(args.output)), indent=2))


if __name__ == "__main__":
    main()
