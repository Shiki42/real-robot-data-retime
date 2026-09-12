"""Render a source-verified, successful interaction checkpoint for visual review."""

import argparse
import json
from pathlib import Path

import numpy as np

from real_robot_data_retime.compositing.layers import composite
from real_robot_data_retime.edit import native_render_inputs, registered_frames
from real_robot_data_retime.interaction.measurements import (
    inputs_fingerprint,
    producer_fingerprint,
)
from real_robot_data_retime.model_experiment import sha256
from real_robot_data_retime.timeline.visual import plan_visual
from real_robot_data_retime.staged import load_joints, export_trajectories


def render_checkpoint(
    source,
    analysis,
    output,
    *,
    joint_data=None,
    urdf=None,
    mesh_root=None,
    right_delay_seconds=0,
    left_delay_seconds=0,
    fixed_workspace=False,
):
    progress = json.loads((analysis / "progress.json").read_text())
    if progress["stage"] != "complete":
        raise ValueError("interaction checkpoint is unfinished")
    report = json.loads((analysis / "report.json").read_text())
    manifest = json.loads((analysis / "measurements.json").read_text())
    if manifest["inputs"]["video_sha256"] != sha256(source):
        raise ValueError("interaction checkpoint belongs to another source video")
    if not report["success"] or not all(report["validation_gates"].values()):
        raise ValueError("interaction checkpoint has failed verification gates")
    output.mkdir(parents=True, exist_ok=False)
    timeline = json.loads((analysis / "interaction_timeline.json").read_text())
    with (
        np.load(analysis / "tracks.npz") as tracks,
        np.load(analysis / "segmentation.npz") as segmentation,
    ):
        width = int(segmentation["frame_shape"][1])
        frames, fps = registered_frames(source, tracks["registration"], width)
        fingerprint = inputs_fingerprint(
            source, frames, manifest["inputs"]["proposals"], tracks["registration"]
        )
        if fingerprint != manifest["inputs"]:
            raise ValueError(
                "interaction checkpoint frame geometry or registration differs"
            )
        joints = (
            load_joints(joint_data, urdf, mesh_root, timeline) if joint_data else None
        )
        if fixed_workspace:
            if joints is None or timeline["task"] != "workpiece":
                raise ValueError("fixed EE workspace requires workpiece joints")
            if right_delay_seconds or left_delay_seconds:
                raise ValueError("fixed EE workspace uses synchronous startup")
            from real_robot_data_retime.timeline.workpiece_workspace import (
                plan_workspace,
            )

            left, right, plan = plan_workspace(timeline, joints)
        else:
            left, right, plan = plan_visual(
                timeline,
                frames,
                tracks,
                segmentation,
                joints=joints,
                right_delay_seconds=right_delay_seconds,
                left_delay_seconds=left_delay_seconds,
            )
        if joints is not None:
            export_trajectories(output, joints, left, right, fps, plan)
        np.savez_compressed(output / "source_mapping.npz", left=left, right=right)
        native, masks, _ = native_render_inputs(
            source, tracks["registration"], segmentation
        )
        rendered = composite(
            native,
            timeline,
            masks,
            left,
            right,
            output / "parallel.mp4",
            output,
            allow_projected_link_overlap=fixed_workspace,
        )
    passed = bool(rendered["automatic_origin_audit"]["passed"])
    result = {
        "source_sha256": sha256(source),
        "joint_data_sha256": sha256(joint_data) if joint_data else None,
        "analysis_report_sha256": sha256(analysis / "report.json"),
        "measurement_producer": manifest["producer"],
        "render_code_producer": producer_fingerprint(),
        "plan": plan,
        "compositing": rendered,
        "source_fps": fps,
        "automatic_checks_passed": passed,
        "validated_for_compositing": False,
        "status": "rendered_pending_visual_review"
        if passed
        else "failed_automatic_verification",
    }
    (output / "report.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "analysis", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--fixed-workspace", action="store_true")
    parser.add_argument("--joint-data", type=Path)
    parser.add_argument("--urdf", type=Path)
    parser.add_argument("--mesh-root", type=Path)
    parser.add_argument("--right-delay-seconds", type=float, default=0)
    parser.add_argument("--left-delay-seconds", type=float, default=0)
    args = parser.parse_args()
    result = render_checkpoint(
        args.input,
        args.analysis,
        args.output,
        fixed_workspace=args.fixed_workspace,
        joint_data=args.joint_data,
        urdf=args.urdf,
        mesh_root=args.mesh_root,
        right_delay_seconds=args.right_delay_seconds,
        left_delay_seconds=args.left_delay_seconds,
    )
    print(json.dumps(result, indent=2))
    if not result["automatic_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
