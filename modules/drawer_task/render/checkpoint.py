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


def render_checkpoint(source, analysis, output):
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
        left, right, plan = plan_visual(timeline, frames, tracks, segmentation)
        np.savez_compressed(output / "source_mapping.npz", left=left, right=right)
        native, masks, _ = native_render_inputs(
            source, tracks["registration"], segmentation
        )
        rendered = composite(
            native, timeline, masks, left, right, output / "parallel.mp4", output
        )
    passed = bool(rendered["automatic_origin_audit"]["passed"])
    result = {
        "source_sha256": sha256(source),
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
    args = parser.parse_args()
    result = render_checkpoint(args.input, args.analysis, args.output)
    print(json.dumps(result, indent=2))
    if not result["automatic_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
