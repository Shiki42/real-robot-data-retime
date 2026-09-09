"""Resegment weak, entry-supported arm observations in source-resolution crops."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from real_robot_data_retime.accuracy_experiment import load_scene
from real_robot_data_retime.interaction.discovery import motion_and_grippers
from real_robot_data_retime.interaction.measurements import producer_fingerprint
from real_robot_data_retime.interaction.robot_discovery import (
    prompt_from_robot_region,
    robot_entry_side,
    robot_mask_audit,
)
from real_robot_data_retime.interaction.video import write_video
from real_robot_data_retime.model_experiment import sha256
from real_robot_data_retime.segmentation.arm_refinement import assess_arm_reseed
from real_robot_data_retime.segmentation.robotseg_backend import RobotSegPromptedVideo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "arms", "checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(4)
    cv2.setNumThreads(4)
    frames, fps, original = load_scene(args.input, args.arms)
    args.output.mkdir(parents=True, exist_ok=False)
    producer = producer_fingerprint()
    n, h, w = frames.shape[:3]
    geometry = motion_and_grippers(frames)
    model = RobotSegPromptedVideo(args.checkpoint)
    candidate = original.copy()
    records = []
    for t, labels in enumerate(geometry["masks"]):
        for side in (0, 1):
            reference = labels == side + 1
            area = int(reference.sum())
            if area < h * w * 0.015 or robot_entry_side(reference) != side:
                continue
            before = np.unpackbits(original[t, side], axis=-1, count=w).astype(bool)
            coverage = float((before & reference).sum() / area)
            if coverage >= 0.6:
                continue
            p = prompt_from_robot_region(reference)
            x, y, bw, bh = p["bbox"]
            local = reference[y : y + bh, x : x + bw]
            prompt = prompt_from_robot_region(local)
            other = (labels[y : y + bh, x : x + bw] == 2 - side) & ~local
            yy, xx = np.where(other)
            ids = np.linspace(0, len(xx) - 1, min(8, len(xx)), dtype=int)
            prompt["negative_points"] = (
                np.column_stack([xx[ids], yy[ids]]).astype(float).tolist()
            )
            output = list(
                model.propagate(frames[t : t + 1, y : y + bh, x : x + bw], [prompt])
            )
            if (
                len(output) != 1
                or output[0][0] != 0
                or output[0][1].shape != (1, bh, bw)
            ):
                raise ValueError(
                    "current-frame robot predictor violated source crop contract"
                )
            mask = np.zeros((h, w), bool)
            mask[y : y + bh, x : x + bw] = output[0][1][0]
            p["negative_points"] = [
                [px + x, py + y] for px, py in prompt["negative_points"]
            ]
            full = list(model.propagate(frames[t : t + 1], [p]))
            if len(full) != 1 or full[0][0] != 0 or full[0][1].shape != (1, h, w):
                raise ValueError("full-frame robot predictor violated source contract")
            selected, record = assess_arm_reseed(
                before, mask, full[0][1][0], reference, side
            )
            candidate[t, side] = np.packbits(selected, axis=-1)
            records.append({"frame": t, "side": side, **record})
        if t % 60 == 0:
            print(f"reseed {t}/{n}", flush=True)
    np.savez_compressed(
        args.output / "masks.npz",
        original_robots=original,
        candidate_robots=candidate,
        source_indices=np.arange(n),
        width=w,
    )

    def preview():
        for t, frame in enumerate(frames):
            panels = []
            for label, masks in [
                ("whole-arm tracking", original),
                ("reseed proposal", candidate),
            ]:
                view = frame.copy()
                for side, color in enumerate(([0, 200, 255], [255, 100, 0])):
                    mask = np.unpackbits(masks[t, side], axis=-1, count=w).astype(bool)
                    view[mask] = (view[mask] * 0.55 + np.asarray(color) * 0.45).astype(
                        np.uint8
                    )
                cv2.putText(
                    view, f"{label}, source {t}", (12, 24), 0, 0.6, (255, 255, 255), 2
                )
                panels.append(view)
            yield np.concatenate(panels, axis=1)

    write_video(args.output / "reseed.mp4", preview(), fps)
    report = {
        "input_sha256": sha256(args.input),
        "arm_report_sha256": sha256(args.arms / "report.json"),
        "checkpoint_sha256": sha256(args.checkpoint),
        "producer": producer,
        "frames": n,
        "method": "crop_full_current_frame_agreement",
        "records": records,
        "motion_audit_before": robot_mask_audit(original, geometry, fps),
        "motion_audit_after": robot_mask_audit(candidate, geometry, fps),
        "validated_for_compositing": False,
        "limitation": "Motion coverage selects proposals, not pixel-accuracy ground truth.",
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
