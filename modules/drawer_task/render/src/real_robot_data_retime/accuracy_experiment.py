"""Accuracy probes with explicit source identity and no human inference prompts."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .interaction.measurements import producer_fingerprint
from .interaction.video import read_video, write_video
from .model_experiment import sha256
from .segmentation.gripper_refinement import refine_grippers, repair_gripper_gaps
from .segmentation.robotseg_backend import RobotSegPromptedVideo


def load_scene(source, arms):
    report = json.loads((arms / "report.json").read_text())
    if report["input_sha256"] != sha256(source):
        raise ValueError("arm measurements belong to a different source video")
    if report["source_start"] != 0 or report["stride"] != 1:
        raise ValueError("accuracy probes require source-aligned, full-frame-rate arms")
    frames, fps = read_video(
        source, stop=report["source_stop"], width=report["frame_shape"][1]
    )
    with np.load(arms / "masks.npz") as data:
        robots, indices = data["robot_masks"], data["source_indices"]
        if int(data["width"]) != frames.shape[2] or not np.array_equal(
            indices, np.arange(len(frames))
        ):
            raise ValueError("arm source geometry or frame mapping changed")
    if robots.shape != (len(frames), 2, frames.shape[1], (frames.shape[2] + 7) // 8):
        raise ValueError("invalid source-aligned arm masks")
    return frames, fps, robots


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "arms", "checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--single-frame",
        type=Path,
        help="Reuse verified direct observations to propose gap repairs",
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    cv2.setNumThreads(4)
    np.random.seed(0)
    frames, fps, robots = load_scene(args.input, args.arms)
    args.output.mkdir(parents=True, exist_ok=False)
    producer = producer_fingerprint()
    if args.single_frame is None:
        result = refine_grippers(frames, robots, RobotSegPromptedVideo(args.checkpoint))
        direct_producer = producer
    else:
        previous = json.loads((args.single_frame / "report.json").read_text())
        if "repairs" in previous:
            raise ValueError("gap proposals cannot be reused as direct observations")
        for key, value in {
            "input_sha256": sha256(args.input),
            "arm_report_sha256": sha256(args.arms / "report.json"),
            "checkpoint_sha256": sha256(args.checkpoint),
        }.items():
            if previous[key] != value:
                raise ValueError(f"direct gripper evidence mismatch: {key}")
        with np.load(args.single_frame / "grippers.npz") as data:
            if int(data["width"]) != frames.shape[2] or not np.array_equal(
                data["source_indices"], np.arange(len(frames))
            ):
                raise ValueError(
                    "direct gripper source geometry or time mapping changed"
                )
            result = {key: data[key] for key in ["raw_grippers", "grippers", "centers"]}
        result["observations"] = previous["observations"]
        direct_producer = previous["producer"]
        agreed, repairs = repair_gripper_gaps(
            frames, robots, result, RobotSegPromptedVideo(args.checkpoint), fps
        )
        result["agreed_grippers"] = agreed
    proposal_masks = (
        result["agreed_grippers"]
        if args.single_frame is not None
        else result["grippers"]
    )
    result["candidate_robot_masks"] = np.bitwise_or(robots, proposal_masks)
    np.savez_compressed(
        args.output / "grippers.npz",
        **{k: v for k, v in result.items() if k != "observations"},
        source_indices=np.arange(len(frames)),
        width=frames.shape[2],
    )
    observations = result["observations"]
    report = {
        "input_sha256": sha256(args.input),
        "arm_report_sha256": sha256(args.arms / "report.json"),
        "checkpoint_sha256": sha256(args.checkpoint),
        "producer": producer,
        "direct_observation_producer": direct_producer,
        "observations": observations,
        "validated_for_compositing": False,
        "added_robot_pixels": int(
            np.unpackbits(
                result["candidate_robot_masks"] & ~robots,
                axis=-1,
                count=frames.shape[2],
            ).sum()
        ),
        "limitation": "Semantic observations and support filtering are not pixel-level ground truth.",
    }

    if args.single_frame is not None:
        report["single_frame_report_sha256"] = sha256(args.single_frame / "report.json")
        report["repairs"] = repairs

    def preview():
        for t, frame in enumerate(frames):
            panels = []
            panels_to_show = [
                ("whole arm", robots),
                ("raw local gripper", result["raw_grippers"]),
                ("supported gripper", result["grippers"]),
            ]
            if args.single_frame is not None:
                panels_to_show.append(("agreement proposal", result["agreed_grippers"]))
            for name, packed in panels_to_show:
                view = frame.copy()
                for side, color in enumerate(([0, 200, 255], [255, 100, 0])):
                    mask = np.unpackbits(
                        packed[t, side], axis=-1, count=frames.shape[2]
                    ).astype(bool)
                    view[mask] = (0.55 * view[mask] + 0.45 * np.array(color)).astype(
                        np.uint8
                    )
                cv2.putText(
                    view, f"{name}, source {t}", (12, 24), 0, 0.6, (255, 255, 255), 2
                )
                panels.append(view)
            yield np.concatenate(panels, axis=1)

    write_video(args.output / "grippers.mp4", preview(), fps)
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "frames": len(frames),
                "observed_side_frames": sum(o["observed"] for o in observations),
            }
        )
    )


if __name__ == "__main__":
    main()
