"""Probe robot/arm/gripper semantics from an automatically selected visible pose."""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from real_robot_data_retime.interaction.discovery import motion_and_grippers
from real_robot_data_retime.interaction.video import read_video, write_video
from real_robot_data_retime.model_experiment import sha256
from real_robot_data_retime.segmentation.robotseg_backend import RobotSegVideo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(4)
    cv2.setNumThreads(4)
    frames, fps = read_video(args.input, width=640)
    frames = frames[::6]
    geometry = motion_and_grippers(frames)
    seed = int(np.argmax(np.count_nonzero(geometry["masks"], axis=(1, 2))))
    frames = frames[seed : seed + 32]
    model = RobotSegVideo(args.checkpoint)
    categories = ["robot", "arm", "gripper"]
    masks, times = [], {}
    for category in categories:
        print(
            json.dumps({"category": category, "seed_source_frame": seed * 6}),
            flush=True,
        )
        torch.cuda.synchronize()
        started = time.monotonic()
        category_masks = np.zeros(frames.shape[:3], bool)
        seen = set()
        for t, mask in model.propagate(frames, category=category):
            if t in seen or not 0 <= t < len(frames):
                raise ValueError("invalid semantic source frame")
            seen.add(t)
            category_masks[t] = mask
        if len(seen) != len(frames):
            raise ValueError("missing semantic source frames")
        torch.cuda.synchronize()
        times[category] = time.monotonic() - started
        masks.append(category_masks)
    packed = np.packbits(np.asarray(masks), axis=-1)
    indices = (seed + np.arange(len(frames))) * 6
    np.savez_compressed(
        args.output / "semantic_masks.npz",
        masks=packed,
        categories=categories,
        source_indices=indices,
        width=640,
    )

    def preview():
        for t, frame in enumerate(frames):
            panels = []
            for category, category_masks in zip(categories, masks):
                view = frame.copy()
                mask = category_masks[t]
                view[mask] = (
                    view[mask] * 0.55 + np.array([0, 180, 255]) * 0.45
                ).astype(np.uint8)
                cv2.putText(
                    view,
                    f"{category}, source {indices[t]}",
                    (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
                panels.append(view)
            yield np.concatenate(panels, axis=1)

    write_video(args.output / "categories.mp4", preview(), fps / 6)
    report = {
        "input_sha256": sha256(args.input),
        "checkpoint_sha256": sha256(args.checkpoint),
        "automatic_seed_source_frame": seed * 6,
        "frames": len(frames),
        "random_seed": 0,
        "category_seconds_excluding_model_load": times,
        "validated_for_compositing": False,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
