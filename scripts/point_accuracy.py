"""Compare 256/512 BootsTAPIR around automatically observed origin departures."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import torch

from real_robot_data_retime.accuracy_experiment import load_scene
from real_robot_data_retime.interaction.discovery import (
    InteractionConfig,
    task_object_proposals,
)
from real_robot_data_retime.interaction.measurements import producer_fingerprint
from real_robot_data_retime.interaction.origin_events import (
    origin_departure_interval,
    tracking_seed_frame,
)
from real_robot_data_retime.interaction.registration import stabilize
from real_robot_data_retime.interaction.video import write_video
from real_robot_data_retime.model_experiment import sha256
from real_robot_data_retime.tracking.consensus import spatial_consensus
from real_robot_data_retime.tracking.tapir import BootsTapir, sample_object_points


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "arms", "checkpoint", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--task", choices=["workpiece", "letters", "drawer"], required=True)
    args = p.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(4)
    cv2.setNumThreads(4)
    raw, fps, packed = load_scene(args.input, args.arms)
    args.output.mkdir(parents=True, exist_ok=False)
    frames, transforms, confidence = stabilize(raw)
    n, h, w = frames.shape[:3]
    robot = np.unpackbits(packed[:, 0] | packed[:, 1], axis=-1, count=w).astype(bool)
    for t in range(n):
        robot[t] = cv2.warpAffine(
            robot[t].astype(np.uint8),
            transforms[t, :2],
            (w, h),
            flags=cv2.INTER_NEAREST,
        ).astype(bool)
    config = replace(
        InteractionConfig(),
        minimum_object_area=round(80 * (w / 424) ** 2),
        maximum_object_area=round(1800 * (w / 424) ** 2),
    )
    proposals = task_object_proposals(frames, args.task, config)
    models = {
        resolution: BootsTapir(args.checkpoint, resolution=resolution)
        for resolution in (256, 512)
    }
    receipt = {
        "input_sha256": sha256(args.input),
        "arm_report_sha256": sha256(args.arms / "report.json"),
        "checkpoint_sha256": sha256(args.checkpoint),
        "producer": producer_fingerprint(),
        "task": args.task,
        "random_seed": 0,
        "consensus_rule": {
            "minimum_points": 3,
            "minimum_fraction_of_visible": 0.6,
            "maximum_diameter_in_origin_box_diagonals": 3,
        },
        "objects": [],
        "validated_for_compositing": False,
        "limitation": "Local pickup windows; consensus can reject drift but cannot certify identity or separation.",
    }
    for k, proposal in enumerate(proposals):
        event, signal = origin_departure_interval(frames, proposal, robot, fps)
        entry = {"object_id": k, "event": event}
        receipt["objects"].append(entry)
        if event is None:
            entry["status"] = "no_verified_origin_departure"
            continue
        seed = tracking_seed_frame(event, fps)
        if signal[seed] <= 0.55 or robot[seed][proposal["mask"]].mean() >= 0.2:
            entry["status"] = "precontact_seed_not_visually_supported"
            continue
        start = max(0, seed - round(fps * 0.5))
        stop = min(n, event["first_observed_empty"] + round(fps * 2))
        # Keep real source time; every frame in this interaction window is used.
        if stop - start > 240:
            entry.update(
                status="window_exceeds_memory_budget", source_window=[start, stop]
            )
            continue
        seed_mask = proposal["mask"] & ~robot[seed]
        if cv2.erode(seed_mask.astype(np.uint8), np.ones((3, 3), np.uint8)).sum() < 3:
            entry["status"] = "no_unoccluded_seed_points"
            continue
        queries, _ = sample_object_points([{**proposal, "mask": seed_mask}], 24)
        entry["query_count"] = len(queries)
        entry["seed_arm_overlap_excluded_pixels"] = int(
            (proposal["mask"] & robot[seed]).sum()
        )
        queries[:, 0] = seed - start
        results = {}
        entry.update(
            status="measured",
            source_window=[start, stop],
            seed_source_frame=seed,
            registration_confidence_at_seed=float(confidence[seed]),
            resolutions={},
        )
        for resolution, model in models.items():
            print(
                f"object {k}, source [{start},{stop}), resolution {resolution}",
                flush=True,
            )
            xy, visible = model.track(frames[start:stop], queries)
            clean, kept, _centers = spatial_consensus(
                xy, visible, maximum_diameter=3 * np.hypot(*proposal["bbox"][2:])
            )
            # Restore each measured coordinate to the original source image.
            for local, source in enumerate(range(start, stop)):
                inverse = cv2.invertAffineTransform(transforms[source, :2])
                xy[local] = cv2.transform(xy[local, :, None], inverse)[:, 0]
                clean[local] = cv2.transform(clean[local, :, None], inverse)[:, 0]
            inside = (
                np.isfinite(xy).all(axis=-1)
                & (xy[..., 0] >= 0)
                & (xy[..., 0] < w)
                & (xy[..., 1] >= 0)
                & (xy[..., 1] < h)
            )
            visible &= inside
            kept &= inside
            xy[~visible] = np.nan
            clean[~kept] = np.nan
            results[resolution] = (xy, visible, clean, kept)
            np.savez_compressed(
                args.output / f"object-{k}-{resolution}.npz",
                xy=xy,
                visible=visible,
                consensus_xy=clean,
                consensus_visible=kept,
                queries_registered_tyx=queries,
                source_indices=np.arange(start, stop),
                registration=transforms[start:stop],
            )
            entry["resolutions"][resolution] = {
                "visible_point_fraction": float(visible.mean()),
                "retained_point_fraction": float(kept.mean()),
                "consensus_frame_fraction": float(kept.any(axis=1).mean()),
                "rejected_visible_points": int((visible & ~kept).sum()),
            }

        def preview(start=start, stop=stop, results=results, k=k):
            for local, source in enumerate(range(start, stop)):
                panels = []
                for resolution in (256, 512):
                    xy, visible, _, kept = results[resolution]
                    view = raw[source].copy()
                    for j in np.flatnonzero(visible[local]):
                        x, y = np.rint(xy[local, j]).astype(int)
                        cv2.circle(
                            view,
                            (x, y),
                            2,
                            (0, 220, 80) if kept[local, j] else (0, 0, 255),
                            -1,
                        )
                    cv2.putText(
                        view,
                        f"object {k}, {resolution}, source {source}",
                        (12, 24),
                        0,
                        0.6,
                        (255, 255, 255),
                        2,
                    )
                    panels.append(view)
                yield np.concatenate(panels, axis=1)

        write_video(args.output / f"object-{k}.mp4", preview(), fps)
        (args.output / "progress.json").write_text(
            json.dumps(receipt, indent=2, allow_nan=False)
        )
    (args.output / "report.json").write_text(
        json.dumps(receipt, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
