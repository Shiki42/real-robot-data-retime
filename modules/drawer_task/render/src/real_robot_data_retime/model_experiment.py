"""Reproducible measurement experiments, separate from validated compositing.

Run with ``python -m real_robot_data_retime.model_experiment --help``.
No clicks, event labels, or robot telemetry are used as inference inputs.
"""

import argparse
import hashlib
import json
import os
import random
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from .interaction.discovery import (
    InteractionConfig,
    motion_and_grippers,
    task_object_proposals,
)
from .interaction.video import read_video, write_video
from .tracking.tapir import BootsTapir, sample_object_points


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--backend",
        choices=["tapir", "cotracker", "robotseg", "robotseg-prompted", "sam2"],
        required=True,
    )
    parser.add_argument(
        "--task", choices=["letters", "workpiece", "drawer"], required=True
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, required=True)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--points-per-object", type=int, default=32)
    parser.add_argument("--tapir-resolution", type=int, default=256)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--upstream-repo", type=Path)
    args = parser.parse_args()
    if not 0 <= args.seed < 2**32:
        parser.error("seed must be between 0 and 2**32 - 1")
    if args.cpu_threads < 1:
        parser.error("cpu-threads must be positive")
    if args.stride < 1:
        parser.error("stride must be positive")
    if args.backend.startswith("robotseg") and args.checkpoint is None:
        parser.error("RobotSeg requires --checkpoint /path/to/robotseg.pt")
    if args.checkpoint is not None and args.backend not in {
        "tapir",
        "robotseg",
        "robotseg-prompted",
    }:
        parser.error("--checkpoint is only used by tapir/robotseg")
    args.output.mkdir(parents=True, exist_ok=False)
    import torch

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    cv2.setNumThreads(args.cpu_threads)
    print(json.dumps({"stage": "decode", "backend": args.backend}), flush=True)
    started = time.monotonic()
    frames, fps = read_video(args.input, args.start, args.stop, args.width)
    source_indices = np.arange(args.start, args.stop, args.stride)
    frames = frames[:: args.stride]
    n, h, w = frames.shape[:3]
    geometry = None
    if args.backend in {"tapir", "cotracker"}:
        config = replace(
            InteractionConfig(),
            minimum_object_area=round(80 * (w / 424) ** 2),
            maximum_object_area=round(1800 * (w / 424) ** 2),
        )
        proposals = task_object_proposals(frames, args.task, config)
        queries, owners = sample_object_points(proposals, args.points_per_object)
    else:
        geometry = motion_and_grippers(frames)
    preprocessing_seconds = time.monotonic() - started
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    gpu_before = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.used,utilization.gpu",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    print(json.dumps({"stage": "model", "frames": n}), flush=True)
    source_hashes = {
        str(path.relative_to(Path(__file__).parent)): sha256(path)
        for path in sorted(Path(__file__).parent.rglob("*.py"))
    }
    model_start = time.monotonic()
    if args.backend == "tapir":
        model = BootsTapir(args.checkpoint, resolution=args.tapir_resolution)
        xy, visible = model.track(frames, queries)
    elif args.backend == "cotracker":
        from .tracking.points import track_points

        xy, visible, indices = track_points(
            frames, [{"origin": q[[2, 1]]} for q in queries], stride=1
        )
        if not np.array_equal(indices, np.arange(n)):
            raise ValueError("CoTracker source frame mapping changed")
    else:
        packed = np.zeros((n, 2, h, (w + 7) // 8), np.uint8)
        unknown = np.zeros((n, h, (w + 7) // 8), np.uint8)
        if args.backend in {"sam2", "robotseg-prompted"}:
            from .interaction.neural_tracks import segment_robots
            from .segmentation.sam_backend import SamVideo

            if args.backend == "sam2":
                model = SamVideo("facebook/sam2.1-hiera-large")
            else:
                from .segmentation.robotseg_backend import RobotSegPromptedVideo

                model = RobotSegPromptedVideo(args.checkpoint)
            packed, _seeds = segment_robots(frames, geometry, model)
        else:
            from .segmentation.robotseg_backend import (
                RobotSegVideo,
                split_anchored_robots,
            )

            model = RobotSegVideo(args.checkpoint)
            seen = set()
            for t, mask in model.segment_semantic(frames):
                if t in seen or not 0 <= t < n or mask.shape != (h, w):
                    raise ValueError("RobotSeg frame contract violated")
                seen.add(t)
                sides, ambiguous = split_anchored_robots(mask)
                packed[t] = np.packbits(sides, axis=-1)
                unknown[t] = np.packbits(ambiguous, axis=-1)
            if len(seen) != n:
                raise ValueError("RobotSeg did not return every input frame")
    torch.cuda.synchronize()
    model_seconds = time.monotonic() - model_start
    peak_bytes = torch.cuda.max_memory_allocated()
    report = {
        "backend": args.backend,
        "input": str(args.input.resolve()),
        "input_sha256": sha256(args.input),
        "source_start": args.start,
        "source_stop": args.stop,
        "stride": args.stride,
        "source_fps": fps,
        "decoded_frames": n,
        "frame_shape": [h, w],
        "preprocessing_seconds": preprocessing_seconds,
        "model_load_and_inference_seconds": model_seconds,
        "processed_fps": n / model_seconds,
        "peak_allocated_bytes": peak_bytes,
        "gpu_at_start": gpu_before,
        "torch_version": torch.__version__,
        "cpu_threads": args.cpu_threads,
        "random_seed": args.seed,
        "runtime_env": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "PYTORCH_CUDA_ALLOC_CONF",
            )
        },
        "experiment_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "worktree_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        ),
        "source_hashes": source_hashes,
        "validated_for_compositing": False,
        "limitation": "Shared-GPU stage measurement; visibility/coverage is not identity or event ground truth.",
    }
    if args.backend.startswith("robotseg"):
        report["robotseg_preparation"] = model.timings
    if args.upstream_repo:
        report["upstream_commit"] = subprocess.check_output(
            ["git", "-C", str(args.upstream_repo), "rev-parse", "HEAD"], text=True
        ).strip()
    if args.backend == "tapir":
        report["checkpoint_sha256"] = sha256(model.checkpoint)
        report["tapir_resolution"] = args.tapir_resolution
    elif args.checkpoint:
        report["checkpoint_sha256"] = sha256(args.checkpoint)
    if args.backend in {"sam2", "cotracker"}:
        from .segmentation.model_versions import MODEL_REVISIONS

        report["model_revisions"] = MODEL_REVISIONS
    colors = [(0, 200, 255), (255, 100, 0), (60, 255, 80), (255, 80, 255)]
    if args.backend in {"tapir", "cotracker"}:
        np.savez_compressed(
            args.output / "tracks.npz",
            xy=xy,
            visible=visible,
            queries_tyx=queries,
            object_ids=owners,
            source_indices=source_indices,
        )
        report.update(
            point_count=len(queries),
            object_count=len(proposals),
            visible_fraction=float(visible.mean()),
            per_object_visible_fraction=[
                float(visible[:, owners == k].mean()) for k in range(len(proposals))
            ],
        )

        def preview():
            for t, frame in enumerate(frames):
                view = frame.copy()
                for k in np.flatnonzero(visible[t]):
                    x, y = np.rint(xy[t, k]).astype(int)
                    cv2.circle(view, (x, y), 2, colors[owners[k] % len(colors)], -1)
                yield view
    else:
        from .interaction.robot_discovery import robot_mask_audit

        report["motion_coverage_audit"] = robot_mask_audit(
            packed, geometry, fps / args.stride
        )
        report["unassigned_pixels"] = int(
            np.unpackbits(unknown, axis=-1, count=w).sum()
        )
        np.savez_compressed(
            args.output / "masks.npz",
            robot_masks=packed,
            unassigned_masks=unknown,
            source_indices=source_indices,
            width=w,
        )

        def preview():
            for t, frame in enumerate(frames):
                view = frame.copy()
                masks = np.unpackbits(packed[t], axis=-1, count=w).astype(bool)
                for side in range(2):
                    view[masks[side]] = (
                        0.55 * view[masks[side]] + 0.45 * np.array(colors[side])
                    ).astype(np.uint8)
                ambiguous = np.unpackbits(unknown[t], axis=-1, count=w).astype(bool)
                view[ambiguous] = (
                    0.55 * view[ambiguous] + 0.45 * np.array([0, 0, 255])
                ).astype(np.uint8)
                yield view

    write_video(args.output / "overlay.mp4", preview(), fps / args.stride)
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
