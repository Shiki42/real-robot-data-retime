"""Shared I/O for the pinned AIST/Molmo research pilot."""

import argparse
import hashlib
import os
import subprocess
from pathlib import Path

import numpy as np


def work_dir():
    value = os.environ.get("RETIME_WORK_DIR")
    if not value:
        raise ValueError("Set RETIME_WORK_DIR or pass --work-dir")
    root = Path(value).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "experiments").mkdir(exist_ok=True)
    return root


def checkpoint_path():
    value = os.environ.get("SAM3_CHECKPOINT")
    if not value or not Path(value).is_file():
        raise ValueError("Provide an existing SAM3_CHECKPOINT or --checkpoint")
    return Path(value).expanduser().resolve()


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run_stage(main, *, renderer=False, segmentation=False):
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--work-dir", default=os.environ.get("RETIME_WORK_DIR"))
    if renderer:
        parser.add_argument("--episode", type=int, choices=[8, 15])
        parser.add_argument("--screened", action="store_true")
    if segmentation:
        parser.add_argument("--checkpoint", default=os.environ.get("SAM3_CHECKPOINT"))
    args = parser.parse_args()
    if not args.work_dir:
        parser.error("--work-dir or RETIME_WORK_DIR is required")
    os.environ["RETIME_WORK_DIR"] = args.work_dir
    if segmentation:
        if not args.checkpoint:
            parser.error("--checkpoint or SAM3_CHECKPOINT is required")
        os.environ["SAM3_CHECKPOINT"] = args.checkpoint
    main(args)


def write_video(path, frames, fps, width, height):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-threads",
            "2",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ],
        stdin=subprocess.PIPE,
    )
    try:
        for frame in frames:
            if frame.shape != (height, width, 3) or frame.dtype != np.uint8:
                raise ValueError("Expected a uint8 RGB frame of the declared size")
            process.stdin.write(np.ascontiguousarray(frame).tobytes())
    finally:
        process.stdin.close()
        code = process.wait()
    if code:
        raise RuntimeError(f"ffmpeg failed while writing {path}")
