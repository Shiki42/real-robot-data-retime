from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np


def compose_vertical_split(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.shape != right.shape or left.ndim != 3:
        raise ValueError(
            f"split frames must have equal HWC shapes: {left.shape}, {right.shape}"
        )
    split = left.shape[1] // 2
    output = left.copy()
    output[:, split:] = right[:, split:]
    return output


def remap_video(
    source: Path,
    output: Path,
    source_indices: np.ndarray,
    fps: int,
    width: int,
    height: int,
) -> None:
    indices = _validate_indices(source_indices)
    reader = _SequentialVideoReader(source)
    encoder = _open_encoder(output, fps, width, height)
    try:
        for target in indices:
            _write_frame(encoder, reader.sample(float(target)))
    finally:
        reader.close()
        _close_encoder(encoder, output)


def remap_vertical_split_video(
    source: Path,
    output: Path,
    left_source_indices: np.ndarray,
    right_source_indices: np.ndarray,
    fps: int,
    width: int,
    height: int,
) -> None:
    left_indices = _validate_indices(left_source_indices)
    right_indices = _validate_indices(right_source_indices)
    if len(left_indices) != len(right_indices):
        raise ValueError("left and right source mappings must have equal lengths")
    left_reader = _SequentialVideoReader(source)
    right_reader = _SequentialVideoReader(source)
    encoder = _open_encoder(output, fps, width, height)
    try:
        for left_index, right_index in zip(left_indices, right_indices):
            frame = compose_vertical_split(
                left_reader.read_to(int(left_index)),
                right_reader.read_to(int(right_index)),
            )
            _write_frame(encoder, frame)
    finally:
        left_reader.close()
        right_reader.close()
        _close_encoder(encoder, output)


class _SequentialVideoReader:
    def __init__(self, source: Path) -> None:
        self.source = source
        self.capture = cv2.VideoCapture(str(source))
        if not self.capture.isOpened():
            raise FileNotFoundError(f"failed to open video: {source}")
        self._flow_key = None
        self._flow = None
        self.index = -1
        self.frame: np.ndarray | None = None

    def read_to(self, target: int) -> np.ndarray:
        if target < self.index:
            if not self.capture.set(cv2.CAP_PROP_POS_FRAMES, target):
                raise RuntimeError(f"failed to seek {self.source} to frame {target}")
            self.index = target - 1
            self.frame = None
        while self.index < target:
            ok, frame = self.capture.read()
            if not ok:
                raise RuntimeError(
                    f"video {self.source} ended at frame {self.index} before {target}"
                )
            self.index += 1
            self.frame = frame
        if self.frame is None:
            raise RuntimeError(f"video {self.source} did not yield frame {target}")
        return self.frame

    def sample(self, target):
        lo, hi = int(np.floor(target)), int(np.ceil(target))
        if lo == hi:
            return self.read_to(lo)
        if self._flow_key != lo:
            from .compositing.interpolation import FlowFrames

            a = self.read_to(lo).copy()
            b = self.read_to(hi).copy()
            self._flow = FlowFrames(np.array([a, b]))
            self._flow_key = lo
        shape = self._flow.frames.shape[1:3]
        return self._flow.sample(target - lo, [np.ones(shape, bool)] * 2)[0]

    def close(self) -> None:
        self.capture.release()


def _open_encoder(
    output: Path, fps: int, width: int, height: int
) -> subprocess.Popen[bytes]:
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        str(output),
    ]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    if encoder.stdin is None:
        raise RuntimeError("ffmpeg encoder has no stdin")
    return encoder


def _write_frame(encoder: subprocess.Popen[bytes], frame: np.ndarray) -> None:
    if encoder.stdin is None:
        raise RuntimeError("ffmpeg encoder stdin closed unexpectedly")
    encoder.stdin.write(frame.tobytes())


def _close_encoder(encoder: subprocess.Popen[bytes], output: Path) -> None:
    if encoder.stdin is not None:
        encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError(f"ffmpeg failed while writing {output}")


def _validate_indices(source_indices: np.ndarray) -> np.ndarray:
    indices = np.asarray(source_indices, dtype=float)
    if indices.ndim != 1 or len(indices) == 0:
        raise ValueError("video source indices must be a non-empty vector")
    if not np.isfinite(indices).all() or indices.min() < 0:
        raise ValueError("video source indices must be non-negative")
    return indices
