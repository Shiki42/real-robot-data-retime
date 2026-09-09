from pathlib import Path
import cv2
import numpy as np


def read_video(path, start=0, stop=None, width=640):
    if width < 2 or width % 2 or start < 0 or (stop is not None and stop <= start):
        raise ValueError("invalid decode width or frame interval")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not np.isfinite(fps) or fps <= 0:
        cap.release()
        raise ValueError("video has no valid frame rate")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    while stop is None or start + len(frames) < stop:
        ok, frame = cap.read()
        if not ok:
            break
        height = 2 * round(frame.shape[0] * width / frame.shape[1] / 2)
        frames.append(cv2.resize(frame, (width, height)))
    cap.release()
    if not frames or (stop is not None and len(frames) != stop - start):
        raise ValueError("video is empty or shorter than episode metadata")
    if stop is None and expected > 0 and len(frames) != expected - start:
        raise ValueError("decoded frame count differs from video metadata")
    return np.asarray(frames), fps


def write_video(path, frames, fps):
    import subprocess

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    first = next(frames)
    h, w = first.shape[:2]
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "bgr24",
            "-video_size",
            f"{w}x{h}",
            "-framerate",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        stdin=subprocess.PIPE,
    )
    process.stdin.write(first.tobytes())
    for frame in frames:
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError(f"video encoding failed: {path}")


def components(mask, min_area=12, max_area=None):
    count, labels, stats, centers = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8)
    )
    return [
        (labels == k, stats[k], centers[k])
        for k in range(1, count)
        if stats[k, cv2.CC_STAT_AREA] >= min_area
        and (max_area is None or stats[k, cv2.CC_STAT_AREA] <= max_area)
    ]


def pixel_kernel(base, width, reference=424):
    size = max(base, round(base * width / reference))
    return int(size if size % 2 else size + 1)
