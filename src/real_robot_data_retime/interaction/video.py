from pathlib import Path
import cv2
import numpy as np


def read_video(path, start=0, stop=None, width=640):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    while stop is None or start + len(frames) < stop:
        ok, frame = cap.read()
        if not ok:
            break
        height = round(frame.shape[0] * width / frame.shape[1])
        frames.append(cv2.resize(frame, (width, height)))
    cap.release()
    if not frames or (stop is not None and len(frames) != stop - start):
        raise ValueError("video is empty or shorter than episode metadata")
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
