"""Audit retimed numeric source identity, wrist pixels, and metadata consistency."""

import json
from pathlib import Path
import cv2
import numpy as np
from .trim import source_episodes, read_episode
from .video import _SequentialVideoReader


def wrist_pixel_error(source, output, mapping, samples=24):
    original, edited = (
        _SequentialVideoReader(Path(source)),
        _SequentialVideoReader(Path(output)),
    )
    errors = []
    try:
        for target in np.unique(
            np.linspace(0, len(mapping) - 1, min(samples, len(mapping))).astype(int)
        ):
            expected = original.read_to(int(mapping[target]))
            actual = edited.read_to(int(target))
            if expected.shape != actual.shape:
                raise ValueError("wrist video dimensions changed")
            errors.append(float(np.abs(expected.astype(float) - actual).mean()))
    finally:
        original.close()
        edited.close()
    if max(errors) > 6:
        raise ValueError(
            f"wrist pixels do not match mapped source: error {max(errors):.3f}"
        )
    return dict(samples=len(errors), maximum_mean_absolute_pixel_error=max(errors))


def validate(source, output):
    source, output = Path(source), Path(output)
    si = json.loads((source / "meta/info.json").read_text())
    oi = json.loads((output / "meta/info.json").read_text())
    original_rows, rows = source_episodes(source), source_episodes(output)
    if len(rows) != len(original_rows):
        raise ValueError("output does not contain every source episode")
    results = []
    total = 0
    for original_row, row in zip(original_rows, rows):
        episode = row["episode_index"]
        if episode != original_row["episode_index"]:
            raise ValueError("episode identities changed")
        original = read_episode(source, si, original_row)
        table = read_episode(output, oi, row)
        maps = np.load(output / f"meta/retime_source_indices/episode_{episode:03d}.npz")
        left, right = maps["left"], maps["right"]
        n = len(left)
        if n != len(table) or right.shape != left.shape:
            raise ValueError("source map and output lengths differ")
        for mapping in [left, right]:
            if (
                mapping.min() < 0
                or mapping.max() >= len(original)
                or np.any(np.diff(mapping) < 0)
            ):
                raise ValueError("invalid or non-monotone source mapping")
        for key in ["action", "observation.state"]:
            values = np.asarray(original[key].to_pylist())
            actual = np.asarray(table[key].to_pylist())
            if not (
                np.array_equal(actual[:, :7], values[left, :7])
                and np.array_equal(actual[:, 7:], values[right, 7:])
            ):
                raise ValueError(f"{episode}: {key} violates per-arm source mapping")
        if table["index"].to_pylist() != list(range(total, total + n)):
            raise ValueError("global data indices are not contiguous")
        if not np.allclose(
            table["timestamp"].to_numpy(), np.arange(n) / oi["fps"], atol=1e-5
        ):
            raise ValueError("timestamps do not follow output frame rate")
        if (
            table["retime.left_source_frame"].to_pylist() != left.tolist()
            or table["retime.right_source_frame"].to_pylist() != right.tolist()
        ):
            raise ValueError("embedded source frame columns differ from receipts")
        synthetic = maps["synthetic_hold"]
        if table["retime.synthetic_hold"].to_pylist() != synthetic.tolist():
            raise ValueError("synthetic hold labels differ from receipts")
        count = int(synthetic.sum())
        if count != round(2 * oi["fps"]) or not synthetic[-count:].all():
            raise ValueError("synthetic terminal hold is not exactly two seconds")
        if len(np.unique(left[-count:])) != 1 or len(np.unique(right[-count:])) != 1:
            raise ValueError("terminal hold moves a source arm")
        camera_reports = {}
        for camera, feature in oi["features"].items():
            if feature["dtype"] != "video":
                continue
            file = output / oi["video_path"].format(
                video_key=camera, chunk_index=0, file_index=episode
            )
            cap = cv2.VideoCapture(str(file))
            observed = (
                int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                cap.get(cv2.CAP_PROP_FPS),
            )
            cap.release()
            if observed != (n, feature["shape"][0], feature["shape"][1], oi["fps"]):
                raise ValueError(f"{episode}: video metadata mismatch for {camera}")
            if camera != "observation.images.top":
                mapping = {
                    "observation.images.left_wrist": left,
                    "observation.images.right_wrist": right,
                }[camera]
                src = source / si["video_path"].format(
                    video_key=camera, chunk_index=0, file_index=episode
                )
                camera_reports[camera] = wrist_pixel_error(src, file, mapping)
        receipt = json.loads(
            (output / f"meta/retime_receipts/episode_{episode:03d}.json").read_text()
        )
        if not receipt["plan"]["swept_edges_verified"]:
            raise ValueError("missing swept collision audit")
        if (
            not receipt["compositing"]
            .get("automatic_origin_audit", {})
            .get("passed", False)
        ):
            raise ValueError("missing rendered object-origin verification")
        results.append(dict(episode=episode, frames=n, wrists=camera_reports))
        total += n
    if total != oi["total_frames"]:
        raise ValueError("total frame metadata differs from episode sum")
    report = dict(passed=True, episodes=len(rows), frames=total, results=results)
    (output / "validation.json").write_text(json.dumps(report, indent=2))
    return report
