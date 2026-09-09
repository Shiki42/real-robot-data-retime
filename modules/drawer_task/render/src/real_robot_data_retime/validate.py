from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

CAMERAS = (
    "observation.images.left_wrist",
    "observation.images.right_wrist",
    "observation.images.top",
)


def vector(table: pq.Table, key: str) -> np.ndarray:
    column = table[key].combine_chunks()
    return column.values.to_numpy().reshape(len(table), 14)


def video_path(root: Path, camera: str) -> Path:
    return root / f"videos/{camera}/chunk-000/file-000.mp4"


def frame_at(capture: cv2.VideoCapture, index: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f"failed to decode video frame {index}")
    return frame


def mae(first: np.ndarray, second: np.ndarray) -> float:
    return float(
        np.mean(
            np.abs(first.astype(np.float32) - second.astype(np.float32))
        )
    )


def validate_video_contract(
    source: Path,
    output: Path,
    mappings: list[dict[str, np.ndarray | int]],
    total_frames: int,
    maximum_mae: float,
) -> dict[str, object]:
    source_caps = {
        camera: cv2.VideoCapture(str(video_path(source, camera)))
        for camera in CAMERAS
    }
    output_caps = {
        camera: cv2.VideoCapture(str(video_path(output, camera)))
        for camera in CAMERAS
    }
    for camera, capture in {**source_caps, **output_caps}.items():
        if not capture.isOpened():
            raise FileNotFoundError(f"failed to open {camera} video")

    observed_counts = {
        camera: int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        for camera, capture in output_caps.items()
    }
    if set(observed_counts.values()) != {total_frames}:
        raise ValueError(f"output video frame counts differ: {observed_counts}")

    errors = {camera: [] for camera in CAMERAS}
    output_start = 0
    for mapping in mappings:
        left = mapping["left"]
        right = mapping["right"]
        source_start = int(mapping["source_start"])
        sample = sorted({0, len(left) // 2, len(left) - 1})
        for local in sample:
            output_index = output_start + local
            left_index = source_start + int(left[local])
            right_index = source_start + int(right[local])
            output_left = frame_at(
                output_caps["observation.images.left_wrist"], output_index
            )
            source_left = frame_at(
                source_caps["observation.images.left_wrist"], left_index
            )
            errors["observation.images.left_wrist"].append(
                mae(output_left, source_left)
            )
            output_right = frame_at(
                output_caps["observation.images.right_wrist"], output_index
            )
            source_right = frame_at(
                source_caps["observation.images.right_wrist"], right_index
            )
            errors["observation.images.right_wrist"].append(
                mae(output_right, source_right)
            )
            output_top = frame_at(
                output_caps["observation.images.top"], output_index
            )
            source_top_left = frame_at(
                source_caps["observation.images.top"], left_index
            )
            source_top_right = frame_at(
                source_caps["observation.images.top"], right_index
            )
            split = output_top.shape[1] // 2
            errors["observation.images.top"].append(
                max(
                    mae(output_top[:, :split], source_top_left[:, :split]),
                    mae(output_top[:, split:], source_top_right[:, split:]),
                )
            )
        output_start += len(left)

    for capture in (*source_caps.values(), *output_caps.values()):
        capture.release()
    maxima = {camera: max(values) for camera, values in errors.items()}
    if max(maxima.values()) > maximum_mae:
        raise ValueError(f"video remap pixel MAE exceeds {maximum_mae}: {maxima}")
    return {
        "frame_counts": observed_counts,
        "sampled_frames_per_camera": sum(len(values) for values in errors.values())
        // len(errors),
        "maximum_pixel_mae": maxima,
        "maximum_allowed_pixel_mae": maximum_mae,
    }


def inventory(root: Path, excluded: Path) -> list[dict[str, object]]:
    records = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or ".git" in path.parts or path == excluded:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        records.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": digest,
            }
        )
    return records



def interval_seconds(mask: np.ndarray, fps: float) -> list[dict[str, float]]:
    values = np.asarray(mask, dtype=bool)
    changes = np.diff(
        np.concatenate(([False], values, [False])).astype(np.int8)
    )
    return [
        {"start": int(start) / fps, "end": int(end) / fps}
        for start, end in zip(
            np.flatnonzero(changes == 1),
            np.flatnonzero(changes == -1),
        )
    ]

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--maximum-video-mae", type=float, default=8.0)
    args = parser.parse_args()

    info = json.loads((args.dataset / "meta/info.json").read_text())
    manifest = json.loads((args.dataset / "retime_manifest.json").read_text())
    episodes = pq.read_table(
        args.dataset / "meta/episodes/chunk-000/file-000.parquet"
    ).to_pylist()
    output_table = pq.read_table(
        args.dataset / "data/chunk-000/file-000.parquet"
    )
    source_table = pq.read_table(
        args.source / "data/chunk-000/file-000.parquet",
        columns=["action", "observation.state"],
    )
    output_action = vector(output_table, "action")
    output_state = vector(output_table, "observation.state")
    output_left_idle = output_table["retime.left_idle"].to_numpy()
    output_right_idle = output_table["retime.right_idle"].to_numpy()
    output_overlap = output_table["retime.overlap"].to_numpy()
    source_action = vector(source_table, "action")
    source_state = vector(source_table, "observation.state")
    if not np.isfinite(output_action).all() or not np.isfinite(output_state).all():
        raise ValueError("retimed action/state contains non-finite values")

    episode_count = int(info["total_episodes"])
    total_frames = int(info["total_frames"])
    if episode_count != len(episodes) or total_frames != len(output_table):
        raise ValueError("info, episode metadata, and parquet counts differ")
    if manifest["random_delay"] is not False:
        raise ValueError("manifest unexpectedly declares random timing")
    if manifest["sampling_policy"] != "uniform_four_samples_per_source_alternating_parity":
        raise ValueError("manifest sampling policy is not the uniform timing grid")
    if int(manifest["both_idle_frames"]) != 0:
        raise ValueError("manifest declares a both-idle gap")

    mappings = []
    output_start = 0
    fps = float(info["fps"])
    source_episode_rows = pq.read_table(
        args.source / "meta/episodes/chunk-000/file-000.parquet"
    ).to_pylist()
    for episode, row in enumerate(episodes):
        mapping = np.load(
            args.dataset / row["retime/source_indices_file"]
        )
        left = mapping["left"]
        right = mapping["right"]
        left_active = mapping["left_active"].astype(bool)
        right_active = mapping["right_active"].astype(bool)
        length = int(row["length"])
        if np.any(~left_active & ~right_active):
            raise ValueError(f"episode {episode} contains a both-idle gap")
        if not all(
            len(values) == length
            for values in (left, right, left_active, right_active)
        ):
            raise ValueError(f"episode {episode} source mapping length mismatch")
        if left[0] == left[-1] or right[0] == right[-1]:
            raise ValueError(f"episode {episode} arm trajectory is static")
        for arm, indices in (("left", left), ("right", right)):
            steps = np.diff(indices)
            if indices.min() < 0 or np.any((steps < 0) | (steps > 1)):
                raise ValueError(f"episode {episode} invalid {arm} source mapping")
        source_episode = int(row["retime/source_episode"])
        source_start = int(
            source_episode_rows[source_episode]["dataset_from_index"]
        )
        source_end = int(source_episode_rows[source_episode]["dataset_to_index"])
        if max(left.max(), right.max()) >= source_end - source_start:
            raise ValueError(f"episode {episode} mapping leaves source episode")

        output_end = output_start + length
        np.testing.assert_array_equal(
            output_left_idle[output_start:output_end], ~left_active
        )
        np.testing.assert_array_equal(
            output_right_idle[output_start:output_end], ~right_active
        )
        np.testing.assert_array_equal(
            output_overlap[output_start:output_end], left_active & right_active
        )
        timing = json.loads(row["retime/timing_json"])
        expected_intervals = {
            "left_idle_seconds": interval_seconds(~left_active, fps),
            "right_idle_seconds": interval_seconds(~right_active, fps),
            "overlap_seconds": interval_seconds(left_active & right_active, fps),
        }
        for key, expected in expected_intervals.items():
            if timing[key] != expected:
                raise ValueError(f"episode {episode} {key} metadata mismatch")
            column = f"retime/{key}_json"
            if json.loads(row[column]) != expected:
                raise ValueError(f"episode {episode} {column} mismatch")
        if int(timing["both_idle_frames"]) != 0:
            raise ValueError(f"episode {episode} timing metadata has a gap")
        if int(row["retime/both_idle_frames"]) != 0:
            raise ValueError(f"episode {episode} metadata has a gap")
        source_left = source_start + left
        source_right = source_start + right
        np.testing.assert_array_equal(
            output_action[output_start:output_end, :7],
            source_action[source_left, :7],
        )
        np.testing.assert_array_equal(
            output_action[output_start:output_end, 7:],
            source_action[source_right, 7:],
        )
        np.testing.assert_array_equal(
            output_state[output_start:output_end, :7],
            source_state[source_left, :7],
        )
        np.testing.assert_array_equal(
            output_state[output_start:output_end, 7:],
            source_state[source_right, 7:],
        )
        mappings.append(
            {"left": left, "right": right, "source_start": source_start}
        )
        output_start = output_end
    if output_start != total_frames:
        raise ValueError("episode mappings do not cover every output frame")
    grid_indices = [int(row["retime/grid_index"]) for row in episodes]
    expected_start = 0 if manifest["grid_parity"] == "even" else 1
    expected_grid = list(range(expected_start, int(manifest["grid_size"]), 2))
    if grid_indices != expected_grid or grid_indices != manifest["grid_indices"]:
        raise ValueError("episode grid indices are not the requested alternating half")

    video = validate_video_contract(
        args.source,
        args.dataset,
        mappings,
        total_frames,
        args.maximum_video_mae,
    )
    files = inventory(args.dataset, args.report)
    report = {
        "status": "passed",
        "dataset": str(args.dataset),
        "source_revision": manifest["source_revision"],
        "producer_commit": manifest["producer_commit"],
        "episodes": episode_count,
        "frames": total_frames,
        "grid_parity": manifest["grid_parity"],
        "grid_indices": manifest["grid_indices"],
        "both_idle_frames": int(
            np.count_nonzero(output_left_idle & output_right_idle)
        ),
        "action_state_mapping": "exact",
        "idle_overlap_labels": "exact",
        "video": video,
        "inventory": {
            "files": len(files),
            "bytes": sum(int(item["bytes"]) for item in files),
            "records": files,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "inventory"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
