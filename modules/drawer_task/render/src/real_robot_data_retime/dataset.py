from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset, Features, Sequence, Value

from real_robot_data_retime.stats import rewrite_dataset_numeric_stats
from real_robot_data_retime.video import remap_vertical_split_video, remap_video
from real_robot_data_retime.retime import (
    MotionHeuristic,
    RetimePlan,
    build_uniform_schedule_plan,
    detect_arm_segments,
    materialize_dual_arm,
)

CAMERAS = {
    "left_wrist": "observation.images.left_wrist",
    "right_wrist": "observation.images.right_wrist",
    "top": "observation.images.top",
}


@dataclass(frozen=True)
class PreparedEpisode:
    episode: int
    source_episode: int
    sample_index: int
    source_start: int
    source_length: int
    action: np.ndarray
    state: np.ndarray
    plan: RetimePlan
    detection: dict[str, object]
    timing: dict[str, object]


def episode_vector(table: pa.Table, episode: int, key: str) -> np.ndarray:
    mask = table["episode_index"].to_numpy() == episode
    column = table[key].combine_chunks()
    return column.values.to_numpy().reshape(len(table), 14)[mask]


def prepare_episode(
    episode: int,
    source_episode: int,
    sample_index: int,
    grid_index: int,
    grid_size: int,
    source_start: int,
    action: np.ndarray,
    state: np.ndarray,
    heuristic: MotionHeuristic,
    fps: float,
) -> PreparedEpisode:
    segments = detect_arm_segments(state, action, heuristic)
    plan = build_uniform_schedule_plan(segments, grid_index, grid_size)
    return PreparedEpisode(
        episode=episode,
        source_episode=source_episode,
        sample_index=sample_index,
        source_start=source_start,
        source_length=len(action),
        action=materialize_dual_arm(action, plan).astype(np.float32),
        state=materialize_dual_arm(state, plan).astype(np.float32),
        plan=plan,
        detection=segments.receipt(),
        timing=plan.timing_receipt(fps),
    )


def selected_grid_indices(grid_size: int, parity: str) -> list[int]:
    if parity not in {"even", "odd"}:
        raise ValueError(f"grid parity must be even or odd: {parity}")
    start = 0 if parity == "even" else 1
    return list(range(start, grid_size, 2))


def write_non_video_dataset(
    prepared: list[PreparedEpisode],
    source_info: dict,
    output: Path,
    repo_id: str,
    task: str,
) -> dict[str, object]:
    output.mkdir(parents=True)
    (output / "data/chunk-000").mkdir(parents=True)
    (output / "meta/episodes/chunk-000").mkdir(parents=True)

    actions = np.concatenate([item.action for item in prepared])
    states = np.concatenate([item.state for item in prepared])
    lengths = np.asarray([item.plan.length for item in prepared], dtype=np.int64)
    episode_index = np.repeat(np.arange(len(prepared), dtype=np.int64), lengths)
    frame_index = np.concatenate(
        [np.arange(length, dtype=np.int64) for length in lengths]
    )
    timestamp = frame_index.astype(np.float32) / float(source_info["fps"])
    left_idle = np.concatenate([~item.plan.left_active for item in prepared])
    right_idle = np.concatenate([~item.plan.right_active for item in prepared])
    overlap = np.concatenate(
        [item.plan.left_active & item.plan.right_active for item in prepared]
    )
    if np.any(left_idle & right_idle):
        raise ValueError("retime output contains a both-idle frame")
    features = Features(
        {
            "action": Sequence(Value("float32"), length=14),
            "episode_index": Value("int64"),
            "frame_index": Value("int64"),
            "index": Value("int64"),
            "observation.state": Sequence(Value("float32"), length=14),
            "retime.left_idle": Value("bool"),
            "retime.right_idle": Value("bool"),
            "retime.overlap": Value("bool"),
            "task_index": Value("int64"),
            "timestamp": Value("float32"),
        }
    )
    dataset = Dataset.from_dict(
        {
            "action": actions,
            "episode_index": episode_index,
            "frame_index": frame_index,
            "index": np.arange(len(actions), dtype=np.int64),
            "observation.state": states,
            "retime.left_idle": left_idle,
            "retime.right_idle": right_idle,
            "retime.overlap": overlap,
            "task_index": np.zeros(len(actions), dtype=np.int64),
            "timestamp": timestamp,
        },
        features=features,
    )
    dataset.to_parquet(output / "data/chunk-000/file-000.parquet")

    info = dict(source_info)
    kept_features = {
        key: source_info["features"][key]
        for key in (
            "action",
            "episode_index",
            "frame_index",
            "index",
            "observation.state",
            "task_index",
            "timestamp",
        )
    }
    for key in ("retime.left_idle", "retime.right_idle", "retime.overlap"):
        kept_features[key] = {"dtype": "bool", "shape": [1], "names": None}
    info.update(
        {
            "repo_id": repo_id,
            "total_episodes": len(prepared),
            "total_frames": len(actions),
            "total_tasks": 1,
            "splits": {"train": f"0:{len(prepared)}"},
            "features": kept_features,
        }
    )
    (output / "meta/info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False) + "\n"
    )
    pq.write_table(
        pa.table({"task_index": [0], "task": [task]}),
        output / "meta/tasks.parquet",
    )
    starts = np.cumsum(np.concatenate(([0], lengths[:-1])))
    pq.write_table(
        pa.table(
            {
                "episode_index": np.arange(len(prepared), dtype=np.int64),
                "tasks": [[task] for _ in prepared],
                "length": lengths,
                "data/chunk_index": np.zeros(len(prepared), dtype=np.int64),
                "data/file_index": np.zeros(len(prepared), dtype=np.int64),
                "dataset_from_index": starts,
                "dataset_to_index": starts + lengths,
            }
        ),
        output / "meta/episodes/chunk-000/file-000.parquet",
    )
    (output / "meta/stats.json").write_text("{}\n")
    return rewrite_dataset_numeric_stats(output)


def write_source_mappings(prepared: list[PreparedEpisode], output: Path) -> None:
    mapping_dir = output / "meta/retime_source_indices"
    mapping_dir.mkdir(parents=True, exist_ok=True)
    for item in prepared:
        np.savez_compressed(
            mapping_dir / f"episode_{item.episode:03d}.npz",
            left=item.plan.left_source_indices,
            right=item.plan.right_source_indices,
            left_active=item.plan.left_active,
            right_active=item.plan.right_active,
        )


def write_videos(
    source: Path,
    output: Path,
    source_info: dict,
    prepared: list[PreparedEpisode],
) -> None:
    global_left = np.concatenate(
        [item.plan.left_source_indices + item.source_start for item in prepared]
    )
    global_right = np.concatenate(
        [item.plan.right_source_indices + item.source_start for item in prepared]
    )
    fps = int(source_info["fps"])
    for short_name in ("left_wrist", "right_wrist"):
        camera = CAMERAS[short_name]
        feature = source_info["features"][camera]
        indices = global_left if short_name == "left_wrist" else global_right
        remap_video(
            source / source_info["video_path"].format(
                video_key=camera, chunk_index=0, file_index=0
            ),
            output / source_info["video_path"].format(
                video_key=camera, chunk_index=0, file_index=0
            ),
            indices,
            fps,
            int(feature["shape"][1]),
            int(feature["shape"][0]),
        )
        print(f"wrote {camera}: {len(indices)} frames", flush=True)

    camera = CAMERAS["top"]
    feature = source_info["features"][camera]
    remap_vertical_split_video(
        source / source_info["video_path"].format(
            video_key=camera, chunk_index=0, file_index=0
        ),
        output / source_info["video_path"].format(
            video_key=camera, chunk_index=0, file_index=0
        ),
        global_left,
        global_right,
        fps,
        int(feature["shape"][1]),
        int(feature["shape"][0]),
    )
    print(f"wrote {camera}: {len(global_left)} split-composite frames", flush=True)


def patch_video_metadata(
    output: Path,
    source_info: dict,
    prepared: list[PreparedEpisode],
) -> None:
    info_path = output / "meta/info.json"
    info = json.loads(info_path.read_text())
    for camera in CAMERAS.values():
        info["features"][camera] = source_info["features"][camera]
    info["video_path"] = source_info["video_path"]
    info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n")

    episodes_path = output / "meta/episodes/chunk-000/file-000.parquet"
    episodes = pq.read_table(episodes_path)
    lengths = np.asarray([item.plan.length for item in prepared], dtype=np.int64)
    starts = np.cumsum(np.concatenate(([0], lengths[:-1])))
    fps = float(source_info["fps"])
    episodes = episodes.append_column(
        "retime/source_indices_file",
        pa.array(
            [
                f"meta/retime_source_indices/episode_{item.episode:03d}.npz"
                for item in prepared
            ]
        ),
    )
    scalar_columns = {
        "retime/source_episode": pa.array(
            [item.source_episode for item in prepared], type=pa.int64()
        ),
        "retime/sample_index": pa.array(
            [item.sample_index for item in prepared], type=pa.int64()
        ),
        "retime/source_frame_count": pa.array(
            [item.source_length for item in prepared], type=pa.int64()
        ),
        "retime/grid_index": pa.array(
            [item.plan.grid_index for item in prepared], type=pa.int64()
        ),
        "retime/grid_size": pa.array(
            [item.plan.grid_size for item in prepared], type=pa.int64()
        ),
        "retime/normalized_position": pa.array(
            [item.timing["normalized_position"] for item in prepared]
        ),
        "retime/left_start_seconds": pa.array(
            [item.timing["left_start_seconds"] for item in prepared]
        ),
        "retime/right_start_seconds": pa.array(
            [item.timing["right_start_seconds"] for item in prepared]
        ),
        "retime/both_idle_frames": pa.array(
            [item.timing["both_idle_frames"] for item in prepared],
            type=pa.int64(),
        ),
    }
    for name, values in scalar_columns.items():
        episodes = episodes.append_column(name, values)
    for name, key in (
        ("retime/left_idle_seconds_json", "left_idle_seconds"),
        ("retime/right_idle_seconds_json", "right_idle_seconds"),
        ("retime/overlap_seconds_json", "overlap_seconds"),
    ):
        episodes = episodes.append_column(
            name,
            pa.array(
                [json.dumps(item.timing[key], separators=(",", ":")) for item in prepared]
            ),
        )
    episodes = episodes.append_column(
        "retime/timing_json",
        pa.array(
            [json.dumps(item.timing, separators=(",", ":")) for item in prepared]
        ),
    )
    episodes = episodes.append_column(
        "retime/main_camera_policy",
        pa.array(
            ["vertical_split_left_source_right_source"] * len(prepared)
        ),
    )
    episodes = episodes.append_column(
        "retime/detection_json",
        pa.array(
            [json.dumps(item.detection, separators=(",", ":")) for item in prepared]
        ),
    )
    for camera in CAMERAS.values():
        episodes = episodes.append_column(
            f"videos/{camera}/chunk_index",
            pa.array(np.zeros(len(prepared), dtype=np.int64)),
        )
        episodes = episodes.append_column(
            f"videos/{camera}/file_index",
            pa.array(np.zeros(len(prepared), dtype=np.int64)),
        )
        episodes = episodes.append_column(
            f"videos/{camera}/from_timestamp", pa.array(starts / fps)
        )
        episodes = episodes.append_column(
            f"videos/{camera}/to_timestamp", pa.array((starts + lengths) / fps)
        )
    pq.write_table(episodes, episodes_path)


def write_dataset_card(
    output: Path,
    source_repo: str,
    source_revision: str,
    episode_count: int,
    grid_parity: str,
    grid_size: int,
    source_episode_count: int,
) -> None:
    first = 0 if grid_parity == "even" else 1
    indices = f"{first}, {first + 2}, ..., {grid_size - 2 + first}"
    text = f"""---
tags:
- robotics
- lerobot
- piperx
- counterfactual-retiming
---

# PiperX Sort Letters - Uniform Timing Retime

Source: https://huggingface.co/datasets/{source_repo}/tree/{source_revision}

This dataset contains {episode_count} counterfactually retimed episodes selected
from a {grid_size}-point uniform schedule grid. It contains the {grid_parity}
global grid indices ({indices}). The full grid position is
u = grid_index / {grid_size}.

At u=0 the left arm completes immediately before the right arm starts. Moving
rightward on the grid continuously advances the right arm relative to the left
arm; the omitted u=1 endpoint would place the left arm immediately after the
right arm. No random delay is used.

No new both-arms-idle gap is ever inserted. Every output starts when the first
arm starts and ends when the last arm finishes. A waiting or finished arm holds
its boundary pose and wrist frame. Per-frame boolean labels are:
retime.left_idle, retime.right_idle, and retime.overlap. Per-episode metadata
records the exact frame and second intervals for each arm's idle periods and
for overlap. Both-idle frames are required to be zero.

The left wrist stream follows the left-arm source trajectory and the right wrist
stream follows the right-arm source trajectory. The top view is counterfactual:
its left half uses the left-arm source time and its right half uses the right-arm
source time. It is not a physically captured simultaneous world observation.

The pinned source contains {source_episode_count} episodes. The complete
four-samples-per-source grid therefore has {grid_size} points; this dataset
contains one alternating half with {episode_count} episodes.
"""
    (output / "README.md").write_text(text)


def payload_inventory(root: Path) -> list[dict[str, object]]:
    records = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or ".git" in path.parts:
            continue
        records.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return records


def git_revision(root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Counterfactually retime sequential 14-D dual-arm LeRobot data"
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source-repo", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--task", default="sort letters")
    parser.add_argument("--grid-parity", choices=("even", "odd"), required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    actual_revision = git_revision(args.dataset)
    if actual_revision != args.source_revision:
        raise ValueError(
            f"source revision mismatch: {actual_revision} != {args.source_revision}"
        )

    info = json.loads((args.dataset / "meta/info.json").read_text())
    episode_rows = pq.read_table(
        args.dataset / "meta/episodes/chunk-000/file-000.parquet"
    ).to_pylist()
    total_episodes = int(info["total_episodes"])
    if len(episode_rows) != total_episodes:
        raise ValueError("source info and episode metadata counts differ")
    table = pq.read_table(
        args.dataset / "data/chunk-000/file-000.parquet",
        columns=["episode_index", "action", "observation.state"],
    )
    if len(table) != int(info["total_frames"]):
        raise ValueError("source info and data frame counts differ")

    heuristic = MotionHeuristic()
    grid_size = total_episodes * 4
    grid_indices = selected_grid_indices(grid_size, args.grid_parity)
    prepared = []
    fps = float(info["fps"])
    for output_episode, grid_index in enumerate(grid_indices):
        source_episode = grid_index % total_episodes
        sample_index = grid_index // total_episodes
        item = prepare_episode(
            output_episode,
            source_episode,
            sample_index,
            grid_index,
            grid_size,
            int(episode_rows[source_episode]["dataset_from_index"]),
            episode_vector(table, source_episode, "action"),
            episode_vector(table, source_episode, "observation.state"),
            heuristic,
            fps,
        )
        prepared.append(item)
        print(
            f"episode={output_episode:03d} source={source_episode:02d} "
            f"sample={sample_index} grid={grid_index}/{grid_size} "
            f"frames={item.plan.length} timing={item.timing}",
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    numeric_stats = write_non_video_dataset(
        prepared, info, args.output, args.repo_id, args.task
    )
    write_source_mappings(prepared, args.output)
    write_videos(args.dataset, args.output, info, prepared)
    patch_video_metadata(args.output, info, prepared)
    write_dataset_card(
        args.output,
        args.source_repo,
        args.source_revision,
        len(prepared),
        args.grid_parity,
        grid_size,
        total_episodes,
    )

    manifest = {
        "schema": "real_robot_data_retime.dual_arm_uniform_schedule.v1",
        "source_repo": args.source_repo,
        "source_revision": args.source_revision,
        "producer_commit": git_revision(Path(__file__).resolve().parents[2]),
        "output_repo": args.repo_id,
        "fps": int(info["fps"]),
        "episodes": len(prepared),
        "source_episodes": total_episodes,
        "source_frames": int(info["total_frames"]),
        "retimed_frames": sum(item.plan.length for item in prepared),
        "sampling_policy": "uniform_four_samples_per_source_alternating_parity",
        "grid_size": grid_size,
        "grid_parity": args.grid_parity,
        "grid_indices": grid_indices,
        "random_delay": False,
        "both_idle_frames": 0,
        "wrist_camera_policy": "arm_specific_source_trajectory",
        "main_camera_policy": "vertical_split_left_source_right_source",
        "main_camera_split_x": int(
            info["features"][CAMERAS["top"]]["shape"][1] // 2
        ),
        "dropped_source_features": sorted(
            set(info["features"])
            - {"action", "observation.state", *CAMERAS.values()}
        ),
        "heuristic": asdict(heuristic),
        "numeric_stats": numeric_stats,
        "source_inventory": payload_inventory(args.dataset),
        "episode_schedules": [
            {
                "episode": item.episode,
                "source_episode": item.source_episode,
                "sample_index": item.sample_index,
                "detection": item.detection,
                "timing": item.timing,
            }
            for item in prepared
        ],
    }
    (args.output / "retime_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )


if __name__ == "__main__":
    main()
