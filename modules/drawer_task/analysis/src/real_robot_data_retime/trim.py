from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from functools import lru_cache

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .stats import feature_statistics
from .video import remap_video


def analyze_episode(action, fps, joint_threshold=0.1, gripper_threshold=0.1,
                    tail_seconds=2.0):
    """Intervals are episode-local [start, stop); joint degrees, gripper mm."""
    values = np.asarray(action, dtype=np.float64)
    parameters = [fps, joint_threshold, gripper_threshold, tail_seconds]
    if not np.isfinite(parameters).all() or fps <= 0 or min(parameters[1:3]) <= 0 or tail_seconds < 0:
        raise ValueError("fps/thresholds must be positive and tail_seconds nonnegative")
    if values.ndim != 2 or values.shape[1] != 14 or not len(values) or not np.isfinite(values).all():
        raise ValueError("action must be a finite, nonempty N x 14 array")
    thresholds = np.tile([joint_threshold] * 6 + [gripper_threshold], 2)
    edges = np.abs(np.diff(values, axis=0)) * fps < thresholds
    masks = []
    for offset in (0, 7):
        still = np.all(edges[:, offset:offset + 7], axis=1)
        mask = np.ones(len(values), dtype=bool)
        mask[:-1] &= still
        mask[1:] &= still
        masks.append(mask)
    arm_edges = []
    for mask in masks:
        moving = np.flatnonzero(~mask)
        head = int(moving[0]) if len(moving) else len(mask)
        tail = len(mask) - int(moving[-1]) - 1 if len(moving) else len(mask)
        arm_edges.append({"head_frames": head, "tail_frames": tail})
    head = min(x["head_frames"] for x in arm_edges)
    tail = min(x["tail_frames"] for x in arm_edges)
    keep = math.ceil(tail_seconds * fps)
    # Entirely static episodes retain only the requested terminal hold (at least one frame).
    if head == len(values):
        start, stop = max(0, len(values) - max(1, keep)), len(values)
    else:
        start, stop = head, len(values) - max(0, tail - keep)
    return {
        "source_frames": len(values), "start": start, "stop": stop,
        "head_removed": start, "tail_removed": len(values) - stop,
        "all_static": head == len(values), "left": arm_edges[0], "right": arm_edges[1],
        "tail_keep_frames": keep,
        "tail_shortfall_frames": max(0, keep - min(tail, stop - start)),
    }


def source_episodes(root):
    files = sorted((root / "meta/episodes").rglob("*.parquet"))
    if not files:
        raise ValueError("expected LeRobot v3 meta/episodes parquet metadata")
    rows = [row for file in files for row in pq.read_table(file).to_pylist()]
    rows.sort(key=lambda row: row["episode_index"])
    if len({r["episode_index"] for r in rows}) != len(rows):
        raise ValueError("duplicate episode metadata")
    return rows


@lru_cache(maxsize=1)
def _read_data_file(path, mtime_ns):
    return pq.read_table(path)


def read_episode(root, info, row):
    path = root / info["data_path"].format(chunk_index=row["data/chunk_index"],
                                          file_index=row["data/file_index"])
    table = _read_data_file(path, path.stat().st_mtime_ns)
    table = table.filter(pa.compute.equal(table["episode_index"], row["episode_index"]))
    if len(table) != row["length"] or table["frame_index"].to_pylist() != list(range(len(table))):
        raise ValueError(f"invalid episode length/frame order: {row['episode_index']}")
    return table


def analyze_dataset(root, joint_threshold=0.1, gripper_threshold=0.1, tail_seconds=2.0):
    root = Path(root).resolve()
    info = json.loads((root / "meta/info.json").read_text())
    rows = source_episodes(root)
    if len(rows) != info["total_episodes"]:
        raise ValueError("episode count differs from info.json")
    episodes = []
    for row in rows:
        table = read_episode(root, info, row)
        result = analyze_episode(table["action"].to_pylist(), info["fps"],
                                 joint_threshold, gripper_threshold, tail_seconds)
        episodes.append({"episode_index": row["episode_index"], **result})
    if sum(e["source_frames"] for e in episodes) != info["total_frames"]:
        raise ValueError("frame count differs from info.json")
    return {"source": str(root), "fps": info["fps"], "joint_threshold_deg_s": joint_threshold,
            "gripper_threshold_mm_s": gripper_threshold, "tail_seconds": tail_seconds,
            "episodes": episodes}


def image_statistics(path, expected):
    """Sample at most 100 uniformly spaced frames, in RGB [0, 1]."""
    capture = cv2.VideoCapture(str(path))
    samples = set(np.linspace(0, expected - 1, min(100, expected), dtype=int).tolist())
    pixels = []
    count = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if count in samples:
            # Bound statistics memory; full video frame count is still verified.
            rgb = cv2.cvtColor(cv2.resize(frame, (64, 64)), cv2.COLOR_BGR2RGB)
            pixels.append(rgb.reshape(-1, 3).astype(np.float64) / 255)
        count += 1
    capture.release()
    if count != expected:
        raise ValueError(f"video frame count {count} != {expected}: {path}")
    stats = feature_statistics(np.concatenate(pixels))
    return {k: (v if k == "count" else np.asarray(v).reshape(3, 1, 1).tolist())
            for k, v in stats.items()} | {"count": [len(samples)]}


def aggregate_stats(items):
    result = {}
    for key in items[0]:
        stats = [item[key] for item in items]
        counts = np.array([s["count"][0] for s in stats])
        weights = counts / counts.sum()
        means = np.array([s["mean"] for s in stats])
        shape = (len(weights),) + (1,) * (means.ndim - 1)
        mean = (means * weights.reshape(shape)).sum(axis=0)
        variance = ((np.array([s["std"] for s in stats]) ** 2 +
                     (means - mean) ** 2) * weights.reshape(shape)).sum(axis=0)
        result[key] = {"min": np.min([s["min"] for s in stats], axis=0).tolist(),
                       "max": np.max([s["max"] for s in stats], axis=0).tolist(),
                       "mean": mean.tolist(), "std": np.sqrt(variance).tolist(),
                       "count": [int(counts.sum())]}
    return result


def trim_dataset(source, output, repo_id, joint_threshold=0.1, gripper_threshold=0.1,
                 tail_seconds=2.0):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() or source in output.parents:
        raise ValueError("output must be a new directory outside the source")
    report = analyze_dataset(source, joint_threshold, gripper_threshold, tail_seconds)
    info = json.loads((source / "meta/info.json").read_text())
    cameras = {k: v for k, v in info["features"].items()
               if v["dtype"] == "video" and not v.get("info", {}).get("is_depth_map", False)
               and v["shape"][-1] == 3}
    if not cameras:
        raise ValueError("no RGB video cameras found")
    output.mkdir(parents=True)
    (output / "data/chunk-000").mkdir(parents=True)
    (output / "meta/episodes/chunk-000").mkdir(parents=True)
    shutil.copy2(source / "meta/tasks.parquet", output / "meta/tasks.parquet")
    metadata, stats_list = [], []
    total = 0
    kept_columns = None
    for new_id, (row, interval) in enumerate(zip(source_episodes(source), report["episodes"])):
        table = read_episode(source, info, row)
        columns = [k for k in table.column_names if k in info["features"] and
                   info["features"][k]["dtype"] not in ("image", "video") and "depth" not in k]
        if kept_columns is not None and columns != kept_columns:
            raise ValueError("episode schemas differ")
        kept_columns = columns
        start, stop = interval["start"], interval["stop"]
        table = table.select(columns).slice(start, stop - start)
        length = len(table)
        for key, values in {
            "episode_index": np.full(length, new_id), "frame_index": np.arange(length),
            "index": np.arange(total, total + length),
            "timestamp": np.arange(length) / info["fps"],
        }.items():
            idx = table.schema.get_field_index(key)
            table = table.set_column(idx, key, pa.array(values, type=table.schema.field(key).type))
        pq.write_table(table, output / f"data/chunk-000/file-{new_id:03d}.parquet")
        ep = {"episode_index": new_id, "tasks": row["tasks"], "length": length,
              "data/chunk_index": 0, "data/file_index": new_id,
              "meta/episodes/chunk_index": 0, "meta/episodes/file_index": 0,
              "dataset_from_index": total, "dataset_to_index": total + length}
        stats = {}
        for key in columns:
            values = np.asarray(table[key].to_pylist(), dtype=float).reshape(length, -1)
            stats[key] = feature_statistics(values)
        for camera, feature in cameras.items():
            prefix = f"videos/{camera}"
            src = source / info["video_path"].format(video_key=camera,
                    chunk_index=row[prefix + "/chunk_index"], file_index=row[prefix + "/file_index"])
            first = round(row[prefix + "/from_timestamp"] * info["fps"])
            dst = output / f"videos/{camera}/chunk-000/file-{new_id:03d}.mp4"
            remap_video(src, dst, np.arange(first + start, first + stop), info["fps"],
                        feature["shape"][1], feature["shape"][0])
            stats[camera] = image_statistics(dst, length)
            ep.update({prefix + "/chunk_index": 0, prefix + "/file_index": new_id,
                       prefix + "/from_timestamp": 0.0, prefix + "/to_timestamp": length / info["fps"]})
        for key, values in stats.items():
            for name, value in values.items():
                ep[f"stats/{key}/{name}"] = value
        metadata.append(ep)
        stats_list.append(stats)
        total += length
        print(f"episode {row['episode_index']}: {interval['source_frames']} -> {length}", flush=True)
    info = dict(info)
    info["features"] = {k: v for k, v in info["features"].items() if k in kept_columns or k in cameras}
    info.pop("depth_storage_format", None)
    info.update(repo_id=repo_id, total_frames=total, total_episodes=len(metadata),
                splits={"train": f"0:{len(metadata)}"},
                data_path="data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
                video_path="videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4")
    pq.write_table(pa.Table.from_pylist(metadata), output / "meta/episodes/chunk-000/file-000.parquet")
    (output / "meta/info.json").write_text(json.dumps(info, indent=2) + "\n")
    (output / "meta/stats.json").write_text(json.dumps(aggregate_stats(stats_list), indent=2) + "\n")
    report["output_frames"] = total
    report["image_stats_sampling"] = "up to 100 uniform frames per episode, resized to 64x64 RGB"
    (output / "trim_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description="Analyze/crop PiperX episode static boundaries.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--report", type=Path, help="Analysis-only JSON output")
    parser.add_argument("--output", type=Path, help="New RGB dataset directory")
    parser.add_argument("--repo-id", help="Output dataset repo id (does not upload)")
    parser.add_argument("--joint-threshold", type=float, default=0.1, help="degrees/second")
    parser.add_argument("--gripper-threshold", type=float, default=0.1, help="mm/second")
    parser.add_argument("--tail-seconds", type=float, default=2.0)
    args = parser.parse_args()
    if not args.output and not args.report:
        parser.error("specify --report for analysis or --output and --repo-id for trimming")
    if args.output and not args.repo_id:
        parser.error("--output requires --repo-id")
    options = (args.joint_threshold, args.gripper_threshold, args.tail_seconds)
    report = (trim_dataset(args.dataset, args.output, args.repo_id, *options) if args.output
              else analyze_dataset(args.dataset, *options))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"episodes": len(report["episodes"]),
                      "head_removed": sum(e["head_removed"] for e in report["episodes"]),
                      "tail_removed": sum(e["tail_removed"] for e in report["episodes"])}))


if __name__ == "__main__":
    main()

