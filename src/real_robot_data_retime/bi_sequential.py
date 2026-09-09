"""Keep original demonstrations and append the right-then-left schedule endpoint."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from real_robot_data_retime.dataset import (
    CAMERAS,
    episode_vector,
    git_revision,
    patch_video_metadata,
    payload_inventory,
    prepare_episode,
    write_non_video_dataset,
    write_source_mappings,
    write_videos,
)
from real_robot_data_retime.retime import MotionHeuristic, detect_arm_segments
from real_robot_data_retime.stats import (
    feature_statistics,
    rewrite_dataset_numeric_stats,
)
from real_robot_data_retime.trim import aggregate_stats, image_statistics
from real_robot_data_retime.validate import validate_video_contract, vector, video_path

DATA = Path("data/chunk-000/file-000.parquet")
EPISODES = Path("meta/episodes/chunk-000/file-000.parquet")
CORE = (
    "action",
    "episode_index",
    "frame_index",
    "index",
    "observation.state",
    "task_index",
    "timestamp",
)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def replace_column(table, key, values):
    index = table.schema.get_field_index(key)
    return table.set_column(
        index,
        table.schema.field(key),
        pa.array(values, type=table.schema.field(key).type),
    )


def load_source(source):
    info = json.loads((source / "meta/info.json").read_text())
    rows = pq.read_table(source / EPISODES).to_pylist()
    table = pq.read_table(source / DATA, columns=list(CORE))
    if len(rows) != info["total_episodes"] or len(table) != info["total_frames"]:
        raise ValueError("source counts disagree")
    if set(table["task_index"].to_pylist()) != {0} or info["total_tasks"] != 1:
        raise ValueError("bi-sequential supports a single task with task_index=0")
    if sorted((source / "data").rglob("*.parquet")) != [source / DATA]:
        raise ValueError("bi-sequential requires one shared source data file")
    offset = 0
    for episode, row in enumerate(rows):
        length = int(row["length"])
        if (
            row["episode_index"],
            row["dataset_from_index"],
            row["dataset_to_index"],
            row["data/chunk_index"],
            row["data/file_index"],
        ) != (episode, offset, offset + length, 0, 0):
            raise ValueError("source episodes must be contiguous in shared file 0")
        part = table.slice(offset, length)
        np.testing.assert_array_equal(part["episode_index"].to_numpy(), episode)
        np.testing.assert_array_equal(part["frame_index"].to_numpy(), np.arange(length))
        for camera in CAMERAS.values():
            prefix = f"videos/{camera}"
            if (row[prefix + "/chunk_index"], row[prefix + "/file_index"]) != (0, 0):
                raise ValueError("source cameras must use shared file 0")
            np.testing.assert_allclose(
                [row[prefix + "/from_timestamp"], row[prefix + "/to_timestamp"]],
                [offset / info["fps"], (offset + length) / info["fps"]],
                atol=1e-6,
            )
        offset += length
    np.testing.assert_array_equal(table["index"].to_numpy(), np.arange(len(table)))
    return info, rows, table


def build_dataset(source, output, repo_id, source_repo, source_revision):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() or source in output.parents:
        raise ValueError("output must be new and outside source")
    if git_revision(source) != source_revision:
        raise ValueError("source revision mismatch")
    info, source_rows, source_table = load_source(source)
    n, original_frames = len(source_rows), len(source_table)
    heuristic = MotionHeuristic()
    prepared = [
        prepare_episode(
            i,
            i,
            0,
            1,
            1,
            row["dataset_from_index"],
            episode_vector(source_table, i, "action"),
            episode_vector(source_table, i, "observation.state"),
            heuristic,
            info["fps"],
        )
        for i, row in enumerate(source_rows)
    ]
    print(
        f"Prepared {n} right-first episodes: {sum(p.plan.length for p in prepared)} frames",
        flush=True,
    )
    # Existing writers produce the reverse half; its video clocks remain local to file 1.
    write_non_video_dataset(prepared, info, output, repo_id, "sort letters")
    write_source_mappings(prepared, output)
    write_videos(source, output, info, prepared)
    patch_video_metadata(output, info, prepared)
    reverse = pq.read_table(output / DATA)
    reverse = replace_column(
        reverse, "episode_index", reverse["episode_index"].to_numpy() + n
    )
    reverse = replace_column(
        reverse, "index", reverse["index"].to_numpy() + original_frames
    )
    original = source_table
    for key, arm in (("retime.left_idle", "left"), ("retime.right_idle", "right")):
        masks = []
        for item in prepared:
            segment = item.detection[arm]
            frames = np.arange(item.source_length)
            masks.append((frames < segment["start"]) | (frames >= segment["end"]))
        original = original.append_column(key, pa.array(np.concatenate(masks)))
    original = original.append_column(
        "retime.overlap", pa.array(np.zeros(original_frames, dtype=bool))
    )
    original = original.select(reverse.column_names).cast(reverse.schema)
    pq.write_table(original, output / DATA)
    pq.write_table(reverse, output / "data/chunk-000/file-001.parquet")

    reverse_rows = pq.read_table(output / EPISODES).to_pylist()
    pq.write_table(
        pa.Table.from_pandas(
            pd.DataFrame({"task_index": [0]}, index=pd.Index(["sort letters"]))
        ),
        output / "meta/tasks.parquet",
    )
    original_rows = []
    for i, (src, item, row) in enumerate(zip(source_rows, prepared, reverse_rows)):
        original_row = {key: None for key in row}
        original_row.update(
            {
                key: value
                for key, value in src.items()
                if key in row and not key.startswith("retime/")
            }
        )
        original_row.update(
            tasks=["sort letters"],
            **{
                "retime/source_episode": i,
                "retime/source_frame_count": item.source_length,
                "retime/order": "original_left_then_right",
                "retime/main_camera_policy": "original_unmodified",
                "retime/detection_json": json.dumps(item.detection),
            },
        )
        original_rows.append(original_row)
        row.update(
            episode_index=n + i,
            **{
                "data/file_index": 1,
                "dataset_from_index": original_frames + row["dataset_from_index"],
                "dataset_to_index": original_frames + row["dataset_to_index"],
                "retime/order": "right_then_left",
            },
        )
        mapping = output / row["retime/source_indices_file"]
        destination = mapping.with_name(f"episode_{n + i:03d}.npz")
        mapping.rename(destination)
        row["retime/source_indices_file"] = str(destination.relative_to(output))
        for camera in CAMERAS.values():
            row[f"videos/{camera}/file_index"] = 1
    rows = original_rows + reverse_rows
    for row in rows:
        row["meta/episodes/chunk_index"] = 0
        row["meta/episodes/file_index"] = 0
    # Preserve source video bytes, including all recorded boundary stillness.
    for camera in CAMERAS.values():
        video_path(output, camera).rename(video_path(output, camera, 1))
        shutil.copy2(video_path(source, camera), video_path(output, camera))
    merged_info = json.loads((output / "meta/info.json").read_text())
    merged_info.update(
        total_episodes=2 * n,
        total_frames=original_frames + len(reverse),
        splits={"train": f"0:{2 * n}"},
    )
    write_json(output / "meta/info.json", merged_info)
    for row in rows:
        table = original if row["episode_index"] < n else reverse
        offset = row["dataset_from_index"] - (
            0 if row["episode_index"] < n else original_frames
        )
        part = table.slice(offset, row["length"])
        for key in CORE + ("retime.left_idle", "retime.right_idle", "retime.overlap"):
            values = np.asarray(part[key].to_pylist()).reshape(len(part), -1)
            for stat, value in feature_statistics(values).items():
                row[f"stats/{key}/{stat}"] = value
    pq.write_table(pa.Table.from_pylist(rows), output / EPISODES)
    rewrite_dataset_numeric_stats(output, features=tuple(original.column_names))
    stats = json.loads((output / "meta/stats.json").read_text())
    for camera in CAMERAS.values():
        samples = [
            {camera: image_statistics(video_path(output, camera, index), count)}
            for index, count in enumerate((original_frames, len(reverse)))
        ]
        stats.update(aggregate_stats(samples))
        print(f"Decoded and verified both {camera} files", flush=True)
    write_json(output / "meta/stats.json", stats)
    manifest = {
        "schema": "real_robot_data_retime.bi_sequential.v1",
        "source_repo": source_repo,
        "source_revision": source_revision,
        "producer_commit": git_revision(Path(__file__).resolve().parents[2]),
        "output_repo": repo_id,
        "source_episodes": n,
        "episodes": 2 * n,
        "source_frames": original_frames,
        "reverse_frames": len(reverse),
        "total_frames": original_frames + len(reverse),
        "fps": info["fps"],
        "heuristic": asdict(heuristic),
        "reverse_normalized_position": 1.0,
        "original_policy": "preserve_core_columns_and_video_bytes",
        "task_text": "sort letters",
        "reverse_policy": "right_segment_then_left_segment_with_boundary_holds",
        "main_camera_policy": "original_first_half; vertical_split_counterfactual_second_half",
        "image_stats_sampling": "100 uniform frames per shared video file, 64x64 RGB",
        "dropped_source_features": sorted(
            set(info["features"]) - set(merged_info["features"])
        ),
    }
    write_json(output / "retime_manifest.json", manifest)
    (output / "README.md").write_text(f"""---
tags:
- robotics
- lerobot
- piperx
- counterfactual-retiming
---
# PiperX Sort Letters — Bi-sequential

{2 * n} episodes: {n} original left-then-right demonstrations followed by {n}
right-then-left counterfactuals, paired by source episode. The name says 53ep,
but the source actually contains {n} episodes.

Source: https://huggingface.co/datasets/{source_repo}/tree/{source_revision}

Episodes 0–{n - 1} preserve all {original_frames} source frames, action/state values,
frame indices, timestamps and original RGB video bytes. Episodes {n}–{2 * n - 1}
contain {len(reverse)} generated frames. Total: {original_frames + len(reverse)} frames
at {info["fps"]} FPS. Episode count doubles; frame count need not double.

The generated half uses the exact u=1 endpoint of the Random Retime scheduler:
the right arm completes before the left arm starts, with no overlap or inserted
both-idle gap. Each active segment runs forward at its recorded speed. Waiting
arms hold their boundary state/action and wrist image. The same motion heuristic
as Random Retime removes inactive boundaries from generated episodes; the original
half retains its recorded stillness. Idle labels describe these detected segments.

Wrist videos follow their respective arm clocks. The generated top view uses
the left image half at the left-arm source time and the right half at the
right-arm source time. This is edited imagery, not a newly recorded physical trial.

The training feature set matches Random Retime: action, observation.state,
three RGB cameras, task/index/timestamp fields and retime idle/overlap labels.
Acquisition diagnostic fields are omitted (listed in retime_manifest.json).
The single task text is normalized to `sort letters` for both halves.
Original RGB is stored in video file 000; generated RGB in file 001. Episode
metadata gives each file's local timestamps. Exact generated source indices are
in meta/retime_source_indices. VALIDATION_RECEIPT.json records validation and hashes.
""")
    return manifest


def validate_dataset(source, output):
    source, output = Path(source), Path(output)
    source_info, source_rows, source_table = load_source(source)
    info = json.loads((output / "meta/info.json").read_text())
    manifest = json.loads((output / "retime_manifest.json").read_text())
    rows = pq.read_table(output / EPISODES).to_pylist()
    original = pq.read_table(output / DATA)
    reverse = pq.read_table(output / "data/chunk-000/file-001.parquet")
    if manifest["source_revision"] != git_revision(source):
        raise ValueError("validation source revision mismatch")
    if info["fps"] != source_info["fps"]:
        raise ValueError("output FPS differs from source")
    tasks = pq.read_table(output / "meta/tasks.parquet").to_pandas()
    if tasks.index.tolist() != ["sort letters"] or tasks["task_index"].tolist() != [0]:
        raise ValueError("task table is not indexed by the task text")
    n, first_frames = len(source_rows), len(source_table)
    if (len(rows), info["total_episodes"], manifest["episodes"]) != (
        2 * n,
        2 * n,
        2 * n,
    ):
        raise ValueError("expected exactly one original and one reverse per source")
    if len(original) != first_frames or info["total_frames"] != len(original) + len(
        reverse
    ):
        raise ValueError("merged frame counts disagree")
    for key in CORE:
        np.testing.assert_array_equal(
            np.asarray(original[key].to_pylist()),
            np.asarray(source_table[key].to_pylist()),
        )
    offset, mappings = 0, []
    heuristic = MotionHeuristic(**manifest["heuristic"])
    for i in range(2 * n):
        row = rows[i]
        table = original if i < n else reverse
        local = offset if i < n else offset - first_frames
        part = table.slice(local, row["length"])
        if (
            row["episode_index"],
            row["dataset_from_index"],
            row["dataset_to_index"],
            row["retime/source_episode"],
            row["data/chunk_index"],
            row["data/file_index"],
        ) != (i, offset, offset + len(part), i % n, 0, int(i >= n)):
            raise ValueError(f"episode {i}: invalid merged metadata")
        np.testing.assert_array_equal(
            part["index"].to_numpy(), np.arange(offset, offset + len(part))
        )
        np.testing.assert_array_equal(
            part["frame_index"].to_numpy(), np.arange(len(part))
        )
        np.testing.assert_array_equal(part["episode_index"].to_numpy(), i)
        np.testing.assert_allclose(
            part["timestamp"].to_numpy(), np.arange(len(part)) / info["fps"], atol=1e-5
        )
        for camera in CAMERAS.values():
            prefix = f"videos/{camera}"
            if (row[prefix + "/chunk_index"], row[prefix + "/file_index"]) != (
                0,
                int(i >= n),
            ):
                raise ValueError(f"episode {i}: wrong video reference")
            np.testing.assert_allclose(
                [row[prefix + "/from_timestamp"], row[prefix + "/to_timestamp"]],
                [local / info["fps"], (local + len(part)) / info["fps"]],
                atol=1e-6,
            )
        if i >= n:
            src = source_rows[i - n]
            source_part = source_table.slice(src["dataset_from_index"], src["length"])
            segments = detect_arm_segments(
                vector(source_part, "observation.state"),
                vector(source_part, "action"),
                heuristic,
            )
            right_length, left_length = segments.right.length, segments.left.length
            if len(part) != right_length + left_length:
                raise ValueError(
                    "reverse output does not contain both complete segments"
                )
            with np.load(output / row["retime/source_indices_file"]) as receipt:
                left, right = receipt["left"], receipt["right"]
                # Independent endpoint construction, without calling the scheduler.
                expected_left = np.r_[
                    np.full(right_length, segments.left.start),
                    np.arange(segments.left.start, segments.left.end),
                ]
                expected_right = np.r_[
                    np.arange(segments.right.start, segments.right.end),
                    np.full(left_length, segments.right.end - 1),
                ]
                np.testing.assert_array_equal(left, expected_left)
                np.testing.assert_array_equal(right, expected_right)
                for arm, active in (
                    ("left", np.arange(len(part)) >= right_length),
                    ("right", np.arange(len(part)) < right_length),
                ):
                    np.testing.assert_array_equal(receipt[arm + "_active"], active)
                    np.testing.assert_array_equal(
                        part[f"retime.{arm}_idle"].to_numpy(), ~active
                    )
            np.testing.assert_array_equal(part["retime.overlap"].to_numpy(), False)
            for key in ("action", "observation.state"):
                values = vector(source_part, key)
                expected = np.concatenate((values[left, :7], values[right, 7:]), axis=1)
                np.testing.assert_array_equal(vector(part, key), expected)
            timing = json.loads(row["retime/timing_json"])
            if (
                timing["normalized_position"],
                timing["right_start_frame"],
                timing["left_start_frame"],
                timing["both_idle_frames"],
                timing["overlap_frames"],
            ) != (1.0, 0, right_length, 0, []):
                raise ValueError("wrong reverse execution timing")
            mappings.append(
                {
                    "left": left,
                    "right": right,
                    "source_start": src["dataset_from_index"],
                    "sample_indices": [right_length - 1, right_length],
                }
            )
        offset += len(part)
    if offset != info["total_frames"]:
        raise ValueError("episode coverage incomplete")
    source_files = {r["path"]: r for r in payload_inventory(source)}
    output_files = [
        item
        for item in payload_inventory(output)
        if item["path"] != "VALIDATION_RECEIPT.json"
    ]
    for camera in CAMERAS.values():
        relative = str(video_path(source, camera).relative_to(source))
        actual = next(r for r in output_files if r["path"] == relative)
        if actual["sha256"] != source_files[relative]["sha256"]:
            raise ValueError("original video bytes changed")
    video = validate_video_contract(
        source, output, mappings, len(reverse), 8.0, output_file_index=1
    )
    for key in ("action", "observation.state"):
        values = np.concatenate((vector(original, key), vector(reverse, key)))
        stats = json.loads((output / "meta/stats.json").read_text())[key]
        for name, expected in feature_statistics(values).items():
            np.testing.assert_allclose(stats[name], expected, atol=1e-8)
    report = {
        "status": "passed",
        "episodes": 2 * n,
        "original_episodes": n,
        "reverse_episodes": n,
        "original_frames": first_frames,
        "reverse_frames": len(reverse),
        "total_frames": offset,
        "original_core_columns": "exact",
        "original_video_bytes": "exact",
        "reverse_action_state_mapping": "exact",
        "reverse_overlap_frames": 0,
        "reverse_both_idle_frames": 0,
        "video": video,
        "source_revision": manifest["source_revision"],
        "producer_commit": manifest["producer_commit"],
        "inventory": output_files,
    }
    write_json(output / "VALIDATION_RECEIPT.json", report)
    return {key: value for key, value in report.items() if key != "inventory"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id")
    parser.add_argument("--source-repo")
    parser.add_argument("--source-revision")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if not args.validate_only:
        if not all((args.repo_id, args.source_repo, args.source_revision)):
            parser.error(
                "generation requires --repo-id, --source-repo and --source-revision"
            )
        build_dataset(
            args.source,
            args.output,
            args.repo_id,
            args.source_repo,
            args.source_revision,
        )
    print(json.dumps(validate_dataset(args.source, args.output), indent=2), flush=True)


if __name__ == "__main__":
    main()
