"""Select balanced original, reversed and concurrent episode groups."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from real_robot_data_retime.bi_sequential import (
    CORE,
    EPISODES,
    replace_column,
    write_json,
)
from real_robot_data_retime.dataset import CAMERAS, git_revision, payload_inventory
from real_robot_data_retime.stats import (
    feature_statistics,
    rewrite_dataset_numeric_stats,
)
from real_robot_data_retime.trim import aggregate_stats, image_statistics, read_episode
from real_robot_data_retime.validate import frame_at, mae, video_path
from real_robot_data_retime.video import remap_video

BI_REPO = "Shiki42/piperx-sortletter-0905-53ep-bi-sequential"
CONCURRENT_REPO = "Shiki42/piperx-sortletter-0905-53ep-retimed"
LABELS = ("retime.left_idle", "retime.right_idle", "retime.overlap")


def build_mixed(
    bisequential,
    concurrent,
    output,
    repo_id,
    bi_revision,
    concurrent_revision,
    count=36,
):
    if not 1 <= count <= 54:
        raise ValueError("count must be between 1 and 54")
    roots = [Path(bisequential).resolve(), Path(concurrent).resolve()]
    output = Path(output).resolve()
    if output.exists() or any(root in output.parents for root in roots):
        raise ValueError("output must be new and outside sources")
    groups = [
        (roots[0], BI_REPO, bi_revision, range(count), "left_then_right"),
        (roots[0], BI_REPO, bi_revision, range(54, 54 + count), "right_then_left"),
        (roots[1], CONCURRENT_REPO, concurrent_revision, range(count), "concurrent"),
    ]
    info = json.loads((roots[0] / "meta/info.json").read_text())
    (output / "data/chunk-000").mkdir(parents=True)
    (output / EPISODES.parent).mkdir(parents=True)
    (output / "meta/retime_source_indices").mkdir(parents=True)
    shutil.copy2(roots[0] / "meta/tasks.parquet", output / "meta/tasks.parquet")
    metadata, group_reports, image_stats = [], [], []
    total = 0
    for group, (source, source_repo, revision, indices, mode) in enumerate(groups):
        source_info = json.loads((source / "meta/info.json").read_text())
        if source_info["fps"] != info["fps"]:
            raise ValueError("source FPS differs")
        for key in (*CORE, *CAMERAS.values()):
            if source_info["features"][key] != info["features"][key]:
                raise ValueError(f"source feature mismatch: {key}")
        source_rows = pq.read_table(source / EPISODES).to_pylist()
        selected = [source_rows[i] for i in indices]
        pieces, source_pieces = [], []
        group_start = total
        local = 0
        for source_episode, src in zip(indices, selected):
            if src["episode_index"] != source_episode:
                raise ValueError("source episode metadata out of order")
            if mode != "concurrent" and src["retime/order"] != (
                "original_left_then_right" if group == 0 else "right_then_left"
            ):
                raise ValueError("source sequence order mismatch")
            if mode == "concurrent" and src["retime/zero_delay_frames"] != 0:
                raise ValueError("concurrent source must start both arms without delay")
            part = read_episode(source, source_info, src)
            if mode == "concurrent":
                detection = json.loads(src["retime/detection_json"])
                timeline = np.arange(len(part))
                active = {
                    arm: timeline < detection[arm]["end"] - detection[arm]["start"]
                    for arm in ("left", "right")
                }
                for arm in active:
                    part = part.append_column(
                        f"retime.{arm}_idle", pa.array(~active[arm])
                    )
                part = part.append_column(
                    "retime.overlap", pa.array(active["left"] & active["right"])
                )
            part = part.select([*CORE, *LABELS]).replace_schema_metadata(None)
            source_pieces.append(part)
            episode = len(metadata)
            part = replace_column(part, "episode_index", np.full(len(part), episode))
            part = replace_column(part, "index", np.arange(total, total + len(part)))
            pieces.append(part)
            row = {
                "episode_index": episode,
                "tasks": ["sort letters"],
                "length": len(part),
                "data/chunk_index": 0,
                "data/file_index": group,
                "meta/episodes/chunk_index": 0,
                "meta/episodes/file_index": 0,
                "dataset_from_index": total,
                "dataset_to_index": total + len(part),
                "mixed/mode": mode,
                "mixed/source_repo": source_repo,
                "mixed/source_revision": revision,
                "mixed/source_episode": source_episode,
                "mixed/source_metadata_json": json.dumps(
                    {k: v for k, v in src.items() if not k.startswith("stats/")}
                ),
                "retime/source_indices_file": None,
            }
            if src.get("retime/source_indices_file"):
                dest = Path(f"meta/retime_source_indices/episode_{episode:03d}.npz")
                shutil.copy2(source / src["retime/source_indices_file"], output / dest)
                row["retime/source_indices_file"] = str(dest)
            for camera in CAMERAS.values():
                prefix = f"videos/{camera}"
                row.update(
                    {
                        prefix + "/chunk_index": 0,
                        prefix + "/file_index": group,
                        prefix + "/from_timestamp": local / info["fps"],
                        prefix + "/to_timestamp": (local + len(part)) / info["fps"],
                    }
                )
            for key in part.column_names:
                values = np.asarray(part[key].to_pylist()).reshape(len(part), -1)
                row.update(
                    {
                        f"stats/{key}/{name}": value
                        for name, value in feature_statistics(values).items()
                    }
                )
            metadata.append(row)
            local += len(part)
            total += len(part)
        table = pa.concat_tables(pieces)
        data_path = output / f"data/chunk-000/file-{group:03d}.parquet"
        pq.write_table(table, data_path)
        readback = pq.read_table(data_path)
        original = pa.concat_tables(source_pieces)
        for key in (*CORE, *LABELS):
            expected = (
                table[key] if key in ("index", "episode_index") else original[key]
            )
            if not readback[key].equals(expected):
                raise ValueError(f"numeric readback mismatch: {mode}/{key}")
        camera_stats, video_report = {}, {}
        sample_indices = sorted(
            {
                i
                for row in metadata[-count:]
                for i in (
                    row["dataset_from_index"] - group_start,
                    row["dataset_from_index"] - group_start + row["length"] // 2,
                    row["dataset_to_index"] - group_start - 1,
                )
            }
        )
        for camera in CAMERAS.values():
            prefix = f"videos/{camera}"
            file_index = selected[0][prefix + "/file_index"]
            expected_start = 0
            for src in selected:
                if (
                    src[prefix + "/chunk_index"] != 0
                    or src[prefix + "/file_index"] != file_index
                ):
                    raise ValueError("selection must use one shared video file")
                np.testing.assert_allclose(
                    [src[prefix + "/from_timestamp"], src[prefix + "/to_timestamp"]],
                    [
                        expected_start / info["fps"],
                        (expected_start + src["length"]) / info["fps"],
                    ],
                    atol=1e-6,
                )
                expected_start += src["length"]
            src_video, dst_video = (
                video_path(source, camera, file_index),
                video_path(output, camera, group),
            )
            shape = info["features"][camera]["shape"]
            remap_video(
                src_video, dst_video, np.arange(local), info["fps"], shape[1], shape[0]
            )
            camera_stats[camera] = image_statistics(dst_video, local)
            src_cap, dst_cap = (
                cv2.VideoCapture(str(src_video)),
                cv2.VideoCapture(str(dst_video)),
            )
            try:
                maximum = max(
                    mae(frame_at(src_cap, i), frame_at(dst_cap, i))
                    for i in sample_indices
                )
            finally:
                src_cap.release()
                dst_cap.release()
            if maximum > 8:
                raise ValueError(f"video pixel mismatch: {mode}/{camera} MAE={maximum}")
            video_report[camera] = {
                "decoded_frames": local,
                "sampled_frames": len(sample_indices),
                "max_pixel_mae": maximum,
            }
            print(
                f"{mode}: verified {camera}, {local} frames, MAE={maximum:.3f}",
                flush=True,
            )
        image_stats.append(camera_stats)
        group_reports.append(
            {
                "mode": mode,
                "source_repo": source_repo,
                "source_revision": revision,
                "source_episodes": list(indices),
                "output_episodes": list(range(group * count, (group + 1) * count)),
                "frames": local,
                "numeric_mapping": "exact_except_reindexed_episode_and_global_index",
                "video": video_report,
            }
        )
    info.update(
        repo_id=repo_id,
        total_episodes=3 * count,
        total_frames=total,
        splits={"train": f"0:{3 * count}"},
    )
    write_json(output / "meta/info.json", info)
    pq.write_table(pa.Table.from_pylist(metadata), output / EPISODES)
    write_json(output / "meta/stats.json", aggregate_stats(image_stats))
    rewrite_dataset_numeric_stats(output, features=(*CORE, *LABELS))
    manifest = {
        "schema": "real_robot_data_retime.mixed.v1",
        "output_repo": repo_id,
        "producer_commit": git_revision(Path(__file__).resolve().parents[2]),
        "episodes": 3 * count,
        "frames": total,
        "fps": info["fps"],
        "groups": group_reports,
        "image_stats_sampling": "up to 100 uniform frames per group video, RGB 64x64",
        "concurrent_idle_labels": "derived from source detected segment durations; trajectories unchanged",
    }
    write_json(output / "mixed_manifest.json", manifest)
    (output / "README.md").write_text(f"""---
tags:
- robotics
- lerobot
- piperx
- counterfactual-retiming
---
# PiperX Sort Letters — Mixed

This dataset mixes **three execution orders**, with **{count} episodes per order**
and **{3 * count} episodes total** ({total:,} frames at {info["fps"]} FPS):

| Output episode indices (zero-based) | Execution order | Source episodes |
|---|---|---|
| 0–{count - 1} | Left arm, then right arm | bi-sequential 0–{count - 1} |
| {count}–{2 * count - 1} | Right arm, then left arm | bi-sequential 54–{54 + count - 1} |
| {2 * count}–{3 * count - 1} | Concurrent: both arms start together | retimed 0–{count - 1} |

## Source datasets

- [PiperX Sort Letters — Bi-sequential](https://huggingface.co/datasets/{BI_REPO})
  supplies the original left-then-right and generated right-then-left groups.
  Pinned revision: `{bi_revision}`.
- [PiperX Sort Letters — Retimed (Concurrent)](https://huggingface.co/datasets/{CONCURRENT_REPO})
  supplies the concurrent group. Pinned revision: `{concurrent_revision}`.

Each group selects the first {count} demonstrations of its execution order.
The groups use the same underlying first {count} source demonstrations, with three
execution schedules. Rows are stored in the group order shown above; they are not
shuffled. All episodes use the task text `sort letters`.

Action, state, per-episode frame indices and timestamps are preserved exactly
from the selected source episodes. Episode IDs and global row indices are rebuilt.
RGB videos are trimmed to the selected episodes and re-encoded, with unchanged
frame order and FPS. No additional retiming is applied. The generated right-first
and concurrent top views inherit their sources' left/right split compositing.

The data uses LeRobot v3 metadata, three RGB cameras and per-frame idle/overlap
labels. Concurrent labels are derived from its original detected arm durations.
Per-episode `mixed/source_repo`, `mixed/source_revision`, `mixed/source_episode`
and `mixed/mode` fields record provenance. Original retime mappings are copied
where available. `mixed_manifest.json` and `VALIDATION_RECEIPT.json` record
selection, numeric validation, decoded video counts and pixel checks.
""")
    # Check the complete merged index and provenance contracts after serialization.
    final_rows = pq.read_table(output / EPISODES).to_pylist()
    assert [r["episode_index"] for r in final_rows] == list(range(3 * count))
    assert sum(r["length"] for r in final_rows) == total
    assert [r["mixed/source_episode"] for r in final_rows] == list(range(count)) + list(
        range(54, 54 + count)
    ) + list(range(count))
    assert final_rows[-1]["dataset_to_index"] == total
    receipt = {"status": "passed", **manifest, "inventory": payload_inventory(output)}
    write_json(output / "VALIDATION_RECEIPT.json", receipt)
    return {k: receipt[k] for k in ["status", "episodes", "frames", "producer_commit"]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ["bisequential", "concurrent", "output"]:
        p.add_argument("--" + key, type=Path, required=True)
    for key in ["repo-id", "bi-revision", "concurrent-revision"]:
        p.add_argument("--" + key, required=True)
    p.add_argument("--count", type=int, default=36)
    print(json.dumps(build_mixed(**vars(p.parse_args())), indent=2), flush=True)


if __name__ == "__main__":
    main()
