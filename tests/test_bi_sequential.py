import json
import subprocess

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from real_robot_data_retime.bi_sequential import (
    CORE,
    DATA,
    EPISODES,
    build_dataset,
    validate_dataset,
)
from real_robot_data_retime.dataset import CAMERAS
from real_robot_data_retime.retime import (
    ArmSegment,
    EpisodeSegments,
    build_uniform_schedule_plan,
)
from real_robot_data_retime.video import _open_encoder, _write_frame, _close_encoder


@pytest.mark.parametrize("left_length,right_length", [(30, 70), (100, 31), (51, 51)])
def test_exact_right_first_endpoint(left_length, right_length):
    segments = EpisodeSegments(
        150,
        ArmSegment(10, 10 + left_length),
        ArmSegment(160, 160 + right_length),
        0,
        0,
        0,
        0,
    )
    plan = build_uniform_schedule_plan(segments, 216, 216)
    assert plan.right_start_frame == 0
    assert plan.left_start_frame == right_length
    assert plan.length == left_length + right_length
    assert not np.any(plan.left_active & plan.right_active)
    assert not np.any(~plan.left_active & ~plan.right_active)
    np.testing.assert_array_equal(
        plan.right_source_indices[:right_length], np.arange(160, 160 + right_length)
    )
    np.testing.assert_array_equal(
        plan.left_source_indices[right_length:], np.arange(10, 10 + left_length)
    )
    assert plan.timing_receipt(30)["normalized_position"] == 1
    with pytest.raises(ValueError):
        build_uniform_schedule_plan(segments, 217, 216)


def make_source(source):
    (source / DATA.parent).mkdir(parents=True)
    (source / EPISODES.parent).mkdir(parents=True)
    state = np.zeros((240, 14), dtype=np.float32)
    state[10:101, :7] = np.linspace(0, 70, 91)[:, None]
    state[101:, :7] = 70
    state[140:221, 7:] = np.linspace(0, 50, 81)[:, None]
    state[221:, 7:] = 50
    state = np.concatenate((state, state + 2))
    data = {
        "action": pa.array(state.tolist(), type=pa.list_(pa.float32(), 14)),
        "observation.state": pa.array(state.tolist(), type=pa.list_(pa.float32(), 14)),
        "episode_index": np.repeat(np.arange(2), 240),
        "frame_index": np.tile(np.arange(240), 2),
        "index": np.arange(480),
        "task_index": np.zeros(480, dtype=np.int64),
        "timestamp": np.tile(np.arange(240, dtype=np.float32) / 30, 2),
    }
    pq.write_table(pa.table(data), source / DATA)
    features = {
        key: {
            "dtype": (
                "float32"
                if key in ("action", "observation.state", "timestamp")
                else "int64"
            ),
            "shape": [14 if key in ("action", "observation.state") else 1],
            "names": None,
        }
        for key in CORE
    }
    rows = []
    for i in range(2):
        row = {
            "episode_index": i,
            "length": 240,
            "tasks": ["sort letters"],
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": i * 240,
            "dataset_to_index": (i + 1) * 240,
        }
        for camera in CAMERAS.values():
            row.update(
                {
                    f"videos/{camera}/chunk_index": 0,
                    f"videos/{camera}/file_index": 0,
                    f"videos/{camera}/from_timestamp": i * 8.0,
                    f"videos/{camera}/to_timestamp": (i + 1) * 8.0,
                }
            )
        rows.append(row)
    for camera in CAMERAS.values():
        features[camera] = {"dtype": "video", "shape": [24, 32, 3], "names": None}
        path = source / f"videos/{camera}/chunk-000/file-000.mp4"
        encoder = _open_encoder(path, 30, 32, 24)
        for t in range(480):
            frame = np.zeros((24, 32, 3), dtype=np.uint8)
            frame[:, :16] = (t % 200, 30, 70)
            frame[:, 16:] = (40, t % 200, 90)
            _write_frame(encoder, frame)
        _close_encoder(encoder, path)
    info = {
        "fps": 30,
        "total_episodes": 2,
        "total_frames": 480,
        "total_tasks": 1,
        "codebase_version": "v3.0",
        "features": features,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    }
    (source / "meta/info.json").write_text(json.dumps(info))
    pq.write_table(pa.Table.from_pylist(rows), source / EPISODES)
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "Source fixture",
        ],
        check=True,
    )
    return subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()


def test_full_merge_preserves_original_and_detects_corrupt_reverse(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    revision = make_source(source)
    result = build_dataset(source, output, "owner/output", "owner/source", revision)
    report = validate_dataset(source, output)
    assert result["episodes"] == report["episodes"] == 4
    assert report["original_frames"] == 480
    assert report["original_video_bytes"] == "exact"
    assert report["reverse_overlap_frames"] == 0
    table = pq.read_table(output / "data/chunk-000/file-001.parquet")
    index = table.schema.get_field_index("action")
    values = table["action"].to_pylist()
    values[0][7] += 1
    table = table.set_column(
        index,
        table.schema.field("action"),
        pa.array(values, type=table.schema.field("action").type),
    )
    pq.write_table(table, output / "data/chunk-000/file-001.parquet")
    with pytest.raises(AssertionError):
        validate_dataset(source, output)
