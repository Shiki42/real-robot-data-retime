import numpy as np
import pyarrow as pa
from real_robot_data_retime.automatic_dataset import remap_table


def test_arm_telemetry_and_vectors_follow_their_own_source_clock():
    values = np.arange(70, dtype=np.float32).reshape(5, 14)
    table = pa.table(
        {
            "action": values.tolist(),
            "observation.state": (values + 100).tolist(),
            "episode_index": [0] * 5,
            "frame_index": range(5),
            "index": range(5),
            "timestamp": np.arange(5) / 30,
            "task_index": [0] * 5,
            "complementary_info.left_work.gripper_position_mm": [1, 2, 3, 4, 5],
            "complementary_info.right_work.gripper_position_mm": [6, 7, 8, 9, 10],
            "complementary_info.rgb_device_timestamp_ns.left_wrist": [
                100,
                200,
                300,
                400,
                500,
            ],
            "complementary_info.rgb_device_timestamp_ns.right_wrist": [
                101,
                201,
                301,
                401,
                501,
            ],
            "complementary_info.rgb_device_timestamp_ns.top": [90, 190, 290, 390, 490],
        }
    )
    left = np.array([0, 1, 2, 2, 4])
    right = np.array([0, 0, 1, 2, 3])
    result = remap_table(table, left, right, 3)
    actual = np.asarray(result["action"].to_pylist())
    assert np.array_equal(actual[:, :7], values[left, :7])
    assert np.array_equal(actual[:, 7:], values[right, 7:])
    assert result["complementary_info.left_work.gripper_position_mm"].to_pylist() == [
        1,
        2,
        3,
        3,
        5,
    ]
    assert result["complementary_info.right_work.gripper_position_mm"].to_pylist() == [
        6,
        6,
        7,
        8,
        9,
    ]
    assert result[
        "complementary_info.rgb_device_timestamp_ns.left_wrist"
    ].to_pylist() == [100, 200, 300, 300, 500]
    assert result[
        "complementary_info.rgb_device_timestamp_ns.right_wrist"
    ].to_pylist() == [101, 101, 201, 301, 401]
    assert "complementary_info.rgb_device_timestamp_ns.top" not in result.column_names


def test_failed_rerender_preserves_published_video_and_receipt(monkeypatch, tmp_path):
    import json
    import pytest
    import real_robot_data_retime.automatic_dataset as dataset

    source, raw, output, work = [
        tmp_path / name for name in ("source", "raw", "out", "work")
    ]
    debug = work / "episode_000"
    debug.mkdir(parents=True)
    (debug / "analysis_identity.txt").write_text("identity")
    (debug / "interaction_timeline.json").write_text("{}")
    receipt_path = output / "meta/retime_receipts/episode_000.json"
    receipt_path.parent.mkdir(parents=True)
    original = json.dumps(dict(analysis_identity="identity", trim={}, statistics={}))
    receipt_path.write_text(original)
    maps = output / "meta/retime_source_indices/episode_000.npz"
    maps.parent.mkdir(parents=True)
    np.savez(maps, left=[0], right=[0])
    relative = "videos/observation.images.top/chunk-000/file-000.mp4"
    video = output / relative
    video.parent.mkdir(parents=True)
    video.write_bytes(b"old valid video")
    monkeypatch.setattr(dataset, "source_episodes", lambda _: [dict(episode_index=0)])

    def render(*args):
        stage = args[2] / relative
        stage.parent.mkdir(parents=True)
        stage.write_bytes(b"new video")
        return dict(output=relative)

    def fail(*args):
        raise ValueError("invalid frame count")

    monkeypatch.setattr(dataset, "render_main", render)
    monkeypatch.setattr(dataset, "image_statistics", fail)
    with pytest.raises(ValueError, match="invalid frame count"):
        dataset.rerender_episode(source, raw, output, work, 0)
    assert video.read_bytes() == b"old valid video"
    assert receipt_path.read_text() == original
