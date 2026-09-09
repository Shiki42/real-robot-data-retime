import numpy as np
import pytest

from real_robot_data_retime.trim import analyze_episode, aggregate_stats
from real_robot_data_retime.stats import feature_statistics


def test_asymmetric_stop_preserves_two_seconds_for_later_arm():
    a = np.zeros((100, 14))
    a[10:31, 0] = np.arange(21)
    a[31:, 0] = 20
    a[20:51, 7] = np.arange(31)
    a[51:, 7] = 30
    r = analyze_episode(a, 10)
    assert (r["start"], r["stop"]) == (10, 71)
    assert r["left"]["tail_frames"] == 69
    assert r["right"]["tail_frames"] == 49
    assert r["tail_shortfall_frames"] == 0


def test_short_tail_no_padding():
    a = np.zeros((30, 14))
    a[:, 0] = np.minimum(np.arange(30), 25)
    r = analyze_episode(a, 10)
    assert r["stop"] == 30
    assert r["tail_removed"] == 0
    assert r["tail_shortfall_frames"] == 16


def test_gripper_movement_is_not_trimmed():
    a = np.zeros((50, 14))
    a[5:16, 6] = np.arange(11)
    a[16:, 6] = 10
    r = analyze_episode(a, 10, tail_seconds=0)
    assert (r["start"], r["stop"]) == (5, 16)


@pytest.mark.parametrize("n,tail,start", [(100, 2, 80), (5, 2, 0), (1, 0, 0)])
def test_all_static_retains_terminal_hold(n, tail, start):
    r = analyze_episode(np.zeros((n, 14)), 10, tail_seconds=tail)
    assert r["all_static"]
    assert (r["start"], r["stop"]) == (start, n)


def test_threshold_is_strict():
    a = np.zeros((2, 14))
    a[1, 0] = 0.1
    assert analyze_episode(a, 1)["head_removed"] == 0


@pytest.mark.parametrize("a", [np.zeros((0, 14)), np.zeros((5, 12)), np.full((2, 14), np.nan)])
def test_bad_action_fails(a):
    with pytest.raises(ValueError):
        analyze_episode(a, 30)


def test_weighted_stats_matches_concatenation():
    a, b = np.arange(12).reshape(6, 2), np.arange(20).reshape(10, 2)
    actual = aggregate_stats([{"x": feature_statistics(a)}, {"x": feature_statistics(b)}])["x"]
    expected = feature_statistics(np.concatenate([a, b]))
    for key in actual:
        np.testing.assert_allclose(actual[key], expected[key])


def test_shared_files_rgb_depth_and_multiple_tasks(tmp_path):
    import json
    import cv2
    import pyarrow as pa
    import pyarrow.parquet as pq
    from real_robot_data_retime.trim import trim_dataset

    source, output = tmp_path / "source", tmp_path / "output"
    (source / "data/chunk-000").mkdir(parents=True)
    (source / "meta/episodes/chunk-000").mkdir(parents=True)
    camera = "observation.images.top"
    video = source / f"videos/{camera}/chunk-000/file-000.mp4"
    video.parent.mkdir(parents=True)
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (32, 24))
    for i in range(100):
        writer.write(np.full((24, 32, 3), i * 2, dtype=np.uint8))
    writer.release()
    action = np.zeros((100, 14), dtype=np.float32)
    for offset in (0, 50):
        action[offset + 5:offset + 16, 0] = np.arange(11)
        action[offset + 16:offset + 50, 0] = 10
    data = {
        "action": action.tolist(), "observation.state": action.tolist(),
        "episode_index": [3] * 50 + [7] * 50, "frame_index": list(range(50)) * 2,
        "index": list(range(100)), "timestamp": (np.tile(np.arange(50), 2) / 10).tolist(),
        "task_index": [0] * 50 + [1] * 50, "extra": list(range(100)),
        "observation.depth.top": [b"depth"] * 100,
    }
    pq.write_table(pa.table(data), source / "data/chunk-000/file-000.parquet")
    features = {k: {"dtype": "float32", "shape": [14] if k in ("action", "observation.state") else [1]}
                for k in data}
    features["observation.depth.top"] = {"dtype": "image", "shape": [24, 32, 1]}
    features[camera] = {"dtype": "video", "shape": [24, 32, 3]}
    info = {"fps": 10, "total_episodes": 2, "total_frames": 100, "total_tasks": 2,
            "codebase_version": "v3.0", "features": features,
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"}
    (source / "meta/info.json").write_text(json.dumps(info))
    pq.write_table(pa.table({"task_index": [0, 1], "task": ["A", "B"]}), source / "meta/tasks.parquet")
    rows = []
    for i, eid in enumerate((3, 7)):
        rows.append({"episode_index": eid, "length": 50, "tasks": ["AB"[i]],
                     "data/chunk_index": 0, "data/file_index": 0,
                     f"videos/{camera}/chunk_index": 0, f"videos/{camera}/file_index": 0,
                     f"videos/{camera}/from_timestamp": i * 5.0})
    pq.write_table(pa.Table.from_pylist(rows), source / "meta/episodes/chunk-000/file-000.parquet")
    report = trim_dataset(source, output, "test/output")
    assert report["output_frames"] == 62
    for i in range(2):
        table = pq.read_table(output / f"data/chunk-000/file-{i:03d}.parquet")
        assert table["extra"].to_pylist() == list(range(i * 50 + 5, i * 50 + 36))
        assert table["frame_index"].to_pylist() == list(range(31))
        assert set(table["task_index"].to_pylist()) == {i}
        assert "observation.depth.top" not in table.column_names
        cap = cv2.VideoCapture(str(output / f"videos/{camera}/chunk-000/file-{i:03d}.mp4"))
        ok, first = cap.read()
        cap.release()
        assert ok and abs(first.mean() - (i * 50 + 5) * 2) < 8
    assert pq.read_table(output / "meta/tasks.parquet").to_pylist() == [
        {"task_index": 0, "task": "A"}, {"task_index": 1, "task": "B"}]
    with pytest.raises(ValueError):
        trim_dataset(source, output, "test/output")
