import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from real_robot_data_retime.automatic_dataset import finalize
from real_robot_data_retime.stats import feature_statistics
from scripts.process_workpiece_dataset import completed_analysis, finalize_packages


def test_incomplete_analysis_is_never_accepted(tmp_path):
    (tmp_path / "report.json").write_text(json.dumps({"success": True}))
    (tmp_path / "progress.json").write_text(json.dumps({"stage": "robot_discovery"}))
    assert not completed_analysis({"analysis": str(tmp_path)})


def test_unfinished_episode_prevents_final_export(tmp_path):
    done = tmp_path / "done"
    done.mkdir()
    (done / "process_status.json").write_text(
        json.dumps({"status": "processed", "package": "unused"})
    )
    manifest = {
        "episodes": [{"work_dir": str(done)}, {"work_dir": str(tmp_path / "missing")}]
    }
    output = tmp_path / "output"
    with pytest.raises(FileNotFoundError):
        finalize_packages(manifest, tmp_path, output)
    assert not output.exists()


def test_finalization_persists_correct_global_index_statistics(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "out"
    (source / "meta/episodes").mkdir(parents=True)
    pq.write_table(
        pa.table({"episode_index": [0, 1]}), source / "meta/episodes/episodes.parquet"
    )
    pq.write_table(
        pa.table({"task_index": [0], "task": ["workpiece"]}),
        source / "meta/tasks.parquet",
    )
    features = {
        key: {"dtype": "float64", "shape": [1]}
        for key in [
            "index",
            "episode_index",
            "retime.left_source_frame",
            "retime.right_source_frame",
        ]
    }
    features["retime.synthetic_hold"] = {"dtype": "bool", "shape": [1]}
    (source / "meta/info.json").write_text(
        json.dumps({"features": features, "fps": 30})
    )
    for ep in range(2):
        table = pa.table(
            {
                "index": [0, 1],
                "episode_index": [ep, ep],
                "retime.left_source_frame": [0.0, 1.0],
                "retime.right_source_frame": [0.0, 1.0],
                "retime.synthetic_hold": [False, False],
            }
        )
        p = output / f"data/chunk-000/file-{ep:03d}.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, p)
        stats = {
            k: feature_statistics(np.asarray(table[k]).reshape(-1, 1))
            for k in table.column_names
        }
        p = output / f"meta/retime_receipts/episode_{ep:03d}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(
                {
                    "episode_index": ep,
                    "length": 2,
                    "tasks": ["workpiece"],
                    "statistics": stats,
                }
            )
        )
    finalize(source, output, "test", episode_indices=[0, 1])
    assert pq.read_table(output / "data/chunk-000/file-001.parquet")[
        "index"
    ].to_pylist() == [2, 3]
    receipt = json.loads((output / "meta/retime_receipts/episode_001.json").read_text())
    assert receipt["statistics"]["index"]["min"] == [2]
    assert receipt["statistics"]["index"]["max"] == [3]


def test_running_episode_cannot_be_exported_as_a_rejection(tmp_path):
    work = tmp_path / "ep"
    work.mkdir()
    (work / "analysis_status.json").write_text(json.dumps({"status": "running"}))
    manifest = {"episodes": [{"episode": 4, "work_dir": str(work)}]}
    with pytest.raises(ValueError, match="not terminal"):
        finalize_packages(manifest, tmp_path, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_exported_action_tampering_is_detected():
    from scripts.validate_workpiece_dataset import validate_timing
    from real_robot_data_retime.automatic_dataset import remap_table

    values = np.arange(70, dtype=np.float32).reshape(5, 14)
    original = pa.table(
        {"observation.state": values.tolist(), "action": values.tolist()}
    )
    mapped = remap_table(original, np.arange(5.0), np.arange(5.0), 0)
    validate_timing(original, mapped)
    bad = values.copy()
    bad[2, 9] += 1
    mapped = mapped.set_column(
        mapped.schema.get_field_index("action"), "action", pa.array(bad.tolist())
    )
    with pytest.raises(AssertionError):
        validate_timing(original, mapped)
