import json
import os
from pathlib import Path
import shutil
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from real_robot_data_retime.automatic_validation import validate_pending_episode


def test_real_episode_rejects_changed_arm_telemetry(tmp_path):
    sample_dir = os.environ.get("RETIME_SAMPLE_DIR")
    if sample_dir is None:
        pytest.skip("requires the remote generated drawer episode")
    root = Path(sample_dir).parent
    produced = root / "drawer-retimed"
    receipt = produced / "meta/retime_receipts/episode_001.json"
    if not receipt.exists() or "interaction" not in json.loads(receipt.read_text()):
        pytest.skip("current generated episode is not available")
    for relative in [
        "meta/retime_receipts/episode_001.json",
        "meta/retime_source_indices/episode_001.npz",
        "data/chunk-000/file-001.parquet",
    ]:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(produced / relative, destination)
    (tmp_path / "videos").symlink_to(produced / "videos", target_is_directory=True)
    assert (
        validate_pending_episode(root / "drawer-trimmed", tmp_path, 1)["episode"] == 1
    )
    data = tmp_path / "data/chunk-000/file-001.parquet"
    table = pq.read_table(data)
    key = "complementary_info.left_work.gripper_position_mm"
    values = table[key].to_pylist()
    values[0] += 1
    table = table.set_column(
        table.schema.get_field_index(key),
        key,
        pa.array(values, type=table.schema.field(key).type),
    )
    pq.write_table(table, data)
    with pytest.raises(ValueError, match="telemetry field"):
        validate_pending_episode(root / "drawer-trimmed", tmp_path, 1)
