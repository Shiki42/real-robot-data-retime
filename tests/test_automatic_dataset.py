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
