import numpy as np
import pytest

from real_robot_data_retime.video import compose_vertical_split
from real_robot_data_retime.retime import (
    MotionHeuristic,
    build_uniform_schedule_plan,
    detect_arm_segments,
    materialize_dual_arm,
)


def sequential_episode(frames: int = 240) -> tuple[np.ndarray, np.ndarray]:
    state = np.zeros((frames, 14), dtype=np.float64)
    left = np.linspace(0, 70, 91)
    right = np.linspace(0, 50, 81)
    state[10:101, :7] = left[:, None]
    state[101:, :7] = left[-1]
    state[140:221, 7:] = right[:, None]
    state[221:, 7:] = right[-1]
    return state, state.copy()


def test_detects_left_then_right_work_segments() -> None:
    state, action = sequential_episode()
    segments = detect_arm_segments(state, action)

    assert segments.left.start <= 10
    assert 101 <= segments.left.end < segments.right.start
    assert segments.right.start <= 140
    assert segments.right.end >= 221
    assert max(
        segments.state_right_before_split_ratio,
        segments.state_left_after_split_ratio,
    ) == 0.0


def test_schedule_boundaries_and_interior_have_no_both_idle_gap() -> None:
    state, action = sequential_episode()
    segments = detect_arm_segments(state, action)
    left_first = build_uniform_schedule_plan(segments, 0, 216)
    interior = build_uniform_schedule_plan(segments, 108, 216)
    right_first = build_uniform_schedule_plan(segments, 215, 216)

    assert left_first.left_start_frame == 0
    assert left_first.right_start_frame == segments.left.length
    assert np.any(interior.left_active & interior.right_active)
    assert min(interior.left_start_frame, interior.right_start_frame) == 0
    assert right_first.right_start_frame == 0
    for plan in (left_first, interior, right_first):
        assert not np.any(~plan.left_active & ~plan.right_active)
        output = materialize_dual_arm(action, plan)
        np.testing.assert_array_equal(
            output[0, :7], action[plan.left_source_indices[0], :7]
        )
        np.testing.assert_array_equal(
            output[0, 7:], action[plan.right_source_indices[0], 7:]
        )


def test_timing_receipt_labels_idle_and_overlap_in_frames_and_seconds() -> None:
    state, action = sequential_episode()
    segments = detect_arm_segments(state, action)
    plan = build_uniform_schedule_plan(segments, 54, 216)
    receipt = plan.timing_receipt(30.0)

    assert receipt["grid_index"] == 54
    assert receipt["normalized_position"] == 0.25
    assert receipt["both_idle_frames"] == 0
    assert receipt["right_idle_frames"]
    assert receipt["overlap_frames"]
    assert receipt["right_idle_seconds"][0]["start"] == 0.0


def test_rejects_episode_with_substantial_early_right_motion() -> None:
    state, action = sequential_episode()
    state[20:80, 7:] = np.arange(60)[:, None]
    action[:] = state

    with pytest.raises(ValueError, match="not sufficiently left-then-right decoupled"):
        detect_arm_segments(
            state,
            action,
            MotionHeuristic(maximum_cross_motion_ratio=0.05),
        )


def test_alternating_grid_halves_are_complete_and_disjoint() -> None:
    from real_robot_data_retime.dataset import selected_grid_indices

    even = selected_grid_indices(216, "even")
    odd = selected_grid_indices(216, "odd")
    assert len(even) == len(odd) == 108
    assert even[:3] == [0, 2, 4]
    assert odd[:3] == [1, 3, 5]
    assert sorted(even + odd) == list(range(216))


def test_vertical_split_uses_left_and_right_halves() -> None:
    left = np.zeros((2, 6, 3), dtype=np.uint8)
    right = np.full((2, 6, 3), 255, dtype=np.uint8)

    output = compose_vertical_split(left, right)

    assert np.all(output[:, :3] == 0)
    assert np.all(output[:, 3:] == 255)


def test_non_video_writer_emits_shared_v3_layout(tmp_path) -> None:
    from real_robot_data_retime.dataset import PreparedEpisode, write_non_video_dataset

    state, action = sequential_episode()
    segments = detect_arm_segments(state, action)
    plan = build_uniform_schedule_plan(segments, 54, 216)
    item = PreparedEpisode(
        episode=0,
        source_episode=0,
        sample_index=1,
        source_start=0,
        source_length=len(action),
        action=materialize_dual_arm(action, plan).astype(np.float32),
        state=materialize_dual_arm(state, plan).astype(np.float32),
        plan=plan,
        detection=segments.receipt(),
        timing=plan.timing_receipt(30.0),
    )
    feature = {
        "dtype": "float32",
        "shape": [14],
        "names": [f"joint_{index}" for index in range(14)],
    }
    receipt = write_non_video_dataset(
        [item],
        {
            "fps": 30,
            "robot_type": "bi_piperx",
            "features": {
                "action": feature,
                "observation.state": feature,
                **{
                    key: {"dtype": dtype, "shape": [1], "names": None}
                    for key, dtype in (
                        ("episode_index", "int64"),
                        ("frame_index", "int64"),
                        ("index", "int64"),
                        ("task_index", "int64"),
                        ("timestamp", "float32"),
                    )
                },
            },
        },
        tmp_path / "output",
        "owner/output",
        "sort letters",
    )

    assert receipt["frame_count"] == plan.length
