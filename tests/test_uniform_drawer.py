import numpy as np
import pytest
from real_robot_data_retime.timeline.uniform import (
    uniform_samples,
    stage_delays,
    validate_stage_schedule,
)


@pytest.mark.parametrize("count", [1, 2, 86, 87])
def test_every_source_has_two_half_interval_samples_and_global_uniform_grid(count):
    pairs = np.array([uniform_samples(i, count) for i in range(count)])
    np.testing.assert_allclose(np.diff(pairs, axis=1), 0.5)
    np.testing.assert_allclose(
        np.sort(pairs.ravel()), np.arange(2 * count) / (2 * count)
    )


def test_endpoint_delays_and_pair_spacing_at_video_resolution():
    assert stage_delays(157, 211, 0)["right_delay_frames"] == 157
    assert stage_delays(157, 211, 1)["left_delay_frames"] == 211
    for u, v in (uniform_samples(i, 87) for i in range(87)):
        a, b = stage_delays(157, 212, u), stage_delays(157, 212, v)
        assert (
            abs(a["relative_start_frames"] - b["relative_start_frames"] - 369 / 2) <= 1
        )
        assert abs(a["rounding_error_frames"]) <= 0.5
        assert abs(b["rounding_error_frames"]) <= 0.5


def test_validation_checks_intervals_not_only_endpoint_states():
    stages = dict(
        uniform=stage_delays(2, 3, 0.4),
        peak_source_frame=2,
        open_source_frame=3,
        close_source_frame=5,
        withdrawal_source_frame=4,
    )
    left = np.array([0, 1, 2, 2, 3, 4, 5])
    right = np.array([0, 1, 2, 3, 3, 3, 5])
    assert validate_stage_schedule(left, right, stages)["passed"]
    with pytest.raises(ValueError, match="insertion interval"):
        validate_stage_schedule(np.array([0, 1, 2, 3, 4, 5, 5]), right, stages)


def test_invalid_sampling_inputs_fail():
    with pytest.raises(ValueError):
        uniform_samples(2, 2)
    with pytest.raises(ValueError):
        stage_delays(1, 2, float("nan"))


def test_measured_closure_keeps_peak_before_late_visual_confirmation():
    from real_robot_data_retime.timeline.smooth import (
        held_grasp_interval,
        select_lift_peak,
    )

    state = np.zeros((80, 14))
    state[:, 6] = 70
    state[20:50, 6] = 35
    state[65:, 6] = 0  # Post-release closure must not be treated as holding.
    event = dict(approach_start=0, pickup_frame=18, grasp_frame=40, release_frame=75)
    start, end, aperture = held_grasp_interval(state, state, event, 30)
    assert (start, end) == (20, 50)
    tcp = np.zeros((80, 3))
    tcp[30, 2] = 0.25
    tcp[70, 2] = 0.4
    eligible = (np.arange(80) >= start) & (np.arange(80) < end)
    assert select_lift_peak(tcp, eligible, start, end - 1)[0] == 30
    assert aperture == 35.5


def test_no_recorded_closure_does_not_invent_a_grasp():
    from real_robot_data_retime.timeline.smooth import held_grasp_interval

    state = np.zeros((80, 14))
    state[:, 6] = 70
    with pytest.raises(ValueError, match="no settled"):
        held_grasp_interval(
            state, state, dict(approach_start=0, pickup_frame=10, release_frame=70), 30
        )


def test_two_exports_use_original_wrist_files_and_finalize_both_episodes(
    monkeypatch, tmp_path
):
    import json
    import pyarrow as pa
    import pyarrow.parquet as pq
    import real_robot_data_retime.automatic_dataset as dataset
    from real_robot_data_retime.stats import feature_statistics

    source, output = tmp_path / "source", tmp_path / "output"
    (source / "meta").mkdir(parents=True)
    cameras = [
        f"observation.images.{side}" for side in ("top", "left_wrist", "right_wrist")
    ]
    info = dict(
        fps=30,
        total_episodes=1,
        total_frames=5,
        features={key: dict(dtype="video", shape=[2, 2, 3]) for key in cameras},
        video_path="videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    )
    (source / "meta/info.json").write_text(json.dumps(info))
    pq.write_table(
        pa.table({"task_index": [0], "task": ["drawer"]}), source / "meta/tasks.parquet"
    )
    row = dict(episode_index=7, tasks=["drawer"])
    monkeypatch.setattr(dataset, "source_episodes", lambda _: [row])
    calls = []
    monkeypatch.setattr(
        dataset, "remap_video", lambda src, *args: calls.append(str(src))
    )
    monkeypatch.setattr(
        dataset, "image_statistics", lambda *args: feature_statistics(np.zeros((5, 3)))
    )
    values = np.arange(70).reshape(5, 14).tolist()
    table = pa.table(
        {
            "action": values,
            "observation.state": values,
            "episode_index": [7] * 5,
            "frame_index": range(5),
            "index": range(5),
            "timestamp": np.arange(5) / 30,
            "task_index": [0] * 5,
        }
    )
    for ep in range(2):
        receipt = dataset.write_retimed_episode(
            source,
            output,
            row,
            ep,
            table,
            np.arange(5),
            np.arange(5),
            dict(start=0),
            dict(synthetic_terminal_hold=dict(start_frame=5)),
            {},
            "identity",
            {},
        )
        assert receipt["source_episode_index"] == 7
    assert len(calls) == 4 and all(path.endswith("file-007.mp4") for path in calls)
    result = dataset.finalize(source, output, "owner/output", episode_indices=range(2))
    assert result == dict(episodes=2, frames=10)
    assert json.loads((output / "meta/info.json").read_text())["total_episodes"] == 2
    metadata = pq.read_table(
        output / "meta/episodes/chunk-000/file-000.parquet"
    ).to_pylist()
    assert [r["dataset_from_index"] for r in metadata] == [0, 5]
    assert pq.read_table(output / "data/chunk-000/file-001.parquet")[
        "index"
    ].to_pylist() == list(range(5, 10))
