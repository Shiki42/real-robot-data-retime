import numpy as np
import pytest

from real_robot_data_retime.segmentation.robotseg_backend import split_anchored_robots
from real_robot_data_retime.tracking.tapir import model_queries, sample_object_points


def test_dense_queries_preserve_membership_and_interior():
    masks = np.zeros((2, 30, 60), bool)
    masks[0, 4:20, 4:20] = True
    masks[1, 8:24, 35:50] = True
    queries, owners = sample_object_points([{"mask": m} for m in masks], 20)
    assert queries.shape == (40, 3)
    assert (queries[:, 0] == 0).all()
    for q, owner in zip(queries.astype(int), owners):
        _, y, x = q
        assert masks[owner, y - 1 : y + 2, x - 1 : x + 2].all()
    assert np.unique(queries, axis=0).shape[0] == 40


def test_non_square_tyx_scaling_does_not_swap_coordinates():
    queries = np.array([[7, 25, 150]], np.float32)
    np.testing.assert_allclose(model_queries(queries, (100, 200), 256), [[7, 64, 192]])
    np.testing.assert_array_equal(queries, [[7, 25, 150]])


def test_crossing_arms_and_detached_wrist_are_not_arbitrarily_assigned():
    mask = np.zeros((40, 100), bool)
    mask[3:8, :30] = True
    mask[12:17, 70:] = True
    mask[20:25, :] = True
    mask[30:35, 45:55] = True
    sides, unknown = split_anchored_robots(mask)
    assert sides[0].sum() == 150 and sides[1].sum() == 150
    assert unknown.sum() == 550
    assert not (sides[0] & sides[1]).any()
    np.testing.assert_array_equal(mask, sides.any(axis=0) | unknown)


def test_empty_object_is_not_replaced_with_background_points():
    with pytest.raises(ValueError, match="no trackable interior"):
        sample_object_points([{"mask": np.zeros((20, 20), bool)}])


def test_tapir_rgb_coordinates_and_occlusion_contract():
    torch = pytest.importorskip("torch")

    from real_robot_data_retime.tracking.tapir import BootsTapir

    class Model:
        def __call__(self, video, points):
            assert video.shape == (1, 2, 256, 256, 3)
            np.testing.assert_allclose(video[0, 0, 0, 0], [1, -1, -1])
            np.testing.assert_allclose(points, [[[0, 64, 192]]])
            return {
                "tracks": torch.tensor([[[[192.0, 64.0], [192.0, 64.0]]]]),
                "occlusion": torch.tensor([[[-100.0, 100.0]]]),
                "expected_dist": torch.tensor([[[-100.0, -100.0]]]),
            }

    tracker = BootsTapir.__new__(BootsTapir)
    tracker.device, tracker.resolution, tracker.model = "cpu", 256, Model()
    frames = np.zeros((2, 100, 200, 3), np.uint8)
    frames[..., 2] = 255
    xy, visible = tracker.track(frames, [[0, 25, 150]])
    np.testing.assert_allclose(xy[0, 0], [150, 25])
    assert visible[:, 0].tolist() == [True, False]
    assert np.isnan(xy[1, 0]).all()
    with pytest.raises(ValueError, match="outside decoded video"):
        tracker.track(frames, [[2, 25, 150]])


def test_robotseg_prompted_reverse_preserves_source_time_and_negative_points():
    from contextlib import contextmanager

    from real_robot_data_retime.segmentation.robotseg_backend import (
        RobotSegPromptedVideo,
    )

    torch = pytest.importorskip("torch")
    captured = {}

    class Model:
        def add_new_points_or_box(self, state, **kwargs):
            captured.update(kwargs)

        def propagate_in_video(self, state, robot):
            assert robot == "robot"
            for t in range(3):
                yield t, [0], torch.ones((1, 1, 4, 8))

    @contextmanager
    def state(frames):
        np.testing.assert_array_equal(frames[:, 0, 0, 0], [4, 3, 2])
        yield {}

    model = RobotSegPromptedVideo.__new__(RobotSegPromptedVideo)
    model.model, model._state = Model(), state
    frames = np.broadcast_to(np.arange(6)[:, None, None, None], (6, 4, 8, 3))
    proposal = {
        "bbox": [1, 1, 4, 2], "positive_points": [[2, 2]], "negative_points": [[6, 2]]
    }
    output = list(
        model.propagate(frames, [proposal], seed_frame=4, reverse=True, stop_frame=1)
    )
    assert [t for t, _ in output] == [4, 3, 2]
    assert all(masks.shape == (1, 4, 8) and masks.all() for _, masks in output)
    np.testing.assert_array_equal(captured["box"], [1, 1, 5, 3])
    np.testing.assert_array_equal(captured["labels"], [1, 0])
    assert captured["frame_idx"] == 0
    assert captured["robots"] == "robot"
