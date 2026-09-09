import cv2
import numpy as np
from real_robot_data_retime.compositing.layers import composite


def test_unselected_static_object_is_preserved_in_render(tmp_path):
    frames = np.full((15, 64, 96, 3), 120, np.uint8)
    objects = np.zeros((2, 15, 64, 96), bool)
    for t in range(15):
        x = 12 if t < 3 else 42
        frames[t, 12:18, x : x + 6] = [20, 30, 230]
        objects[0, t, 12:18, x : x + 6] = True
        frames[t, 40:48, 70:78] = [30, 220, 30]
        objects[1, t, 40:48, 70:78] = True
    robots = np.zeros((15, 2, 64, 96), bool)
    segmentation = dict(
        robots=np.packbits(robots, axis=-1), objects=np.packbits(objects, axis=-1)
    )
    timeline = dict(
        task="letters",
        fps=30,
        episodes=[
            dict(
                object_id=0,
                robot_id="left",
                pickup_frame=3,
                grasp_frame=3,
                release_frame=8,
                approach_start=0,
            )
        ],
    )
    output = tmp_path / "parallel.mp4"
    report = composite(
        frames,
        timeline,
        segmentation,
        np.arange(15),
        np.zeros(15, dtype=int),
        output,
        tmp_path,
    )
    assert report["automatic_origin_audit"]["passed"]
    cap = cv2.VideoCapture(str(output))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 14)
    ok, first = cap.read()
    cap.release()
    assert ok and first[41:47, 71:77, 1].mean() > 190
    assert first[41:47, 71:77, 0].mean() < 60


def test_released_cube_uses_closing_drawer_pixels(monkeypatch, tmp_path):
    import real_robot_data_retime.compositing.layers as layers

    frames = np.full((20, 64, 96, 3), 120, np.uint8)
    objects = np.zeros((1, 20, 64, 96), bool)
    for t in range(20):
        x = 12 if t < 3 else 55
        objects[0, t, 20:26, x : x + 6] = True
        if t < 12:
            frames[t, 20:26, x : x + 6] = [20, 200, 20]
    drawer = np.zeros((64, 96), bool)
    drawer[10:50, 40:90] = True
    monkeypatch.setattr(layers, "drawer_region", lambda *args: drawer)
    captured = []
    monkeypatch.setattr(
        layers, "write_video", lambda output, images, fps: captured.extend(images)
    )
    robots = np.zeros((20, 2, 64, 96), bool)
    robots[8:, 0, 20:26, 55:61] = True  # SAM can retain cube pixels in the arm mask.
    segmentation = dict(
        robots=np.packbits(robots, axis=-1),
        objects=np.packbits(objects, axis=-1),
    )
    timeline = dict(
        task="drawer",
        fps=30,
        drawer_motion=dict(open_frame=2, close_start=12),
        episodes=[
            dict(
                object_id=0,
                robot_id="left",
                pickup_frame=3,
                grasp_frame=3,
                release_frame=8,
                approach_start=0,
            )
        ],
    )
    composite(
        frames,
        timeline,
        segmentation,
        [7, 8, 10],
        [9, 10, 15],
        tmp_path / "out.mp4",
        tmp_path,
    )
    assert captured[0][22, 57, 1] > 180  # Held cube.
    assert captured[1][22, 57, 1] > 180  # Placed cube in open drawer.
    assert (
        np.max(np.abs(captured[2][22, 57].astype(int) - 120)) < 5
    )  # Closed drawer occludes it.


def test_initial_arm_leaves_no_feathered_ghost_when_it_moves(monkeypatch, tmp_path):
    import real_robot_data_retime.compositing.layers as layers

    n, h, w = 12, 64, 96
    frames = np.full((n, h, w, 3), 120, np.uint8)
    robots = np.zeros((n, 2, h, w), bool)
    for t in range(n):
        x = 10 if t < 4 else 50
        frames[t, 20:40, x : x + 20] = 0
        robots[t, 0, 20:40, x : x + 20] = True
    captured = []
    monkeypatch.setattr(
        layers, "write_video", lambda output, images, fps: captured.extend(images)
    )
    segmentation = dict(
        robots=np.packbits(robots, axis=-1),
        objects=np.zeros((0, n, h, (w + 7) // 8), np.uint8),
    )
    composite(
        frames,
        dict(task="letters", fps=30, episodes=[]),
        segmentation,
        [0, 8],
        [0, 7],
        tmp_path / "out.mp4",
        tmp_path,
    )
    assert np.min(captured[-1][20:40, 10:30]) == 120


def test_arm_boundary_fragment_uses_its_current_source_clock(monkeypatch, tmp_path):
    import real_robot_data_retime.compositing.layers as layers

    n, h, w = 12, 64, 96
    frames = np.full((n, h, w, 3), 120, np.uint8)
    robots = np.zeros((n, 2, h, w), bool)
    frames[:4, 40:55, :5] = 0
    robots[:4, 0, 40:55, :5] = True
    frames[10:, 50:54, :3] = 0  # The mostly off-screen arm returns, but SAM missed it.
    captured = []
    monkeypatch.setattr(
        layers, "write_video", lambda output, images, fps: captured.extend(images)
    )
    segmentation = dict(
        robots=np.packbits(robots, axis=-1),
        objects=np.zeros((0, n, h, (w + 7) // 8), np.uint8),
    )
    report = composite(
        frames,
        dict(task="letters", fps=30, episodes=[]),
        segmentation,
        [0, 8, 10],
        [0, 7, 8],
        tmp_path / "out.mp4",
        tmp_path,
    )
    assert report["paired_source_frames"] == 1
    assert np.array_equal(captured[0], frames[0])
    assert np.min(captured[1][40:55, :5]) == 120
    assert np.max(captured[2][50:54, :3]) == 0


def test_occluded_object_prediction_cannot_erase_departing_gripper(
    monkeypatch, tmp_path
):
    import real_robot_data_retime.compositing.layers as layers

    n, h, w = 20, 64, 96
    frames = np.full((n, h, w, 3), 120, np.uint8)
    objects = np.zeros((1, n, h, w), bool)
    robots = np.zeros((n, 2, h, w), bool)
    for t in range(n):
        x = 12 if t < 3 else 55
        frames[t, 20:26, x : x + 6] = [20, 200, 20]
        objects[0, t, 20:26, x : x + 6] = True
    frames[10, 45:55, 10:20] = 20
    robots[10, 0, 45:55, 10:20] = True
    objects[0, 10] = robots[10, 0]
    drawer = np.zeros((h, w), bool)
    drawer[10:40, 40:90] = True
    monkeypatch.setattr(layers, "drawer_region", lambda *a: drawer)
    captured = []
    monkeypatch.setattr(
        layers, "write_video", lambda output, images, fps: captured.extend(images)
    )
    timeline = dict(
        task="drawer",
        fps=30,
        drawer_motion=dict(open_frame=2, close_start=15),
        episodes=[
            dict(
                robot_id="left",
                object_id=0,
                pickup_frame=3,
                grasp_frame=3,
                release_frame=8,
                approach_start=0,
            )
        ],
    )
    segmentation = dict(
        robots=np.packbits(robots, axis=-1), objects=np.packbits(objects, axis=-1)
    )
    composite(frames, timeline, segmentation, [10], [5], tmp_path / "out.mp4", tmp_path)
    assert np.max(captured[0][45:55, 10:20]) < 30
