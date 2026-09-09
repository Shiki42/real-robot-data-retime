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
        frames, timeline, segmentation, np.arange(15), np.arange(15), output, tmp_path
    )
    assert report["automatic_origin_audit"]["passed"]
    cap = cv2.VideoCapture(str(output))
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
    segmentation = dict(
        robots=np.packbits(np.zeros((20, 2, 64, 96), bool), axis=-1),
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
