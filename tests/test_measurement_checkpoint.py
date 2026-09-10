import json

import numpy as np
import pytest

from real_robot_data_retime.interaction import measurements, pipeline, robot_discovery
from real_robot_data_retime.interaction.photometric_motion import PhotometricMotion
from real_robot_data_retime.tracking import points


def test_point_tracking_failure_keeps_complete_measurements_for_retry(
    monkeypatch, tmp_path
):
    n, h, w = 12, 16, 24
    frames = np.zeros((n, h, w, 3), np.uint8)
    transforms = np.repeat(np.eye(3)[None], n, axis=0)
    centers = np.full((n, 2, 2), 8.0)
    aperture = np.full((n, 2), 3.0)
    masks = np.ones((n, 2, h, w), bool)
    grippers = dict(
        centers=centers,
        apertures=aperture,
        masks=masks,
        robot_masks=np.packbits(masks, axis=-1),
    )
    tracks = [
        dict(
            centers=np.full((n, 2), 7.0),
            packed_masks=np.packbits(np.ones((n, h, w), bool), axis=-1),
        )
    ]
    inputs = dict(video_sha256="test-video", shape=list(frames.shape))
    (tmp_path / "measurements.json").write_text(
        json.dumps(dict(inputs=inputs, producer="original-automatic-producer"))
    )
    np.savez(tmp_path / "drawer_point_tracks.npz", stale=[1])
    monkeypatch.setattr(pipeline, "read_video", lambda *a, **k: (frames, 10))
    monkeypatch.setattr(
        pipeline, "stabilize", lambda *a: (frames, transforms, np.ones(n))
    )
    geometry = dict(
        centers=centers, apertures=aperture, masks=np.zeros((n, h, w), np.uint8)
    )
    monkeypatch.setattr(
        pipeline,
        "photometric_motion",
        lambda *a: PhotometricMotion(
            geometry,
            geometry,
            np.zeros((n, h, w), bool),
            np.zeros((n, h, w), bool),
            np.zeros((n, 3, 2)),
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "task_object_proposals",
        lambda *a: [dict(bbox=[1, 1, 4, 4], origin=[3, 3], area=16)],
    )
    monkeypatch.setattr(measurements, "inputs_fingerprint", lambda *a: inputs)
    monkeypatch.setattr(
        measurements,
        "load_hypotheses",
        lambda *a: (grippers, tracks, None, "original-automatic-producer"),
    )
    monkeypatch.setattr(
        robot_discovery,
        "robot_mask_audit",
        lambda *a, **k: dict(passed=True, arms=[dict(passed=True), dict(passed=True)]),
    )

    def interrupted(*args):
        raise RuntimeError("point tracking interrupted")

    monkeypatch.setattr(points, "track_points", interrupted)
    with pytest.raises(RuntimeError, match="point tracking interrupted"):
        pipeline.run(tmp_path / "input.mp4", tmp_path, "drawer")
    manifest = json.loads((tmp_path / "measurements.json").read_text())
    assert manifest["producer"] == "original-automatic-producer"
    with np.load(tmp_path / "tracks.npz") as saved:
        assert np.array_equal(saved["grippers"], centers)
        assert np.array_equal(saved["registration"], transforms)
    with np.load(tmp_path / "segmentation.npz") as saved:
        assert saved["objects"].shape[0] == 1
    assert not (tmp_path / "drawer_point_tracks.npz").exists()
    assert not (tmp_path / "report.json").exists()
