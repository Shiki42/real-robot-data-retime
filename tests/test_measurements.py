import json
import numpy as np
import pytest
from real_robot_data_retime.interaction.measurements import (
    inputs_fingerprint,
    load_hypotheses,
)


def test_cached_hypotheses_require_exact_input_and_rederive_gripper_geometry(tmp_path):
    video = tmp_path / "input.mp4"
    video.write_bytes(b"fixed source payload")
    frames = np.zeros((3, 30, 40, 3), np.uint8)
    transforms = np.repeat(np.eye(3)[None], 3, axis=0)
    proposals = [dict(bbox=[24, 19, 3, 3], origin=np.array([25.0, 20.0]), area=9)]
    expected = inputs_fingerprint(video, frames, proposals, transforms)
    robots = np.zeros((3, 2, 30, 40), bool)
    robots[:, 0, 10:22, :19] = True
    objects = np.zeros((1, 3, 30, 40), bool)
    objects[:, :, 19:22, 24:27] = True
    np.savez(
        tmp_path / "segmentation.npz",
        frame_shape=[30, 40],
        robots=np.packbits(robots, axis=-1),
        objects=np.packbits(objects, axis=-1),
    )
    centers = np.tile([25.0, 20.0], (1, 3, 1))
    np.savez(tmp_path / "tracks.npz", objects=centers, grippers=np.zeros((3, 2, 2)))
    (tmp_path / "measurements.json").write_text(
        json.dumps(dict(inputs=expected, producer="test automatic producer"))
    )
    grip, tracks, points, producer = load_hypotheses(tmp_path, expected, proposals)
    assert np.all(grip["centers"][:, 0, 0] > 10)
    assert np.array_equal(tracks[0]["centers"], centers[0])
    assert np.all(tracks[0]["areas"] == 9)
    video.write_bytes(b"changed source payload")
    with pytest.raises(ValueError, match="do not match"):
        load_hypotheses(
            tmp_path,
            inputs_fingerprint(video, frames, proposals, transforms),
            proposals,
        )
