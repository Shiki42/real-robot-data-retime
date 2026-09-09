import cv2
import numpy as np
from real_robot_data_retime.tasks.drawer import release_confirmations
from real_robot_data_retime.interaction.neural_tracks import object_gripper_distances
from real_robot_data_retime.interaction.verification import validate_track_colors


def test_object_contact_uses_mask_boundary_not_centroid():
    grippers = np.zeros((1, 1, 32, 32), bool)
    grippers[0, 0, 10:15, 5:8] = True
    mask = np.zeros((1, 32, 32), bool)
    mask[0, 10:15, 8:18] = True
    track = dict(
        centers=np.array([[12.5, 12.0]]), packed_masks=np.packbits(mask, axis=-1)
    )
    assert object_gripper_distances(grippers, [track])[0, 0, 0] == 1


def test_release_requires_separated_colored_object_in_drawer():
    n, h, w = 45, 120, 200
    frames = np.full((n, h, w, 3), 200, np.uint8)
    red = cv2.cvtColor(np.uint8([[[0, 220, 200]]]), cv2.COLOR_HSV2BGR)[0, 0]
    blue = cv2.cvtColor(np.uint8([[[100, 200, 200]]]), cv2.COLOR_HSV2BGR)[0, 0]
    frames[:, 10:35, 110:170] = red
    masks = np.zeros((2, n, h, w), bool)
    centers = np.zeros((2, n, 2))
    for t in range(n):
        x, y = (60, 90) if t < 15 else (145, 65)
        masks[0, t, y : y + 8, x : x + 8] = True
        frames[t, y : y + 8, x : x + 8] = blue
        centers[0, t] = [x + 3.5, y + 3.5]
        masks[1, t, 20:28, 130:138] = True
        frames[t, 20:28, 130:138] = blue
        centers[1, t] = [133.5, 23.5]
    tracks = [
        dict(centers=centers[k], packed_masks=np.packbits(masks[k], axis=-1))
        for k in range(2)
    ]
    proposals = [dict(color=np.array([100, 200, 200])) for _ in range(2)]
    distance = np.full((2, n, 2), 12.0)
    distance[0, :25, 0] = 0
    result = release_confirmations(frames, tracks, proposals, distance, 10)
    assert result[0, 15, 0] == -1
    assert result[0, 20, 0] == 25
    assert result[0, 25, 0] == 25
    assert np.all(result[1, :, 0] == -1)


def test_wrong_surface_is_removed_and_occlusion_is_not_observed_motion():
    frames = np.full((3, 32, 32, 3), 200, np.uint8)
    mask = np.zeros((3, 32, 32), bool)
    mask[:, 10:14, 10:14] = True
    frames[0, 10:14, 10:14] = cv2.cvtColor(
        np.uint8([[[100, 200, 200]]]), cv2.COLOR_HSV2BGR
    )[0, 0]
    frames[1, 10:14, 10:14] = cv2.cvtColor(
        np.uint8([[[60, 200, 200]]]), cv2.COLOR_HSV2BGR
    )[0, 0]
    frames[2, 10:14, 10:14] = 0
    robots = np.zeros((3, 2, 32, 32), bool)
    robots[2, 0] = mask[2]
    track = dict(
        centers=np.full((3, 2), 11.5),
        packed_masks=np.packbits(mask, axis=-1),
        areas=np.full(3, 16.0),
    )
    issues, visible = validate_track_colors(
        frames,
        [dict(appearance_model="hue", color=np.array([100, 200, 200]))],
        [track],
        np.packbits(robots, axis=-1),
    )
    assert visible.tolist() == [[True, False, False]]
    assert issues.tolist() == [[False, True, False]]
    assert np.isnan(track["centers"][1:]).all()
    assert not track["packed_masks"][1].any()
    assert track["packed_masks"][2].any()
