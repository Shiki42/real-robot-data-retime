import numpy as np
from real_robot_data_retime.background.clean_plate import (
    temporal_plate,
    real_patch,
    match_background_colors,
)


def test_temporal_plate_removes_objects_even_when_majority_of_frames_occluded():
    frames = np.full((9, 20, 24, 3), 100, np.uint8)
    excluded = np.zeros((9, 20, 24), bool)
    frames[:8, 5:12, 6:14] = [0, 0, 200]
    excluded[:8, 5:12, 6:14] = True
    plate, coverage = temporal_plate(frames, excluded)
    assert np.all(plate == 100)
    assert np.all(coverage[5:12, 6:14] == 1)


def test_origin_reconstruction_never_reintroduces_pre_pickup_object():
    frames = np.full((12, 20, 24, 3), 100, np.uint8)
    frames[:5, 5:12, 6:14] = [0, 0, 200]
    excluded = np.zeros((12, 20, 24), bool)
    excluded[5:8, 5:12, 6:14] = True
    frames[5:8, 5:12, 6:14] = 0
    region = np.zeros((20, 24), bool)
    region[5:12, 6:14] = True
    patch, missing = real_patch(
        frames,
        5,
        region,
        excluded,
        np.full((20, 24, 3), 70, np.uint8),
        np.arange(5, 12),
    )
    assert np.all(patch[region] == 100)
    assert missing == 0


def test_exposure_registration_ignores_moving_foreground():
    gradient = np.tile(np.arange(30, 190, dtype=np.uint8), (40, 1))
    base = np.repeat(gradient[:, :, None], 3, axis=2)
    frames = np.stack([base + 20, base])
    frames[0, 10:30, 10:40] = [0, 0, 240]
    excluded = np.zeros(frames.shape[:3], bool)
    excluded[0, 10:30, 10:40] = True
    corrected, fits = match_background_colors(frames, excluded)
    assert (
        np.abs(corrected[0][~excluded[0]].astype(float) - base[~excluded[0]]).mean()
        < 1.1
    )
