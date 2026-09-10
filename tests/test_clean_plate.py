import numpy as np

from real_robot_data_retime.background.clean_plate import (
    match_background_colors,
    real_patch,
    temporal_plate,
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
    corrected, _fits = match_background_colors(frames, excluded)
    assert (
        np.abs(corrected[0][~excluded[0]].astype(float) - base[~excluded[0]]).mean()
        < 1.1
    )


def test_foreground_in_color_ring_cannot_whiten_object_patch():
    from real_robot_data_retime.background.clean_plate import blend_scene_patch, dilate

    base = np.full((60, 60, 3), 100, np.uint8)
    patch = base.copy()
    region = np.zeros((60, 60), bool)
    region[20:40, 20:40] = True
    patch[region] = [180, 70, 30]
    foreground = dilate(region, 8) & ~region
    foreground[:, 43:] = False
    patch[foreground] = 0
    result = blend_scene_patch(
        base,
        patch,
        region,
        color_match=True,
        color_reference_mask=~foreground & ~region,
    )
    np.testing.assert_array_equal(result[30, 30], [180, 70, 30])


def test_local_color_match_requires_valid_background():
    import pytest

    from real_robot_data_retime.background.clean_plate import blend_scene_patch

    image = np.zeros((20, 20, 3), np.uint8)
    region = np.ones((20, 20), bool)
    with pytest.raises(ValueError, match="background-reference"):
        blend_scene_patch(image, image, region, color_match=True)
