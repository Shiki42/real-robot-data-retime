import numpy as np

from real_robot_data_retime.tasks.workpiece import dark_object_masks
from real_robot_data_retime.interaction.discovery import task_object_proposals


def test_shadow_joined_objects_keep_distinct_full_pixel_support():
    value = np.full((100, 200), 180, np.uint8)
    value[60:80, 65:80] = 40
    value[65:85, 95:110] = 50
    value[69:72, 80:95] = 85
    masks = dark_object_masks(value, 80, 1800)
    assert len(masks) == 3
    assert not masks[0].any()
    assert np.array_equal(masks[1] | masks[2], value < 95)
    assert not (masks[1] & masks[2]).any()
    assert masks[1][60:80, 65:80].all()
    assert masks[2][65:85, 95:110].all()
    frame = np.repeat(value[:, :, None], 3, axis=2)
    proposals = task_object_proposals(frame[None], "workpiece")
    assert len(proposals) == 2
    assert not (proposals[0]["mask"] & proposals[1]["mask"]).any()


def test_separate_objects_preserve_original_masks():
    value = np.full((100, 200), 180, np.uint8)
    value[60:80, 65:80] = 40
    value[65:85, 95:110] = 50
    masks = dark_object_masks(value, 80, 1800)
    assert len(masks) == 1
    assert np.array_equal(masks[0], value < 95)


def test_one_large_object_is_not_invented_into_two():
    value = np.full((100, 200), 180, np.uint8)
    value[60:80, 65:110] = 40
    masks = dark_object_masks(value, 80, 1800)
    assert len(masks) == 1
    assert np.array_equal(masks[0], value < 95)
