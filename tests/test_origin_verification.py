import numpy as np
from real_robot_data_retime.interaction.verification import validate_origin_departure


def test_track_swap_rejected_when_original_object_remains():
    rng = np.random.default_rng(11)
    f = np.full((30, 80, 100, 3), 200, dtype=np.uint8)
    patch = rng.integers(10, 120, (10, 10, 3), dtype=np.uint8)
    f[:, 35:45, 35:45] = patch
    p = dict(bbox=[35, 35, 10, 10], origin=[40, 40])
    g = np.tile([80, 70], (30, 1))
    result = validate_origin_departure(f, p, g, 2, 29)
    assert not result["verified"]
    assert result["reason"] == "object_still_at_origin"
    f[5:, 35:45, 35:45] = 200
    assert validate_origin_departure(f, p, g, 2, 29)["verified"]


def test_pickup_interval_preserves_occlusion_uncertainty():
    from real_robot_data_retime.interaction.verification import pickup_interval

    obj = np.tile([40.0, 40.0], (60, 1))
    obj[20:25, 0] = np.arange(5) * 5 + 45
    obj[25:35] = np.nan
    obj[35:, 0] = 100
    grip = obj.copy()
    p = dict(origin=[40, 40], bbox=[35, 35, 10, 10])
    result = pickup_interval(obj, grip, p, 35, 30)
    assert result["pickup_frame"] == 20
    assert result["uncertainty_frames"] == [20, 35]


def test_small_foreground_color_is_not_a_duplicate_object():
    from real_robot_data_retime.compositing.verification import OriginAudit

    frames = np.full((20, 30, 30, 3), 170, np.uint8)
    frames[0, 10:17, 10:17] = [0, 0, 180]
    objects = np.zeros((1, 20, 30, 30), bool)
    objects[0, 0, 10:17, 10:17] = True
    event = {"object_id": 0, "robot_id": "left", "grasp_frame": 0}
    audit = OriginAudit(frames, objects, [event])
    image = frames[7].copy()
    image[7, 10:14] = [0, 0, 180]
    foreground = np.zeros((30, 30), bool)
    foreground[7, 10:14] = True
    for t in [7, 8, 9]:
        audit.observe(image, frames, [t, t], foreground)
    assert audit.report()["passed"]
    duplicate = OriginAudit(frames, objects, [event])
    image = frames[0].copy()
    for t in [7, 8, 9]:
        duplicate.observe(image, frames, [t, t], np.zeros_like(foreground))
    assert not duplicate.report()["passed"]
