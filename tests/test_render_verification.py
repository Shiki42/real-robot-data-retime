import numpy as np
from real_robot_data_retime.compositing.verification import OriginAudit


def test_duplicate_at_origin_is_rejected_even_with_successful_source_pickup():
    frames = np.full((15, 30, 40, 3), 120, np.uint8)
    frames[:5, 12:18, 18:24] = [0, 0, 220]
    masks = np.zeros((1, 15, 30, 40), bool)
    masks[0, :5, 12:18, 18:24] = True
    events = [dict(object_id=0, robot_id="left", grasp_frame=5)]
    audit = OriginAudit(frames, masks, events)
    for t in [11, 12, 13, 14]:
        wrong = frames[t].copy()
        wrong[12:18, 18:24] = [0, 0, 220]
        audit.observe(wrong, frames, [t, t], np.zeros((30, 40), bool))
    assert not audit.report()["passed"]
    assert audit.report()["objects"][0]["duplicate_observations"] == 4


def test_clean_origin_passes_and_occlusion_does_not_invent_observations():
    frames = np.full((15, 30, 40, 3), 120, np.uint8)
    frames[:5, 12:18, 18:24] = [0, 0, 220]
    masks = np.zeros((1, 15, 30, 40), bool)
    masks[0, :5, 12:18, 18:24] = True
    events = [dict(object_id=0, robot_id="left", grasp_frame=5)]
    audit = OriginAudit(frames, masks, events)
    for t in [11, 12, 13]:
        audit.observe(frames[t], frames, [t, t], np.zeros((30, 40), bool))
    assert audit.report()["passed"]
    audit.observe(frames[14], frames, [14, 14], np.ones((30, 40), bool))
    assert audit.report()["objects"][0]["clear_origin_observations"] == 3


def test_chromatic_object_absence_is_observable_on_similar_luminance_background():
    import cv2

    base = np.repeat(
        np.tile(np.arange(20, 100, 2, dtype=np.uint8), (30, 1))[:, :, None], 3, axis=2
    )
    frames = np.repeat(base[None], 15, axis=0)
    frames[:5, 12:17, 18:23] = [120, 60, 20]
    objects = np.zeros((1, 15, 30, 40), bool)
    objects[0, :5, 12:17, 18:23] = True
    audit = OriginAudit(
        frames, objects, [dict(object_id=0, robot_id="left", grasp_frame=5)]
    )
    roi = audit.items[0]["roi"]
    score = cv2.matchTemplate(
        cv2.cvtColor(base[roi], cv2.COLOR_BGR2GRAY),
        audit.items[0]["template"],
        cv2.TM_CCOEFF_NORMED,
    )[0, 0]
    assert score > 0.5
    for t in [11, 12, 13]:
        audit.observe(frames[t], frames, [t, t], np.zeros((30, 40), bool))
    assert audit.report()["passed"]
    assert audit.report()["objects"][0]["appearance_method"] == "chromatic_occupancy"
