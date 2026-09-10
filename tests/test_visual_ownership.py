import numpy as np


def test_visual_schedule_checks_carried_object_extent(monkeypatch):
    from real_robot_data_retime.timeline import visual
    from real_robot_data_retime.timeline.scheduler import Schedule

    n, h, w = 6, 32, 48
    robots = np.zeros((n, 2, h, w), bool)
    robots[:, 0, 10:15, 5:10] = True
    robots[:, 1, 10:15, 30:35] = True
    objects = np.zeros((1, n, h, w), bool)
    objects[:, :, 10:15, 20:32] = True
    event = {
        "object_id": 0,
        "robot_id": "left",
        "pickup_frame": 2,
        "release_frame": 4,
        "approach_start": 0,
        "retract_end": 5,
    }

    def inspect(nleft, nright, safe, **kwargs):
        assert safe(0, 0, 1, 1)  # Scene object does not freeze arm approach.
        assert not safe(2, 2, 3, 3)  # Held object crosses the other arm.
        assert safe(4, 4, 5, 5)  # Released object belongs to the scene.
        return Schedule(np.arange(n), np.arange(n), 0, 0)

    monkeypatch.setattr(visual, "schedule_sources", inspect)
    visual.plan_visual(
        {"task": "letters", "episodes": [event], "fps": 30},
        np.zeros((n, h, w, 3), np.uint8),
        {},
        {
            "robots": np.packbits(robots, axis=-1),
            "objects": np.packbits(objects, axis=-1),
        },
    )
