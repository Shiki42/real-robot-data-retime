import numpy as np
from real_robot_data_retime.tasks.workpiece import refine_deposition_releases


def sample():
    return {
        "fps": 30,
        "episodes": [
            {
                "robot_id": "left",
                "object_id": 0,
                "pickup_frame": 5,
                "release_frame": 40,
                "release_evidence": {
                    "verified": True,
                    "entry_frame": 15,
                    "clearance_frame": 40,
                    "release_frame": 40,
                },
            }
        ],
    }


def test_measured_opening_removes_return_motion_from_verified_placement():
    timeline = sample()
    state = np.zeros((50, 14))
    state[:, 6] = 10
    state[20:, 6] = 40
    refined, audit = refine_deposition_releases(timeline, state)
    assert refined["episodes"][0]["release_frame"] == 22
    assert timeline["episodes"][0]["release_frame"] == 40
    assert audit[0]["visual_release_frame"] == 40


def test_opening_cannot_replace_missing_visual_deposition():
    timeline = sample()
    timeline["episodes"][0]["release_evidence"]["verified"] = False
    state = np.zeros((50, 14))
    state[20:, 6] = 40
    refined, audit = refine_deposition_releases(timeline, state)
    assert not audit
    assert refined["episodes"][0]["release_frame"] == 40


def test_brief_or_previsit_opening_cannot_advance_release():
    state = np.zeros((50, 14))
    state[:10, 6] = 40
    state[20:22, 6] = 40
    assert not refine_deposition_releases(sample(), state)[1]
    state[12:15, 6] = [0, 10, 20]
    state[20:, 6] = 40
    assert not refine_deposition_releases(sample(), state)[1]


def test_opening_after_bin_clearance_cannot_advance_release():
    timeline = sample()
    timeline["episodes"][0]["release_evidence"]["clearance_frame"] = 18
    state = np.zeros((50, 14))
    state[20:, 6] = 40
    assert not refine_deposition_releases(timeline, state)[1]
