import json
import numpy as np
import pytest
from real_robot_data_retime.interaction.robot_recovery import reference_geometry
from real_robot_data_retime.interaction import pipeline


def test_clean_reference_has_no_departed_arm_ghost_and_detects_reentry():
    n, h, w = 20, 64, 96
    frames = np.full((n, h, w, 3), 120, np.uint8)
    robots = np.zeros((n, 2, h, w), bool)
    for t in range(n):
        x = 0 if t < 5 or t >= 15 else 30
        frames[t, 30:50, x : x + 20] = 20
        if t < 15:
            robots[t, 0, 30:50, x : x + 20] = True
    old = np.zeros((n, h, w), np.uint8)
    old[:, 30:50, :20] = 1
    result = reference_geometry(frames, dict(masks=old), np.packbits(robots, axis=-1))
    assert result["masks"][10, 40, 10] == 0
    assert result["masks"][19, 40, 10] == 1


@pytest.mark.parametrize(
    "failed,objects,sides",
    [("drawer_open_close", False, (1,)), ("all_objects_identified", True, ()),
     ("drawer_event_order", True, ())],
)
def test_recovery_targets_failed_evidence_only(
    monkeypatch, tmp_path, failed, objects, sides
):
    calls = []

    def attempt(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            (tmp_path / "robot_mask_audit.json").write_text(
                json.dumps(dict(arms=[dict(passed=True), dict(passed=True)]))
            )
            return dict(
                success=False,
                task="drawer",
                validation_gates={failed: False},
                retries=[],
            )
        assert kwargs["retry_objects"] == objects
        assert kwargs["retry_robot_sides"] == sides
        return dict(
            success=True, task="drawer", validation_gates={failed: True}, retries=[]
        )

    monkeypatch.setattr(pipeline, "_run_once", attempt)
    assert pipeline.run("video.mp4", tmp_path, "drawer")["success"]
    assert len(calls) == 2


@pytest.mark.parametrize("surface", ["cabinet", "tray"])
def test_robot_mask_cannot_claim_scene_surfaces(surface):
    import cv2
    from real_robot_data_retime.interaction.robot_discovery import robot_mask_audit

    n, h, w = 20, 64, 96
    frames = np.full((n, h, w, 3), 180, np.uint8)
    red = cv2.cvtColor(np.uint8([[[0, 220, 180]]]), cv2.COLOR_HSV2BGR)[0, 0]
    frames[:, 5:20, 40:75] = red
    frames[:, 40:60, 70:] = 20
    robots = np.zeros((n, 2, h, w), bool)
    robots[:, 1, 40:60, 70:] = True
    if surface == "cabinet":
        robots[:, 1, 5:20, 40:75] = True
    else:
        robots[:, 1, 25:40, 40:75] = True
    geometry = dict(masks=np.zeros((n, h, w), np.uint8))
    geometry["masks"][:, 40:60, 70:] = 2
    reference = reference_geometry(frames, geometry, np.packbits(robots, axis=-1))
    assert reference["scene_contamination"][:, 1].all()
    assert not robot_mask_audit(np.packbits(robots, axis=-1), reference, 10)["arms"][1][
        "passed"
    ]


def test_forearm_motion_without_robot_hardware_is_not_an_arm_reference():
    n, h, w = 20, 64, 96
    frames = np.full((n, h, w, 3), 200, np.uint8)
    frames[10:, 30:45, :35] = [95, 125, 160]
    robots = np.zeros((n, 2, h, w), np.uint8)
    geometry = dict(masks=np.zeros((n, h, w), np.uint8))
    reference = reference_geometry(frames, geometry, np.packbits(robots, axis=-1))
    assert not reference["masks"][15].any()
