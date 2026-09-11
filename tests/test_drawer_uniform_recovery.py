import numpy as np
import pytest
from real_robot_data_retime.interaction.origin_identity import detached_origin
from real_robot_data_retime.timeline.smooth import held_grasp_interval
from real_robot_data_retime.timeline.uniform import validate_open_phase_motion
from real_robot_data_retime.timeline.holds import compress_static_spans


def test_initial_robot_fragment_is_not_a_separate_object():
    robots = np.zeros((15, 2, 20, 20), bool)
    objects = np.zeros((2, 15, 20, 20), bool)
    robots[:, 0, :10, :10] = True
    objects[0, :, :10, :10] = True
    objects[1, :, 15:19, 15:19] = True
    assert not detached_origin(robots, objects, 0, 12, 30)["verified"]
    assert detached_origin(robots, objects, 1, 12, 30)["verified"]


def test_empty_closure_must_not_skip_forward_to_a_different_grasp():
    state = np.zeros((80, 14))
    state[:, 6] = 70
    state[10:20, 6] = 0.1
    state[40:60, 6] = 35
    with pytest.raises(ValueError, match="empty"):
        held_grasp_interval(
            state, state, dict(approach_start=0, pickup_frame=10, release_frame=65), 30
        )


def test_post_open_adjustment_is_preserved_without_extending_B():
    state = np.zeros((80, 14))
    state[25:35, 7] = np.arange(10)
    state[35:, 7] = 9
    right = compress_static_spans(
        state[:, 7:],
        state[:, 7:],
        [(21, 60)],
        30,
        minimum_seconds=0.1,
        guard_seconds=1 / 30,
    )
    assert np.flatnonzero(right == 20)[0] == 20
    assert np.isin(np.arange(25, 35), right).all()
    assert validate_open_phase_motion(state, state, right, 20, 60)["passed"]
    with pytest.raises(ValueError, match="post-opening motion"):
        validate_open_phase_motion(
            state, state, np.r_[np.arange(21), np.arange(60, 80)], 20, 60
        )


def test_drawer_handle_stops_following_free_hand_after_opening():
    from real_robot_data_retime.timeline.drawer_braking import audit_braking

    class Checker:
        margin = 0.005
        max_reach_m = 0.001

        def __init__(self):
            self.handles = []

        def _pose(self, row, side):
            if side:
                self.handles.append(row[0])
            return (
                None,
                None,
                None,
                0,
                np.array([row[0], 0, 0]) if side else np.array([100, 0, 100]),
            )

        def pose_clears_volume(self, *args):
            return True

    checker = Checker()
    state = np.zeros((20, 14))
    state[:, 7] = np.arange(20)
    audit_braking(
        checker,
        state,
        state,
        np.zeros((2, 20, 3)),
        (np.eye(3), np.zeros(3), np.ones(3)),
        dict(pull_start=3, open_frame=10),
        dict(pickup_frame=1),
        np.arange(5),
        np.array([0, 5, 10, 15, 19]),
        0,
        4,
    )
    assert min(checker.handles) == 3 and max(checker.handles) < 10


def test_color_parts_are_completed_at_origin_but_not_merged_while_carried():
    import cv2
    from real_robot_data_retime.compositing.ownership import complete_drawer_origins

    hsv = np.zeros((64, 64, 3), np.uint8)
    hsv[:, :, 2] = 80
    hsv[20:30, 20:35] = [90, 240, 180]
    hsv[30:40, 20:35] = [110, 240, 180]
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    objects = np.zeros((1, 8, 64, 64), bool)
    objects[:, :, 30:40, 20:35] = True
    report = complete_drawer_origins(
        frame,
        np.zeros((8, 2, 64, 64), bool),
        objects,
        [dict(object_id=0, pickup_frame=5)],
    )
    assert objects[0, 0, 20:40, 20:35].all()
    assert not objects[0, 5, 20:30].any()
    assert report[0]["added_origin_pixels"] == 150


def test_small_colored_ghost_at_departed_origin_is_detected():
    from real_robot_data_retime.compositing.verification import OriginAudit

    frames = np.full((12, 48, 48, 3), 80, np.uint8)
    frames[0, 10:30, 10:30] = [220, 150, 10]
    objects = np.zeros((1, 12, 48, 48), bool)
    objects[0, 0, 10:30, 10:30] = True
    audit = OriginAudit(
        frames, objects, [dict(object_id=0, robot_id="left", grasp_frame=1)]
    )
    out = frames[10].copy()
    out[10:12, 10:30] = [220, 150, 10]
    audit.observe(out, frames, [10, 10], np.zeros((48, 48), bool))
    assert audit.items[0]["duplicates"] == 1


def test_both_uniform_variants_use_the_same_wait_candidate_set():
    from real_robot_data_retime.timeline.drawer_wait import opening_wait_candidates
    from real_robot_data_retime.timeline.uniform import uniform_samples

    n = 30
    robots = np.zeros((n, 2, 20, 20), bool)
    objects = np.zeros((1, n, 20, 20), bool)
    robots[:, 1, 8:10, 2:4] = True
    for t in range(n):
        robots[t, 0, 8:10, (2 if t % 2 else 14) : (4 if t % 2 else 16)] = True
    events = [dict(object_id=0, robot_id="left", pickup_frame=0, release_frame=n)]
    for index in range(87):
        first, second = uniform_samples(index, 87)
        a = opening_wait_candidates(
            np.ones(n, bool), robots, objects, events, 0, 10, first, 30
        )
        b = opening_wait_candidates(
            np.ones(n, bool), robots, objects, events, 0, 10, second, 30
        )
        assert np.array_equal(a, b)


def test_candidate_search_cannot_turn_lift_stage_into_a_low_grasp_pose(monkeypatch):
    from real_robot_data_retime.timeline.drawer_wait import prepare_uniform_lift
    from real_robot_data_retime.collision import piperx

    class Rig:
        def __init__(self, left, right, *args, **kwargs):
            self.poses = [
                [self._pose(row, side) for row in values]
                for side, values in enumerate((left, right))
            ]

        def _pose(self, row, side):
            return (None, None, None, 0, np.array([row[0], side * 0.5, row[1]]))

        def arm_clears_volume(self, *args, **kwargs):
            return True

        def pose_clears_volume(self, *args, **kwargs):
            return True

    monkeypatch.setattr(piperx, "PiperXClearance", Rig)
    state = np.zeros((60, 14))
    state[:, 6] = 70
    state[12:40, 6] = 35
    state[:, 1] = 0.1
    state[20, 1] = 0.3
    state[21, 1] = 0.299
    state[:, 7] = np.arange(60) * 0.02
    robots = np.zeros((60, 2, 20, 20), bool)
    robots[:, 0, 10, 2] = True
    robots[:, 1, 10, 15] = True
    event = dict(
        approach_start=0,
        pickup_frame=10,
        release_frame=50,
        object_id=0,
        robot_id="left",
    )
    prepared = prepare_uniform_lift(
        state,
        state,
        event,
        dict(open_frame=8, close_start=45, pull_start=2),
        robots,
        np.zeros((1, 60, 20, 20), bool),
        [event],
        "unused",
        "unused",
        30,
        0,
    )
    assert set(prepared.candidates) == {20, 21}
    assert prepared.recorded_peak_height == 0.3


def test_post_open_smoothing_preserves_B_completion_time():
    from real_robot_data_retime.timeline.smooth import smooth_wait_boundaries

    opening = 20
    suffix = np.arange(opening, 70)
    clock, _, report = smooth_wait_boundaries(suffix, suffix, 30, stop_indices=[10, 25])
    full = np.r_[np.arange(opening), clock]
    assert np.array_equal(full[: opening + 1], np.arange(opening + 1))
    for phase in report["transitions"]:
        stop = phase["stop_output_frame"]
        assert stop - phase["brake_start_output_frame"] == 15
        assert phase["restart_end_output_frame"] - stop == 9
        assert clock[stop] - clock[stop - 1] < 0.01


def test_partial_closure_plateau_does_not_hide_later_release():
    state = np.zeros((100, 14))
    state[:, 6] = 70
    state[10:20, 6] = 57
    state[20:60, 6] = 35
    state[60:70, 6] = 45  # Reopened, even though below the early 57 mm plateau.
    state[70:90, 6] = 1.3
    start, end, limit = held_grasp_interval(
        state, state, dict(approach_start=0, pickup_frame=10, release_frame=90), 30
    )
    assert (start, end, limit) == (20, 60, 35.5)


def test_finalization_rejects_an_explicitly_failed_visual_receipt(
    tmp_path, monkeypatch
):
    import json
    from real_robot_data_retime import uniform_drawer as module

    source = tmp_path / "source"
    output = tmp_path / "output"
    (source / "meta").mkdir(parents=True)
    (output / "meta/retime_receipts").mkdir(parents=True)
    (source / "meta/info.json").write_text(json.dumps(dict(total_episodes=1)))
    (output / "meta/retime_receipts/episode_000.json").write_text(
        json.dumps(dict(visual_review=dict(passed=False)))
    )
    monkeypatch.setattr(module, "source_episodes", lambda _: [{}])
    monkeypatch.setattr(module, "read_episode", lambda *args: None)
    with pytest.raises(ValueError, match="rejected by visual"):
        module.finalize_dataset(dict(source=str(source), output=str(output)))


def test_fractional_stop_index_is_not_silently_rounded():
    from real_robot_data_retime.timeline.smooth import smooth_wait_boundaries

    with pytest.raises(ValueError, match="integer path vertices"):
        smooth_wait_boundaries(np.arange(40), np.arange(40), 30, stop_indices=[12.5])


def test_placed_cube_cannot_block_left_withdrawal_as_a_robot_fragment():
    from real_robot_data_retime.compositing.ownership import (
        exclude_placed_objects,
        arm_foreground,
    )

    robots = np.zeros((10, 2, 20, 20), bool)
    objects = np.zeros((1, 10, 20, 20), bool)
    objects[0, :, 8:12, 8:12] = True
    robots[:, 0, 8:12, 8:12] = True
    event = dict(object_id=0, robot_id="left", pickup_frame=2, release_frame=5)
    exclude_placed_objects(robots, objects, [event])
    assert arm_foreground(robots, objects, [event], 0, 4).any()
    assert not arm_foreground(robots, objects, [event], 0, 5).any()
    assert objects[0, 5].any()


def test_scene_geometry_is_explicit_validated_and_shared_by_box_models():
    from real_robot_data_retime.collision.drawer import DrawerGeometry, drawer_body

    geometry = DrawerGeometry.from_mapping(
        dict(
            drawer_width_m=0.12,
            drawer_depth_m=0.10,
            drawer_height_m=0.04,
            held_object_radius_m=0.02,
        )
    )
    _, lo, hi = drawer_body(np.zeros(3), np.eye(3), **geometry.box_kwargs)
    assert np.allclose(hi - lo, [0.10, 0.12, 0.04])
    assert geometry.held_object_radius_m == 0.02
    with pytest.raises(ValueError):
        DrawerGeometry.from_mapping(dict(drawer_width_m=0.12))
    with pytest.raises(ValueError):
        DrawerGeometry(held_object_radius_m=-1)
    with pytest.raises(ValueError):
        DrawerGeometry(drawer_width_m="0.12")


def test_source_mapping_digest_detects_post_prerequisite_changes():
    from real_robot_data_retime.uniform_drawer import source_map_digest

    left = np.arange(20)
    right = np.arange(20)
    digest = source_map_digest(left, right)
    assert source_map_digest(left.astype(float), right.astype(float)) == digest
    right = right.astype(float)
    right[15] += 0.1
    assert source_map_digest(left, right) != digest
