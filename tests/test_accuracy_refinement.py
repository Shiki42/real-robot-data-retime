import json

import numpy as np
import pytest

from real_robot_data_retime.accuracy_experiment import load_scene
from real_robot_data_retime.segmentation.gripper_refinement import (
    crop_bounds,
    supported_gripper,
)
from real_robot_data_retime.tracking.consensus import spatial_consensus


def test_gripper_recovers_nearby_gap_but_not_detached_object_or_other_arm():
    parent = np.zeros((30, 60), bool)
    parent[10:20, 5:25] = True
    raw = np.zeros_like(parent)
    raw[12:18, 20:28] = True  # A few pixels extend beyond the parent silhouette.
    raw[12:18, 45:52] = True  # Separate table object.
    other = np.zeros_like(parent)
    other[12:18, 26:30] = True
    mask, audit = supported_gripper(raw, parent, other, radius=3)
    assert mask[:, 25].any()  # Repair a nearby gap without requiring exact overlap.
    assert not mask[:, 26:].any()
    assert audit["ambiguous_pixels"] > 0
    assert audit["raw_pixels"] > audit["accepted_pixels"] > 0


def test_empty_gripper_stays_unobserved():
    empty = np.zeros((20, 20), bool)
    mask, audit = supported_gripper(empty, ~empty, empty, radius=2)
    assert not mask.any()
    assert audit["accepted_pixels"] == 0


def test_crop_bounds_at_image_edge_preserve_offset():
    assert crop_bounds([4, 7], (30, 60), 10) == (0, 0, 15, 18)
    assert crop_bounds([55, 27], (30, 60), 10) == (45, 17, 60, 30)
    json.dumps({"crop": crop_bounds(np.array([25.5, 15.5]), (30, 60), 10)})


def test_consensus_retains_cluster_without_interpolating_hidden_frames():
    xy = np.array(
        [
            [[10, 10], [11, 10], [10, 11], [11, 11], [100, 100], [90, 80]],
            [[np.nan, np.nan]] * 6,
        ],
        dtype=float,
    )
    visible = np.isfinite(xy).all(axis=-1)
    clean, kept, centers = spatial_consensus(xy, visible, maximum_diameter=10)
    assert kept[0].tolist() == [True, True, True, True, False, False]
    np.testing.assert_allclose(centers[0], [10.5, 10.5])
    assert np.isnan(clean[1]).all() and np.isnan(centers[1]).all()
    assert not kept[1].any()


def test_consensus_does_not_turn_one_visible_point_into_object_evidence():
    xy = np.zeros((1, 24, 2))
    visible = np.zeros((1, 24), bool)
    visible[0, 0] = True
    clean, kept, center = spatial_consensus(xy, visible, maximum_diameter=10)
    assert not kept.any() and np.isnan(clean).all() and np.isnan(center).all()


def test_other_video_cannot_supply_arm_identity(tmp_path):
    source = tmp_path / "input.mp4"
    source.write_bytes(b"video identity")
    arms = tmp_path / "arms"
    arms.mkdir()
    (arms / "report.json").write_text(json.dumps({"input_sha256": "other-video"}))
    with pytest.raises(ValueError, match="different source video"):
        load_scene(source, arms)


def test_bidirectional_masks_require_nonempty_agreement():
    from real_robot_data_retime.segmentation.gripper_refinement import agreed_mask

    a = np.zeros((30, 30), bool)
    a[10:20, 10:20] = True
    b = np.roll(a, 1, axis=1)
    agreed, iou = agreed_mask(a, b)
    assert iou > 0.8
    np.testing.assert_array_equal(agreed, a & b)
    empty, iou = agreed_mask(a, np.roll(a, 12, axis=1))
    assert not empty.any() and iou < 0.6
    empty, iou = agreed_mask(~a & False, ~a & False)
    assert not empty.any() and iou == 0


def test_tracking_seed_precedes_contact_instead_of_using_partial_occlusion():
    from real_robot_data_retime.interaction.origin_events import tracking_seed_frame

    event = {"contact_start": 120, "last_observed_at_origin": 128}
    assert tracking_seed_frame(event, 30) == 117
    event["contact_start"] = None
    assert tracking_seed_frame(event, 30) == 125
    assert tracking_seed_frame(None, 30) == 0


def test_partial_occlusion_keeps_a_consistent_visible_subset():
    xy = np.full((1, 24, 2), np.nan)
    xy[0, :3] = [[10, 10], [11, 10], [12, 10]]
    visible = np.isfinite(xy).all(axis=-1)
    clean, kept, _ = spatial_consensus(xy, visible, maximum_diameter=10)
    assert kept.sum() == 3
    assert np.isnan(clean[0, 3:]).all()


def test_gap_proposal_does_not_change_direct_observations_or_centers():
    from real_robot_data_retime.segmentation.gripper_refinement import (
        repair_gripper_gaps,
    )

    h, w = 40, 60
    parents = np.zeros((3, 2, h, w), bool)
    parents[:, 0, 5:35, :45] = True
    hands = np.zeros_like(parents)
    hands[[0, 2], 0, 17:23, 25:31] = True
    packed = np.packbits(hands, axis=-1)
    observations = [
        {
            "frame": t,
            "side": side,
            "observed": bool(hands[t, side].any()),
            **(
                {"crop": [0, 0, 50, 40], "accepted_pixels": int(hands[t, side].sum())}
                if side == 0
                else {}
            ),
        }
        for t in range(3)
        for side in range(2)
    ]
    direct = {
        "grippers": packed.copy(),
        "centers": np.full((3, 2, 2), np.nan),
        "observations": observations,
    }

    class Model:
        def propagate(self, frames, proposals, seed_frame, reverse, category):
            assert category == "gripper"
            for t in [2, 1, 0] if reverse else [0, 1, 2]:
                mask = np.zeros(frames.shape[1:3], bool)
                mask[17:23, 25:31] = True
                yield t, mask[None]

    candidate, repairs = repair_gripper_gaps(
        np.zeros((3, h, w, 3), np.uint8),
        np.packbits(parents, axis=-1),
        direct,
        Model(),
        30,
    )
    assert np.unpackbits(candidate[1, 0], axis=-1, count=w).sum() == 36
    np.testing.assert_array_equal(direct["grippers"], packed)
    assert np.isnan(direct["centers"]).all()
    assert not direct["observations"][2]["observed"]
    assert repairs[0]["proposal_only"] and repairs[0]["seed_frames"] == [0, 2]


def test_complementary_arm_views_can_recover_missing_parts_without_noise():
    from real_robot_data_retime.segmentation.arm_refinement import assess_arm_reseed

    reference = np.zeros((40, 60), bool)
    reference[15:30, :30] = True
    before = reference.copy()
    before[:, 15:] = False
    crop = reference & ~before
    full = crop.copy()
    crop[0:4, 50:54] = True  # Not supported by the independent field of view.
    result, report = assess_arm_reseed(before, crop, full, reference, side=0)
    np.testing.assert_array_equal(result, reference)
    assert report["selected_as_proposal"]
    assert report["disagreed_pixels"] == 16
    assert np.all(result[before])


def test_arm_reseed_cannot_merge_both_entry_identities():
    from real_robot_data_retime.segmentation.arm_refinement import assess_arm_reseed

    reference = np.zeros((40, 60), bool)
    reference[15:30, :30] = True
    before = np.zeros_like(reference)
    wrong = reference.copy()
    wrong[15:30, -1] = True
    result, report = assess_arm_reseed(before, wrong, wrong, reference, side=0)
    assert not report["selected_as_proposal"] and not result.any()
