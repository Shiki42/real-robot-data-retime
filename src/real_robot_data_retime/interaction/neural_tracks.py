"""Automatic segmentation seeds, bounded propagation and measured-track checks."""

import numpy as np
from ..segmentation.sam_backend import SamVideo


def mask_measurements(mask):
    yy, xx = np.where(mask)
    if len(xx) == 0:
        return np.array([np.nan, np.nan]), 0
    return np.array([xx.mean(), yy.mean()]), len(xx)


def segment_candidates(frames, proposals, geometry, sam=None):
    sam = sam or SamVideo()
    n, h, w = frames.shape[:3]
    centers = np.full((len(proposals), n, 2), np.nan)
    areas = np.zeros((len(proposals), n))
    packed = np.zeros((len(proposals), n, h, (w + 7) // 8), np.uint8)
    for t, masks in sam.propagate(frames, proposals):
        for k, mask in enumerate(masks):
            center, area = mask_measurements(mask)
            # Reject a mask that expands to the robot or jumps to a new region.
            if area > proposals[k]["area"] * 4:
                continue
            if t > 0 and np.isfinite(centers[k, t - 1]).all():
                if np.linalg.norm(center - centers[k, t - 1]) > w * 0.12:
                    continue
            centers[k, t] = center
            areas[k, t] = area
            packed[k, t] = np.packbits(mask, axis=-1)
    tracks = [
        dict(
            proposal=p,
            centers=centers[k],
            areas=areas[k],
            packed_masks=packed[k],
            confidence=np.minimum(areas[k] / max(1, p["area"]), 1.0),
        )
        for k, p in enumerate(proposals)
    ]
    return tracks


def segment_grippers(frames, geometry, sam=None):
    """Derive end effectors from identity-anchored whole-arm video masks."""
    sam = sam or SamVideo("facebook/sam2.1-hiera-large")
    n, h, w = frames.shape[:3]
    robots, seeds = segment_robots(frames, geometry, sam)
    result = grippers_from_robots(robots, (h, w))
    result["seeds"] = seeds
    return result


def grippers_from_robots(robots, frame_shape):
    n = len(robots)
    h, w = frame_shape
    centers = np.full((n, 2, 2), np.nan)
    apertures = np.full((n, 2), np.nan)
    masks_by_side = np.zeros((n, 2, h, w), bool)
    for t in range(n):
        for side in [0, 1]:
            robot = np.unpackbits(robots[t, side], axis=-1, count=w).astype(bool)
            from .gripper_geometry import end_effector

            estimate = end_effector(robot, side)
            if estimate is None:
                continue
            center, spread, local = estimate
            masks_by_side[t, side] = local
            centers[t, side] = center
            apertures[t, side] = spread
    return dict(
        centers=centers,
        apertures=apertures,
        masks=masks_by_side,
        seeds=[],
        robot_masks=robots,
    )


def recover_candidates(frames, proposals, tracks, geometry, sam, *, task, robots):
    """Reinitialize from automatic reappearance hypotheses and propagate back.

    Preserve a measured origin track; do not interpolate invisible object points.
    Candidate identities are retained only when visible overlap agrees.
    """
    import cv2
    from scipy.ndimage import gaussian_filter
    from .discovery import track_candidates
    from .verification import origin_presence

    from .verification import validate_track_colors, color_support

    validate_track_colors(frames, proposals, tracks, robots)
    appearances = track_candidates(frames, proposals)
    n, h, w = frames.shape[:3]
    bounds = None
    if task == "drawer":
        from ..tasks.drawer import destination_bounds

        bounds = destination_bounds(frames)
    retries = []
    for k, (proposal, track, appearance) in enumerate(
        zip(proposals, tracks, appearances)
    ):
        distance_to_origin = np.linalg.norm(
            geometry["centers"] - proposal["origin"], axis=-1
        )
        closest = np.argmin(np.nan_to_num(distance_to_origin, nan=np.inf), axis=1)
        nearest_gripper = geometry["centers"][np.arange(n), closest]
        observations = origin_presence(frames, proposal, nearest_gripper, 0, n)
        reset_candidates = []
        for t, similarity in observations:
            wrong = not np.isfinite(track["centers"][t]).all() or np.linalg.norm(
                track["centers"][t] - proposal["origin"]
            ) > max(proposal["bbox"][2:])
            if similarity > 0.92 and wrong:
                reset_candidates.append(t)
        if len(reset_candidates) >= 3:
            # Last reliable origin observation starts a fresh forward hypothesis
            # after an occluding distractor has passed the object.
            reset = reset_candidates[-1]
            for t, masks in sam.propagate(frames, [proposal], seed_frame=reset):
                c, area = mask_measurements(masks[0])
                if area <= proposal["area"] * 4:
                    track["centers"][t] = c
                    track["areas"][t] = area
                    if "packed_masks" in track:
                        track["packed_masks"][t] = np.packbits(masks[0], axis=-1)
            retries.append(
                dict(
                    object_id=k,
                    seed_frame=reset,
                    kind="origin_identity_reset",
                    accepted=True,
                )
            )
        centers = appearance["centers"]
        dist = np.linalg.norm(centers - proposal["origin"], axis=1)
        valid = (appearance["confidence"] > 0.55) & (dist > w * 0.06)
        if bounds is not None:
            valid &= (
                (centers[:, 0] >= bounds[:, 0])
                & (centers[:, 0] < bounds[:, 2])
                & (centers[:, 1] >= bounds[:, 1])
                & (centers[:, 1] < bounds[:, 3])
            )
        candidates = np.flatnonzero(valid)
        if len(candidates) < 5:
            continue
        seed = int(candidates[len(candidates) // 2])
        cx, cy = centers[seed]
        bw, bh = proposal["bbox"][2:]
        box = [max(0, cx - bw / 2), max(0, cy - bh / 2), bw, bh]
        recovered = np.full((n, 2), np.nan)
        areas = np.zeros(n)
        recovered_masks = np.zeros((n, h, (w + 7) // 8), np.uint8)
        seed_mask = None
        for reverse in [False, True]:
            for t, masks in sam.propagate(
                frames, [{"bbox": box}], seed_frame=seed, reverse=reverse
            ):
                if t == seed:
                    seed_mask = masks[0].copy()
                c, area = mask_measurements(masks[0])
                if proposal["appearance_model"] == "hue":
                    count, fraction = color_support(
                        cv2.cvtColor(frames[t], cv2.COLOR_BGR2HSV), masks[0], proposal
                    )
                    if count < 8 or fraction < 0.05:
                        continue
                if area <= proposal["area"] * 4:
                    recovered[t] = c
                    areas[t] = area
                    recovered_masks[t] = np.packbits(masks[0], axis=-1)
        observed = np.isfinite(recovered).all(axis=1)
        overlap = observed & np.isfinite(track["centers"]).all(axis=1)
        consistent = (
            np.linalg.norm(recovered - track["centers"], axis=1) < w * 0.04
        ) & overlap
        moved = observed & (
            np.linalg.norm(recovered - proposal["origin"], axis=1) > w * 0.04
        )
        # A reappearing candidate must be near the corresponding moving gripper
        # over several observed frames, beyond just being the same color.
        distance = np.linalg.norm(recovered[:, None, :] - geometry["centers"], axis=-1)
        contact = np.any(distance < w * 0.13, axis=1) & moved
        original_hsv = cv2.cvtColor(frames[0], cv2.COLOR_BGR2HSV)
        seed_hsv = cv2.cvtColor(frames[seed], cv2.COLOR_BGR2HSV)
        # Hue is circular and less sensitive than saturation to a dark-table
        # versus white-container background. Discard nearly achromatic pixels.
        mask0 = proposal["mask"] & (original_hsv[:, :, 1] > 80)
        mask1 = seed_mask & (seed_hsv[:, :, 1] > 80)
        hist0 = cv2.calcHist(
            [original_hsv], [0], mask0.astype(np.uint8), [36], [0, 180]
        )
        hist1 = cv2.calcHist([seed_hsv], [0], mask1.astype(np.uint8), [36], [0, 180])
        hist0 = gaussian_filter(hist0, (2.0, 0.0), mode=("wrap", "nearest"))
        hist1 = gaussian_filter(hist1, (2.0, 0.0), mode=("wrap", "nearest"))
        cv2.normalize(hist0, hist0)
        cv2.normalize(hist1, hist1)
        appearance_distance = (
            float(cv2.compareHist(hist0, hist1, cv2.HISTCMP_BHATTACHARYYA))
            if min(mask0.sum(), mask1.sum()) >= 10
            else 1.0
        )
        # A fully occluded singleton has no overlapping visible observations.
        # Permit that hypothesis only with strong appearance and later contact;
        # origin removal is still independently required by the event verifier.
        color = proposal["color"]
        competitors = []
        for other in proposals:
            if other is proposal:
                continue
            dh = abs(float(color[0] - other["color"][0]))
            dh = min(dh, 180 - dh)
            brightness_ratio = max(color[2], other["color"][2]) / max(
                1, min(color[2], other["color"][2])
            )
            if dh < 15 and brightness_ratio < 3:
                competitors.append(other)
        singleton_bridge = (
            not competitors and appearance_distance < 0.45 and contact.sum() >= 10
        )
        ok = (int(consistent.sum()) >= 3 or singleton_bridge) and int(
            contact.sum()
        ) >= 5
        if proposal["appearance_model"] == "hue":
            ok = ok and appearance_distance < 0.45
        retries.append(
            dict(
                object_id=k,
                seed_frame=seed,
                accepted=bool(ok),
                appearance_distance=appearance_distance,
                occlusion_bridge=bool(singleton_bridge),
                consistent_overlap_frames=int(consistent.sum()),
                contact_frames=int(contact.sum()),
            )
        )
        if ok:
            missing = ~np.isfinite(track["centers"]).all(axis=1)
            preserve_origin = np.isfinite(track["centers"]).all(axis=1) & (
                np.linalg.norm(track["centers"] - proposal["origin"], axis=1)
                < max(proposal["bbox"][2:]) * 0.5
            )
            trusted_movement = moved & singleton_bridge & ~preserve_origin
            replace = observed & (missing | consistent | trusted_movement)
            track["centers"][replace] = recovered[replace]
            track["areas"][replace] = areas[replace]
            if "packed_masks" in track:
                track["packed_masks"][replace] = recovered_masks[replace]
    return tracks, retries


def object_gripper_distances(gripper_masks, object_tracks):
    """Measure actual mask proximity; a fingertip centroid can be off-center."""
    import cv2

    n, sides, h, w = gripper_masks.shape
    result = np.full((len(object_tracks), n, sides), np.nan)
    for t in range(n):
        for side in range(sides):
            mask = gripper_masks[t, side]
            if not mask.any():
                continue
            distance = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
            for k, track in enumerate(object_tracks):
                c = track["centers"][t]
                if not np.isfinite(c).all():
                    continue
                object_mask = np.unpackbits(
                    track["packed_masks"][t], axis=-1, count=w
                ).astype(bool)
                if object_mask.any():
                    result[k, t, side] = float(distance[object_mask].min())
    return result


def terminal_letter_recovery(frames, proposals, tracks, sam):
    """Match all visible terminal letters jointly, then track backward."""
    from scipy.optimize import linear_sum_assignment
    from .discovery import task_object_proposals, InteractionConfig

    terminal = task_object_proposals(
        frames[-8:], "letters", InteractionConfig(minimum_object_area=40)
    )
    if len(terminal) < len(proposals):
        return tracks, [
            dict(
                kind="terminal_matching",
                accepted=False,
                reason="incomplete_terminal_objects",
            )
        ]
    cost = np.zeros((len(proposals), len(terminal)))
    for i, p in enumerate(proposals):
        for j, q in enumerate(terminal):
            dh = abs(float(p["color"][0] - q["color"][0]))
            dh = min(dh, 180 - dh)
            cost[i, j] = (
                dh / 15
                + abs(np.log((p["color"][2] + 10) / (q["color"][2] + 10)))
                + 0.2 * abs(np.log(p["area"] / q["area"]))
            )
    rows, cols = linear_sum_assignment(cost)
    retries = []
    n, h, w = frames.shape[:3]
    for k, j in zip(rows, cols):
        if cost[k, j] > 2.5:
            retries.append(
                dict(
                    object_id=int(k),
                    kind="terminal_matching",
                    accepted=False,
                    cost=float(cost[k, j]),
                )
            )
            continue
        recovered = np.full((n, 2), np.nan)
        areas = np.zeros(n)
        recovered_masks = np.zeros((n, h, (w + 7) // 8), np.uint8)
        for t, masks in sam.propagate(
            frames, [terminal[j]], seed_frame=n - 1, reverse=True
        ):
            c, area = mask_measurements(masks[0])
            if area <= proposals[k]["area"] * 4:
                recovered[t] = c
                areas[t] = area
                recovered_masks[t] = np.packbits(masks[0], axis=-1)
        observed = np.isfinite(recovered).all(axis=1)
        overlap = observed & np.isfinite(tracks[k]["centers"]).all(axis=1)
        consistent = overlap & (
            np.linalg.norm(recovered - tracks[k]["centers"], axis=1) < w * 0.04
        )
        tracks[k]["terminal_hypothesis"] = recovered
        tracks[k]["terminal_hypothesis_areas"] = areas
        accepted = consistent.sum() >= 3
        if not accepted:
            large = SamVideo("facebook/sam2.1-hiera-large")
            forward = np.full((n, 2), np.nan)
            forward_areas = np.zeros(n)
            forward_masks = np.zeros_like(recovered_masks)
            for t, masks in large.propagate(frames, [proposals[k]]):
                c, area = mask_measurements(masks[0])
                if area <= proposals[k]["area"] * 4:
                    forward[t] = c
                    forward_areas[t] = area
                    forward_masks[t] = np.packbits(masks[0], axis=-1)
            end_ok = (
                np.isfinite(forward[-1]).all()
                and np.linalg.norm(forward[-1] - terminal[j]["origin"]) < w * 0.04
            )
            if end_ok:
                measured = np.isfinite(forward).all(axis=1)
                tracks[k]["centers"][measured] = forward[measured]
                tracks[k]["areas"][measured] = forward_areas[measured]
                if "packed_masks" in tracks[k]:
                    tracks[k]["packed_masks"][measured] = forward_masks[measured]
                consistent = (
                    observed
                    & measured
                    & (np.linalg.norm(recovered - forward, axis=1) < w * 0.04)
                )
                accepted = consistent.sum() >= 3
            retries.append(
                dict(
                    object_id=int(k),
                    kind="large_model_identity_retry",
                    accepted=bool(accepted),
                    terminal_verified=bool(end_ok),
                )
            )
            del large
        retries.append(
            dict(
                object_id=int(k),
                kind="terminal_matching",
                accepted=bool(accepted),
                cost=float(cost[k, j]),
                consistent_overlap_frames=int(consistent.sum()),
            )
        )
        if accepted:
            tracks[k]["centers"][observed] = recovered[observed]
            tracks[k]["areas"][observed] = areas[observed]
            if "packed_masks" in tracks[k]:
                tracks[k]["packed_masks"][observed] = recovered_masks[observed]
    return tracks, retries


def propagate_robot(frames, proposal, sam, side, seed, *, start=0, stop=None):
    """Propagate one identity with the same entry support at every source frame."""
    import cv2
    from .video import components, pixel_kernel

    n, h, w = frames.shape[:3]
    stop = n if stop is None else stop
    packed = np.zeros((n, h, (w + 7) // 8), np.uint8)
    for reverse in [False, True]:
        for t, masks in sam.propagate(
            frames,
            [proposal],
            seed_frame=seed,
            reverse=reverse,
            stop_frame=start - 1 if reverse else stop,
        ):
            keep = np.zeros((h, w), bool)
            linked = cv2.dilate(
                masks[0].astype(np.uint8),
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (pixel_kernel(15, w),) * 2
                ),
            )
            for region, stat, center in components(linked, 20):
                anchored = (
                    stat[0] < w * 0.08 if side == 0 else stat[0] + stat[2] > w * 0.92
                )
                opposite = (
                    stat[0] + stat[2] > w * 0.92 if side == 0 else stat[0] < w * 0.08
                )
                exits_top = stat[1] < h * 0.02 and stat[4] > max(
                    100, proposal["mask"].sum() * 0.03
                )
                if anchored or (not opposite and exits_top):
                    keep |= region & masks[0]
            packed[t] = np.packbits(keep, axis=-1)
    return packed


def segment_robots(frames, geometry, sam, sides=(0, 1)):
    """Track whole articulated arms using automatically generated support points."""
    from .robot_discovery import robot_prompt, prompt_from_robot_region
    from .evidence import stable_runs

    n, h, w = frames.shape[:3]
    packed = np.zeros((n, 2, h, (w + 7) // 8), np.uint8)
    seeds = []
    for side in sides:
        seed, proposal = robot_prompt(frames, geometry, side)
        seeds.append(dict(frame=seed, bbox=proposal["bbox"]))
        packed[:, side] = propagate_robot(frames, proposal, sam, side, seed)
        reference_area = np.sum(geometry["masks"] == side + 1, axis=(1, 2))
        visible = np.unpackbits(packed[:, side], axis=-1, count=w).sum(axis=(1, 2))
        missing = (visible < 40) & (reference_area > 150)
        for a, b in stable_runs(missing, 15)[:2]:
            reset = int(a + np.argmax(reference_area[a : min(b, a + 30)]))
            prompt = prompt_from_robot_region(geometry["masks"][reset] == side + 1)
            seeds.append(
                dict(
                    frame=reset,
                    bbox=prompt["bbox"],
                    side=side,
                    kind="visibility_reentry",
                )
            )
            restored = propagate_robot(
                frames, prompt, sam, side, reset, start=a, stop=b
            )
            packed[a:b, side] = restored[a:b]
    return packed, seeds
