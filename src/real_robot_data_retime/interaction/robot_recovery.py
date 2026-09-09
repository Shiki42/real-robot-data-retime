"""Automatic robot cues that separate moving hardware from scene changes."""

import cv2
import numpy as np
from ..background.clean_plate import match_background_colors, temporal_plate, dilate
from .video import components
from .robot_discovery import robot_entry_side


def scene_region(frames):
    h, w = frames.shape[1:3]
    hsv = cv2.cvtColor(frames[0], cv2.COLOR_BGR2HSV)
    red = (
        ((hsv[:, :, 0] < 12) | (hsv[:, :, 0] > 170))
        & (hsv[:, :, 1] > 120)
        & (hsv[:, :, 2] > 70)
    )
    cabinets = components(red, int(h * w * 0.005))
    if any(stat[2] > w * 0.15 for _, stat, _ in cabinets):
        from ..compositing.layers import drawer_region

        return drawer_region(frames, 0)
    return np.zeros((h, w), bool)


def reference_geometry(frames, geometry, robots):
    """Use observed clean background, not a first frame containing parked arms."""
    n, h, w = frames.shape[:3]
    scene = scene_region(frames)
    robot_union = np.unpackbits(robots, axis=-1, count=w).astype(bool).any(axis=1)
    excluded = np.array([dilate(mask, 3) for mask in robot_union])
    normalized, _ = match_background_colors(frames, excluded, scene)
    plate, coverage = temporal_plate(normalized, excluded | scene[None])
    observed_background = (coverage >= 3) & ~scene
    labels = np.zeros((n, h, w), np.uint8)
    scene_contamination = np.zeros((n, 2), bool)
    for t, frame in enumerate(frames):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        scene_surface = scene & (
            (
                ((hsv[:, :, 0] < 12) | (hsv[:, :, 0] > 170))
                & (hsv[:, :, 1] > 120)
                & (hsv[:, :, 2] > 70)
            )
            | ((hsv[:, :, 1] < 60) & (hsv[:, :, 2] > 130))
        )
        for side in [0, 1]:
            predicted = np.unpackbits(robots[t, side], axis=-1, count=w).astype(bool)
            scene_contamination[t, side] = (predicted & scene_surface).sum() > max(
                h * w * 0.007, predicted.sum() * 0.1
            )
        hardware = (hsv[:, :, 1] < 100) | (hsv[:, :, 2] < 25)
        changed = np.max(cv2.absdiff(normalized[t], plate), axis=-1) > 25
        support = observed_background & changed & hardware
        support |= scene & (geometry["masks"][t] > 0) & hardware & (hsv[:, :, 2] < 80)
        support[: int(h * 0.23)] = False
        support = cv2.morphologyEx(
            support.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)
        )
        for region, stat, center in components(support, int(h * w * 0.015)):
            if (
                region[:, : max(1, int(w * 0.08))].any()
                and region[:, int(w * 0.92) :].any()
            ):
                continue
            side = robot_entry_side(region)
            if side is None:
                if center[1] < h * 0.4:
                    continue
                side = int(center[0] >= w * 0.5)
            labels[t, region] = side + 1
    return {
        **geometry,
        "masks": labels,
        "reference_method": "observed_clean_background_and_dark_scene_hardware",
        "scene_contamination": scene_contamination,
    }


def recovery_prompts(frames, side, other_robot):
    """Generate early and peak-motion hypotheses with explicit scene negatives."""
    from .evidence import stable_runs
    from .robot_discovery import prompt_from_robot_region

    n, h, w = frames.shape[:3]
    normalized, _ = match_background_colors(frames, np.zeros((n, h, w), bool))
    background = np.median(normalized[: min(8, n)], axis=0).astype(np.uint8)
    candidates = []
    for t, frame in enumerate(normalized):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        appearance = (hsv[:, :, 2] < 115) & (hsv[:, :, 1] < 80)
        appearance[: int(h * 0.23)] = False
        moving = appearance & (np.max(cv2.absdiff(frame, background), axis=-1) > 25)
        closed = cv2.morphologyEx(
            moving.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8)
        )
        for region, stat, center in components(closed, 150):
            if region[:, : int(w * 0.08)].any() and region[:, int(w * 0.92) :].any():
                continue
            identity = robot_entry_side(region)
            if identity is None:
                if center[1] < h * 0.4:
                    continue
                identity = int(center[0] >= w * 0.5)
            if identity != side:
                continue
            support = region & appearance
            score = support.sum() * np.exp(
                -abs(center[0] / w - (0.4 if side == 0 else 0.65)) * 6
            )
            candidates.append((float(score), t, support))
    if not candidates:
        raise ValueError("no automatic robot recovery hypothesis")
    peak = max(candidates, key=lambda item: item[0])
    high = np.zeros(n, bool)
    for score, t, _ in candidates:
        if t >= 8 and score >= peak[0] * 0.5:
            high[t] = True
    runs = stable_runs(high, 5)
    choices = []
    if runs:
        frame = runs[0][0] + 2
        choices.append(
            max(
                (item for item in candidates if item[1] == frame),
                key=lambda item: item[0],
            )
        )
    if not choices or choices[0][1] != peak[1]:
        choices.append(peak)
    scene = scene_region(frames)
    result = []
    for score, t, support in choices:
        hsv = cv2.cvtColor(frames[t], cv2.COLOR_BGR2HSV)
        positive = support & (hsv[:, :, 1] < 80) & (~scene | (hsv[:, :, 2] < 80))
        if positive.sum() < 20:
            continue
        proposal = prompt_from_robot_region(positive)
        x, y, bw, bh = proposal["bbox"]
        proposal["bbox"] = (
            [0, 0, min(w, x + bw + 50), h]
            if side == 0
            else [max(0, x - 50), 0, w - max(0, x - 50), h]
        )
        negative = []
        for region, _, _ in components(
            (hsv[:, :, 1] > 120) & (hsv[:, :, 2] > 70), int(h * w * 0.005)
        ):
            distance = cv2.distanceTransform(region.astype(np.uint8), cv2.DIST_L2, 5)
            yy, xx = np.unravel_index(np.argmax(distance), distance.shape)
            negative.append([float(xx), float(yy)])
        # The white drawer floor is also a scene surface, not robot hardware.
        for region, _, _ in components(
            scene & (hsv[:, :, 1] < 60) & (hsv[:, :, 2] > 130),
            int(h * w * 0.005),
        ):
            distance = cv2.distanceTransform(region.astype(np.uint8), cv2.DIST_L2, 5)
            yy, xx = np.unravel_index(np.argmax(distance), distance.shape)
            negative.append([float(xx), float(yy)])
        other = np.unpackbits(other_robot[t], axis=-1, count=w).astype(bool)
        # A reliable opposing entry patch is a negative even if scene pixels
        # contaminated the opposite arm's interior mask.
        entry = np.zeros((h, w), bool)
        if side == 0:
            entry[int(h * 0.4) :, int(w * 0.8) :] = True
        else:
            entry[int(h * 0.4) :, : int(w * 0.2)] = True
        other &= entry
        if other.sum() >= 20:
            distance = cv2.distanceTransform(other.astype(np.uint8), cv2.DIST_L2, 5)
            yy, xx = np.unravel_index(np.argmax(distance), distance.shape)
            negative.append([float(xx), float(yy)])
        proposal["negative_points"] = negative
        result.append((t, proposal))
    if not result:
        raise ValueError("robot recovery lacks independent hardware support")
    return result


def recover_robot_masks(frames, geometry, robots, sam, sides, fps, task, point_tracks):
    from .neural_tracks import grippers_from_robots, propagate_robot
    from .robot_discovery import robot_mask_audit, prompt_from_robot_region
    from .evidence import stable_runs
    from ..tasks.drawer_constraints import discover_drawer_motion

    n, h, w = frames.shape[:3]
    repaired = robots.copy()
    trials = []
    for side in sides:
        best = None
        for seed, proposal in recovery_prompts(frames, side, repaired[:, 1 - side]):
            candidate = repaired.copy()
            candidate[:, side] = propagate_robot(frames, proposal, sam, side, seed)
            reference = reference_geometry(frames, geometry, candidate)
            area = np.unpackbits(candidate[:, side], axis=-1, count=w).sum(axis=(1, 2))
            reference_area = (reference["masks"] == side + 1).sum(axis=(1, 2))
            reentries = []
            for start, stop in stable_runs(area < 40, 15):
                if (reference_area[start:stop] > 150).sum() < 15:
                    continue
                reset = start + int(np.argmax(reference_area[start:stop]))
                prompt = prompt_from_robot_region(reference["masks"][reset] == side + 1)
                interval = propagate_robot(
                    frames, prompt, sam, side, reset, start=start, stop=stop
                )
                candidate[start:stop, side] = interval[start:stop]
                reentries.append(dict(start=start, stop=stop, seed=reset))
            quality = robot_mask_audit(candidate, reference, fps)["arms"][side]
            confidence = 1.0
            if task == "drawer" and side == 1 and point_tracks is not None:
                g = grippers_from_robots(candidate, (h, w))
                ids = point_tracks["source_indices"]
                motion = discover_drawer_motion(
                    point_tracks["xy"].transpose(1, 0, 2),
                    g["centers"][ids, side],
                    fps / 3,
                    w,
                )
                confidence = motion.confidence if motion is not None else 0.0
            accepted = quality["passed"] and confidence >= 0.5
            score = 1 - quality["insufficient_coverage_fraction"] + confidence
            record = dict(
                side=side,
                seed=seed,
                accepted=bool(accepted),
                mask_quality=quality,
                drawer_motion_confidence=confidence,
                reentries=reentries,
            )
            trials.append(record)
            if best is None or (accepted, score) > best[:2]:
                best = bool(accepted), score, candidate
            if accepted:
                break
        repaired = best[2]
    return grippers_from_robots(repaired, (h, w)), trials
