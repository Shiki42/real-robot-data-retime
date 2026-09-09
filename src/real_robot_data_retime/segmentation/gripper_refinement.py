"""Per-frame semantic gripper refinement in identity-anchored local crops."""

import cv2
import numpy as np

from ..interaction.gripper_geometry import end_effector
from ..interaction.neural_tracks import mask_measurements
from ..interaction.video import components


def crop_bounds(center, shape, radius):
    h, w = shape
    x, y = map(int, np.rint(center))
    return (
        max(0, x - radius),
        max(0, y - radius),
        min(w, x + radius + 1),
        min(h, y + radius + 1),
    )


def supported_gripper(raw, parent, other, radius):
    """Retain nearby semantic evidence, exposing rejected and ambiguous pixels.

    A small support dilation permits recovery of reflective gaps, not unlimited
    growth into a deposited object. Cross-arm overlap remains unassigned.
    """
    if raw.shape != parent.shape or raw.shape != other.shape or radius < 0:
        raise ValueError("inconsistent gripper/arm geometry")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
    support = cv2.dilate(parent.astype(np.uint8), kernel).astype(bool)
    accepted = np.zeros_like(raw, dtype=bool)
    for region, stat, _ in components(raw, 8):
        if np.mean(support[region]) >= 0.5:
            accepted |= region & support
    ambiguous = accepted & other
    accepted &= ~other
    return accepted, {
        "raw_pixels": int(raw.sum()),
        "accepted_pixels": int(accepted.sum()),
        "ambiguous_pixels": int(ambiguous.sum()),
    }


def refine_grippers(frames, robots, model):
    n, h, w = frames.shape[:3]
    if robots.shape != (n, 2, h, (w + 7) // 8) or robots.dtype != np.uint8:
        raise ValueError("robot masks do not match source frames")
    raw_packed = np.zeros_like(robots)
    refined = np.zeros_like(robots)
    centers = np.full((n, 2, 2), np.nan)
    observations = []
    for t, frame in enumerate(frames):
        parents = np.unpackbits(robots[t], axis=-1, count=w).astype(bool)
        for side in (0, 1):
            parent = parents[side]
            entry = {"frame": t, "side": side, "observed": False}
            observations.append(entry)
            if parent.sum() < h * w * 0.002:
                entry["reason"] = "insufficient_arm_support"
                continue
            estimate = end_effector(parent, side)
            if estimate is None:
                entry["reason"] = "unresolved_end_effector"
                continue
            x0, y0, x1, y1 = crop_bounds(estimate[0], (h, w), max(24, round(w * 0.16)))
            crop = frame[y0:y1, x0:x1]
            outputs = list(model.segment_semantic(crop[None], category="gripper"))
            if (
                len(outputs) != 1
                or outputs[0][0] != 0
                or outputs[0][1].shape != crop.shape[:2]
            ):
                raise ValueError(
                    "single-frame gripper predictor violated crop contract"
                )
            raw = np.zeros((h, w), bool)
            raw[y0:y1, x0:x1] = outputs[0][1]
            mask, audit = supported_gripper(
                raw, parent, parents[1 - side], max(2, round(w * 0.01))
            )
            raw_packed[t, side] = np.packbits(raw, axis=-1)
            refined[t, side] = np.packbits(mask, axis=-1)
            center, area = mask_measurements(mask)
            centers[t, side] = center
            entry.update(
                audit,
                crop=[x0, y0, x1, y1],
                observed=bool(area),
                reason="semantic_observation"
                if area
                else "semantic_unobserved_or_unsupported",
            )
        if t % 30 == 0:
            print(f"gripper refinement {t}/{n}", flush=True)
    return {
        "raw_grippers": raw_packed,
        "grippers": refined,
        "centers": centers,
        "observations": observations,
    }


def agreed_mask(forward, backward, minimum_iou=0.6):
    union = np.count_nonzero(forward | backward)
    intersection = forward & backward
    iou = float(intersection.sum() / union) if union else 0.0
    return (
        intersection
        if iou >= minimum_iou and intersection.sum() >= 8
        else np.zeros_like(intersection)
    ), iou


def repair_gripper_gaps(frames, robots, direct, model, fps):
    """Propose only bounded, bidirectionally agreeing masks; preserve direct data."""
    from ..interaction.evidence import stable_runs
    from ..interaction.robot_discovery import prompt_from_robot_region

    n, h, w = frames.shape[:3]
    if direct["grippers"].shape != robots.shape or len(direct["observations"]) != n * 2:
        raise ValueError("direct gripper observations do not match source geometry")
    proposals = direct["grippers"].copy()
    repairs = []
    obs = {(entry["frame"], entry["side"]): entry for entry in direct["observations"]}
    for side in (0, 1):
        visible = np.array([obs[t, side]["observed"] for t in range(n)])
        eligible = np.array(["crop" in obs[t, side] for t in range(n)])
        reliable = visible & np.array(
            [obs[t, side].get("accepted_pixels", 0) >= 32 for t in range(n)]
        )
        for a, b in stable_runs(~visible & eligible, 1):
            if b - a > round(fps):
                continue
            margin = max(1, round(fps * 0.5))
            before = np.flatnonzero(reliable[max(0, a - margin) : a]) + max(
                0, a - margin
            )
            after = np.flatnonzero(reliable[b : min(n, b + margin)]) + b
            if not len(before) or not len(after):
                continue
            lo, hi = int(before[-1]), int(after[0])
            boxes = np.array(
                [
                    obs[t, side]["crop"]
                    for t in range(lo, hi + 1)
                    if "crop" in obs[t, side]
                ]
            )
            x0, y0 = boxes[:, :2].min(axis=0)
            x1, y1 = boxes[:, 2:].max(axis=0)
            clip = frames[lo : hi + 1, y0:y1, x0:x1]
            predictions = []
            for reverse, source in ((False, lo), (True, hi)):
                seed_mask = np.unpackbits(
                    direct["grippers"][source, side], axis=-1, count=w
                ).astype(bool)[y0:y1, x0:x1]
                prompt = prompt_from_robot_region(seed_mask)
                parent = np.unpackbits(robots[source, side], axis=-1, count=w).astype(
                    bool
                )[y0:y1, x0:x1]
                negative = parent & ~cv2.dilate(
                    seed_mask.astype(np.uint8), np.ones((9, 9), np.uint8)
                ).astype(bool)
                yy, xx = np.where(negative)
                chosen = np.linspace(0, len(xx) - 1, min(8, len(xx)), dtype=int)
                prompt["negative_points"] = (
                    np.column_stack([xx[chosen], yy[chosen]]).astype(float).tolist()
                )
                masks = {}
                for local, mask in model.propagate(
                    clip,
                    [prompt],
                    seed_frame=source - lo,
                    reverse=reverse,
                    category="gripper",
                ):
                    t = local + lo
                    if a <= t < b:
                        raw = np.zeros((h, w), bool)
                        raw[y0:y1, x0:x1] = mask[0]
                        parents = np.unpackbits(robots[t], axis=-1, count=w).astype(
                            bool
                        )
                        masks[t], _ = supported_gripper(
                            raw,
                            parents[side],
                            parents[1 - side],
                            max(2, round(w * 0.01)),
                        )
                predictions.append(masks)
            for t in range(a, b):
                mask, iou = agreed_mask(predictions[0][t], predictions[1][t])
                if mask.any():
                    proposals[t, side] = np.packbits(mask, axis=-1)
                repairs.append(
                    {
                        "frame": t,
                        "side": side,
                        "gap": [int(a), int(b)],
                        "seed_frames": [lo, hi],
                        "iou": iou,
                        "proposed_pixels": int(mask.sum()),
                        "proposal_only": True,
                    }
                )
    return proposals, repairs
