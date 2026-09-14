"""Evidence-gated mask cleanup and automatic foreground-reference selection."""

import cv2
import numpy as np

from ..background.clean_plate import dilate
from .foreground_reference import ForegroundRegistrationError


def clean_background_masks(frames, masks, plate, coverage):
    """Remove mask regions independently observed as textured static background.

    Flat color agreement alone is insufficient: a white arm on a white table
    must not be erased. Only connected matches with textured evidence and at
    least three unmasked background observations can be removed.
    """
    result = masks.copy()
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean = cv2.boxFilter(gray, -1, (7, 7))
    variance = cv2.boxFilter(gray * gray, -1, (7, 7)) - mean * mean
    textured = (variance > 64) & (coverage >= 3)
    receipts = []
    for t, frame in enumerate(frames):
        difference = np.max(cv2.absdiff(frame, plate), axis=-1)
        agrees = (difference <= 12) & (coverage >= 3)
        reliable = cv2.boxFilter(agrees.astype(np.float32), -1, (7, 7)) > 0.9
        for side in range(2):
            matching = agrees & reliable & result[t, side]
            count, labels, stats, _ = cv2.connectedComponentsWithStats(
                matching.astype(np.uint8), connectivity=8
            )
            removed = 0
            for label in range(1, count):
                if stats[label, cv2.CC_STAT_AREA] < 32:
                    continue
                region = labels == label
                if np.count_nonzero(region & textured) < 12:
                    continue
                result[t, side, region] = False
                removed += int(region.sum())
            if removed:
                receipts.append(
                    {
                        "source_frame": t,
                        "side": side,
                        "removed_background_pixels": removed,
                    }
                )
    return result, receipts


def repair_occlusions(compositor, state, source_pairs, *, radius=30):
    """Find visible references without episode ids, frame annotations or prompts."""
    if radius < 1 or int(radius) != radius:
        raise ValueError("reference search radius must be a positive integer")
    state = np.asarray(state)
    n, _, _h, w = compositor.layers.shape
    if state.shape != (n, 14) or not np.isfinite(state).all():
        raise ValueError("automatic repair requires finite N x 14 measured poses")
    pairs = np.asarray(source_pairs)
    if (
        pairs.ndim != 2
        or pairs.shape[1] != 2
        or not np.isfinite(pairs).all()
        or np.any(pairs < 0)
        or np.any(pairs > n - 1)
    ):
        raise ValueError("invalid requested paired source clocks")
    needed = {}
    for pair in pairs:
        if pair[0] == pair[1] and pair[0] == int(pair[0]):
            continue
        for side in range(2):
            for t in {int(np.floor(pair[side])), int(np.ceil(pair[side]))}:
                needed.setdefault((t, side), set()).update(
                    (int(np.floor(pair[1 - side])), int(np.ceil(pair[1 - side])))
                )
    masks = np.unpackbits(compositor.observed_packed, axis=-1, count=w).astype(bool)
    areas = masks.sum(axis=(2, 3))
    contacts = np.zeros((n, 2), int)
    for t in range(n):
        for side in range(2):
            contacts[t, side] = np.count_nonzero(
                masks[t, side] & dilate(masks[t, 1 - side], 8)
            )
    applied, unresolved, rejected = [], [], []
    limit = np.array([0.05] * 4 + [3.0, 3.0, 0.1])
    for t in range(n):
        for side in range(2):
            if (
                (t, side) not in needed
                or contacts[t, side] < 64
                or areas[t, side] < 128
            ):
                continue
            boundary = compositor.layers[t, 1 - side] & dilate(masks[t, side], 16)
            if not any(
                np.count_nonzero(boundary & ~compositor.layers[other, 1 - side]) >= 128
                for other in needed[(t, side)]
            ):
                continue
            values = state[:, side * 7 : side * 7 + 7]
            ids = np.arange(max(0, t - radius), min(n, t + radius + 1))
            candidate = (np.abs(values[ids] - values[t]) <= limit).all(axis=1)
            candidate &= contacts[ids, side] <= max(8, contacts[t, side] // 10)
            candidate &= areas[ids, side] >= areas[t, side] + 128
            ids = ids[candidate]
            # Prefer the most complete clear silhouette, then nearest in time.
            ids = sorted(
                ids, key=lambda k: (-int(areas[k, side]), abs(int(k) - t), int(k))
            )
            successful = False
            considered = []
            for ref in ids[:5]:
                ref = int(ref)
                # An opening/closing between equal endpoint apertures changes
                # carried-object identity and disqualifies the reference.
                a, b = sorted((t, ref))
                if np.ptp(values[a : b + 1, 6]) > 0.1:
                    continue
                missing = (
                    masks[ref, side] & compositor.layers[t, 1 - side] & ~masks[t, side]
                )
                if missing.sum() < 128:
                    continue
                considered.append(ref)
                before = len(compositor.foreground_references)
                try:
                    compositor.restore_foreground(
                        side, t, t + 1, ref, values, register=True
                    )
                except ForegroundRegistrationError as exc:
                    rejected.append(
                        {
                            "source_frame": t,
                            "side": side,
                            "reference_frame": ref,
                            "reason": str(exc),
                        }
                    )
                    continue
                receipt = compositor.foreground_references[before]
                if receipt["repairs"]:
                    applied.append(receipt)
                    successful = True
                    break
            if not successful:
                unresolved.append(
                    {
                        "source_frame": t,
                        "side": side,
                        "reason": "no verified clear same-object reference",
                        "candidate_frames": considered,
                    }
                )
    return {
        "applied": applied,
        "unresolved": unresolved,
        "rejected_candidates": rejected,
        "requires_review": bool(unresolved),
        "search_radius_frames": radius,
    }
