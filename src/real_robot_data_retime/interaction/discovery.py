from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import median_filter

from .video import components, pixel_kernel


@dataclass(frozen=True)
class InteractionConfig:
    minimum_object_area: int = 80
    maximum_object_area: int = 1800
    evidence_frames: int = 30
    weights: tuple = (0.15, 0.10, 0.15, 0.20, 0.25, 0.10, 0.05)
    minimum_confidence: float = 0.65


def motion_and_grippers(frames):
    """Background/color/connected geometry cues; no action or state inputs."""
    n, h, w = frames.shape[:3]
    bg = np.median(frames[: min(8, n)], axis=0).astype(np.uint8)
    grippers = np.full((n, 2, 2), np.nan)
    apertures = np.full((n, 2), np.nan)
    boxes = np.zeros((n, 2, 4), dtype=int)
    masks = np.zeros((n, h, w), np.uint8)
    previous = [None, None]
    energies = np.zeros((n, 2))
    gray_prev = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    for t, f in enumerate(frames):
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            gray_prev, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        gray_prev = gray
        delta = np.max(cv2.absdiff(f, bg), axis=-1)
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        robot = ((hsv[:, :, 2] < 115) & (delta > 25)).astype(np.uint8)
        robot[: int(h * 0.23)] = 0
        robot = cv2.morphologyEx(
            robot, cv2.MORPH_CLOSE, np.ones((pixel_kernel(7, w),) * 2, np.uint8)
        )
        for side in range(2):
            region = robot.copy()
            # Side follows spatial anchoring, never episode order.
            if side == 0:
                region[:, int(w * 0.7) :] = 0
            else:
                region[:, : int(w * 0.3)] = 0
            options = []
            for mask, stat, center in components(region, 150):
                x, y, bw, bh, area = stat
                anchor = (x < w * 0.3) if side == 0 else (x + bw > w * 0.7)
                reference = previous[side]
                near = (
                    reference is not None
                    and x - 40 <= reference[0] <= x + bw + 40
                    and y - 40 <= reference[1] <= y + bh + 40
                )
                if not anchor and not near:
                    continue
                continuity = (
                    1.0
                    if reference is None
                    else np.exp(-np.linalg.norm(center - reference) / (w * 0.25))
                )
                options.append((area**0.5 * (1 + 3 * continuity), mask, stat, center))
            if not options:
                continue
            _, mask, stat, center = max(options, key=lambda x: x[0])
            ys, xs = np.where(mask)
            # Distal direction is toward the workspace from the observed
            # entry edge, rather than down-image (which selects cables/wrists).
            q = np.percentile(xs, 94 if side == 0 else 6)
            distal = xs >= q if side == 0 else xs <= q
            center = np.array([np.median(xs[distal]), np.median(ys[distal])])
            grippers[t, side] = center
            previous[side] = center
            radius = int(w * 0.065)
            x0, y0 = np.maximum(center.astype(int) - radius, 0)
            x1, y1 = np.minimum(center.astype(int) + radius, [w, h])
            boxes[t, side] = [x0, y0, x1, y1]
            # Estimate visible finger spread at the distal silhouette. Use
            # the raw foreground before closing morphology, preserving gaps.
            raw = (hsv[:, :, 2] < 115) & (delta > 25)
            yy, xx = np.where(mask & raw)
            local = (np.abs(xx - center[0]) < radius * 0.65) & (
                np.abs(yy - center[1]) < radius
            )
            spread = 0.0
            if local.sum() > 8:
                points = np.column_stack([xx[local], yy[local]])
                # Each image column samples the two opposing finger contours.
                widths = []
                for col in np.unique(points[:, 0]):
                    rows = points[points[:, 0] == col, 1]
                    if len(rows) >= 3:
                        widths.append(np.percentile(rows, 90) - np.percentile(rows, 10))
                if widths:
                    spread = float(np.percentile(widths, 65))
            gap = spread
            apertures[t, side] = gap
            energies[t, side] = np.mean(np.linalg.norm(flow[mask], axis=-1))
            masks[t][mask] = side + 1
    for side in range(2):
        valid = np.isfinite(grippers[:, side, 0])
        if valid.sum() < 5:
            continue
        for axis in range(2):
            grippers[:, side, axis] = np.interp(
                np.arange(n), np.flatnonzero(valid), grippers[valid, side, axis]
            )
        grippers[:, side] = median_filter(grippers[:, side], size=(5, 1))
        apertures[:, side] = median_filter(
            np.interp(np.arange(n), np.flatnonzero(valid), apertures[valid, side]),
            size=5,
        )
    return dict(
        centers=grippers,
        apertures=apertures,
        boxes=boxes,
        masks=masks,
        prompt_support=masks > 0,
        energy=energies,
        background=bg,
    )


def object_proposals(frames, kind, config):
    initial = np.median(frames[: min(8, len(frames))], axis=0).astype(np.uint8)
    hsv = cv2.cvtColor(initial, cv2.COLOR_BGR2HSV)
    n, h, w = frames.shape[:3]
    if kind == "saturated":
        masks = []
        # Separate color modes so a knob cannot merge into a drawer face.
        for center_hue in range(0, 180, 15):
            dh = np.abs(hsv[:, :, 0].astype(float) - center_hue)
            dh = np.minimum(dh, 180 - dh)
            masks.append((dh < 12) & (hsv[:, :, 1] > 85) & (hsv[:, :, 2] > 15))
    else:
        from ..tasks.workpiece import dark_object_masks

        masks = dark_object_masks(
            hsv[:, :, 2], config.minimum_object_area, config.maximum_object_area
        )
    proposals = []
    for mask in masks:
        mask[: int(h * 0.38)] = False
        mask[:, : int(w * 0.08)] = False
        mask[:, int(w * 0.92) :] = False
        for region, stat, center in components(
            mask, config.minimum_object_area, config.maximum_object_area
        ):
            x, y, bw, bh, area = stat
            if max(bw, bh) > w * 0.15:
                continue
            overlaps = [
                p
                for p in proposals
                if np.sum(region & p["mask"]) / min(area, p["area"]) > 0.65
            ]
            for old in overlaps:
                region |= old["mask"]
                proposals = [p for p in proposals if p is not old]
            if overlaps:
                yy, xx = np.where(region)
                x, y = int(xx.min()), int(yy.min())
                bw, bh = int(xx.max() - x + 1), int(yy.max() - y + 1)
                area = int(region.sum())
                stat = np.array([x, y, bw, bh, area])
                center = np.array([xx.mean(), yy.mean()])
            pixels = hsv[region]
            bright = pixels[:, 2] >= np.percentile(pixels[:, 2], 65)
            color = np.median(pixels[bright], axis=0)
            proposals.append(
                dict(
                    origin=center,
                    area=int(area),
                    bbox=stat[:4],
                    color=color,
                    appearance_model="hue"
                    if kind == "saturated" or color[1] > 95
                    else "darkness",
                    mask=region,
                )
            )
    return proposals


def track_candidates(frames, proposals, grippers=None):
    """Track each appearance hypothesis independently across the full future."""
    n, h, w = frames.shape[:3]
    tracks = []
    hsvs = [cv2.cvtColor(f, cv2.COLOR_BGR2HSV) for f in frames]
    for p in proposals:
        centers = np.full((n, 2), np.nan)
        areas = np.zeros(n)
        confidence = np.zeros(n)
        last = p["origin"].copy()
        velocity = np.zeros(2)
        attached_side = None
        missing = 0
        for t, hsv in enumerate(hsvs):
            hue = np.abs(hsv[:, :, 0].astype(float) - p["color"][0])
            hue = np.minimum(hue, 180 - hue)
            if p["appearance_model"] == "hue":
                mask = (
                    (hue < 7)
                    & (hsv[:, :, 1] > max(65, p["color"][1] * 0.45))
                    & (hsv[:, :, 2] > max(12, p["color"][2] * 0.2))
                    & (hsv[:, :, 2] < min(256, p["color"][2] * 2 + 15))
                )
            else:
                mask = hsv[:, :, 2] < min(110, p["color"][2] + 45)
            mask[: int(h * 0.25)] = False
            options = []
            if grippers is not None and t > 0:
                distances = np.linalg.norm(grippers[t - 1] - last, axis=1)
                nearest = int(np.argmin(distances))
                if (
                    attached_side is None
                    and np.isfinite(distances[nearest])
                    and distances[nearest] < w * 0.12
                ):
                    attached_side = nearest
            predicted = last + velocity
            if attached_side is not None and t > 0:
                step = grippers[t, attached_side] - grippers[t - 1, attached_side]
                if np.isfinite(step).all() and np.linalg.norm(step) < w * 0.15:
                    predicted = last + step
            for region, stat, c in components(
                mask, max(8, p["area"] * 0.10), p["area"] * 3
            ):
                dist = np.linalg.norm(c - predicted)
                if dist > w * 0.18 + min(w * 0.4, missing * 3):
                    continue
                score = np.exp(-dist / (w * 0.12 + missing * 3)) * np.exp(
                    -abs(np.log(stat[4] / p["area"]))
                )
                if attached_side is not None:
                    distance = np.linalg.norm(c - grippers[t, attached_side])
                    score *= np.exp(
                        -max(0, distance - w * 0.10) / (w * 0.12 + missing * 3)
                    )
                options.append((score, c, stat[4]))
            if options:
                score, c, area = max(options, key=lambda z: z[0])
                if score > 0.12:
                    velocity = 0.6 * velocity + 0.4 * (c - last)
                    last = c
                    centers[t] = c
                    areas[t] = area
                    confidence[t] = score
                    missing = 0
                else:
                    missing += 1
            else:
                missing += 1
            if missing and attached_side is not None:
                last = predicted
        tracks.append(
            dict(proposal=p, centers=centers, areas=areas, confidence=confidence)
        )
    return tracks


def task_object_proposals(frames, task, config=InteractionConfig()):
    """Apply shared scene priors, never per-episode coordinates or object IDs."""
    kind = "dark" if task == "workpiece" else "saturated"
    proposals = object_proposals(frames, kind, config)
    h, w = frames.shape[1:3]
    if task == "drawer":
        hsv = cv2.cvtColor(frames[0], cv2.COLOR_BGR2HSV)
        red = (
            ((hsv[:, :, 0] < 12) | (hsv[:, :, 0] > 170))
            & (hsv[:, :, 1] > 120)
            & (hsv[:, :, 2] > 70)
        )
        boxes = components(red, int(h * w * 0.005))
        if not boxes:
            raise ValueError("drawer cabinet not discovered from scene")
        _, stat, _ = max(boxes, key=lambda x: x[1][4])
        y_limit = stat[1] + stat[3] + h * 0.15
        return [p for p in proposals if p["origin"][1] > y_limit and p["color"][2] > 60]
    return [
        p
        for p in proposals
        if w * 0.2 < p["origin"][0] < w * 0.8 and p["origin"][1] > h * 0.48
    ]
