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
    tracks = [
        dict(
            proposal=p,
            centers=centers[k],
            areas=areas[k],
            confidence=np.minimum(areas[k] / max(1, p["area"]), 1.0),
        )
        for k, p in enumerate(proposals)
    ]
    return tracks


def segment_grippers(frames, geometry, sam=None):
    sam = sam or SamVideo()
    n, h, w = frames.shape[:3]
    centers = np.full((n, 2, 2), np.nan)
    apertures = np.full((n, 2), np.nan)
    seeds = []
    for side in [0, 1]:
        xy = geometry["centers"][:, side]
        area = np.sum(geometry["masks"] == side + 1, axis=(1, 2))
        central = np.abs(xy[:, 0] / w - (0.4 if side == 0 else 0.65))
        score = np.sqrt(area) * np.exp(-central * 8)
        if not np.isfinite(score).any() or np.nanmax(score) <= 0:
            seeds.append(None)
            continue
        t = int(np.nanargmax(score))
        x, y = xy[t]
        radius = w * 0.095
        x0, y0 = max(0, x - radius), max(0, y - radius)
        box = [x0, y0, min(w, x + radius) - x0, min(h, y + radius) - y0]
        seeds.append(dict(frame=t, bbox=box))
        for reverse in [False, True]:
            for idx, masks in sam.propagate(
                frames, [{"bbox": box}], seed_frame=t, reverse=reverse
            ):
                yy, xx = np.where(masks[0])
                if len(xx) < 10:
                    continue
                quantile = np.percentile(xx, 90 if side == 0 else 10)
                distal = xx >= quantile if side == 0 else xx <= quantile
                centers[idx, side] = [np.median(xx[distal]), np.median(yy[distal])]
                apertures[idx, side] = np.percentile(yy[distal], 90) - np.percentile(
                    yy[distal], 10
                )
    return dict(centers=centers, apertures=apertures, seeds=seeds)
