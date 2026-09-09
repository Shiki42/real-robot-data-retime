"""Automatic articulated-foreground prompts from motion and entry-edge support."""

import numpy as np


def robot_prompt(frames, geometry, side):
    n, h, w = frames.shape[:3]
    areas = np.sum(geometry["masks"] == side + 1, axis=(1, 2))
    center = geometry["centers"][:, side]
    interior = np.exp(-abs(center[:, 0] / w - (0.4 if side == 0 else 0.65)) * 6)
    scores = areas * interior
    if not np.isfinite(scores).any() or np.nanmax(scores) <= 0:
        raise ValueError("no articulated foreground supports robot discovery")
    t = int(np.nanargmax(scores))
    mask = geometry["masks"][t] == side + 1
    proposal = prompt_from_robot_region(mask)
    other = (geometry["masks"][t] == 2 - side) & ~mask
    yy, xx = np.where(other)
    negatives = []
    if len(xx) > 20:
        for quantile in [0.25, 0.5, 0.75]:
            col = np.quantile(xx, quantile)
            near = abs(xx - col) < 4
            negatives.append([float(np.median(xx[near])), float(np.median(yy[near]))])
    proposal["negative_points"] = negatives
    return t, proposal


def prompt_from_robot_region(mask):
    h, w = mask.shape
    yy, xx = np.where(mask)
    x0, y0 = max(0, int(xx.min()) - 12), max(0, int(yy.min()) - 15)
    x1, y1 = min(w, int(xx.max()) + 12), min(h, int(yy.max()) + 15)
    positives = []
    for quantile in [0.05, 0.25, 0.5, 0.75, 0.95]:
        col = int(np.quantile(xx, quantile))
        local = mask[:, max(0, col - 3) : min(w, col + 4)].sum(axis=1) > 2
        runs = np.flatnonzero(np.diff(np.r_[False, local, False].astype(int))).reshape(
            -1, 2
        )
        if len(runs):
            a, b = max(runs, key=lambda z: z[1] - z[0])
            positives.append([float(col), float((a + b - 1) / 2)])
    return dict(bbox=[x0, y0, x1 - x0, y1 - y0], mask=mask, positive_points=positives)
