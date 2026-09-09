"""Automatic articulated-foreground prompts from motion and entry-edge support."""

import numpy as np


def robot_entry_side(region):
    """Use a workspace entry, excluding people moving behind the tabletop."""
    h, w = region.shape
    workspace = region[int(h * 0.4) :]
    left = workspace[:, : max(1, int(np.ceil(w * 0.08)))].any()
    right = workspace[:, int(w * 0.92) :].any()
    return None if left == right else (0 if left else 1)


def robot_prompt(frames, geometry, side):
    n, h, w = frames.shape[:3]
    from .video import components

    best = None
    # Per-side motion labels may overlap and overwrite one another. Recover the
    # whole connected silhouette, assigning identity by its actual entry edge.
    for t, labels in enumerate(geometry["masks"]):
        for mask, stat, center in components(labels > 0, 150):
            x, y, bw, bh, area = stat
            if robot_entry_side(mask) != side:
                continue
            interior = np.exp(-abs(center[0] / w - (0.4 if side == 0 else 0.65)) * 6)
            score = area * interior
            if best is None or score > best[0]:
                best = score, t, mask
    if best is None:
        raise ValueError(
            "no entry-anchored articulated foreground supports robot discovery"
        )
    _, t, mask = best
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


def robot_mask_audit(robots, geometry, fps):
    """Check whole-arm support against independently observed motion silhouettes."""
    from .video import components
    from .evidence import stable_runs

    n, h, w = geometry["masks"].shape
    coverage = np.full((n, 2), np.nan)
    for t, labels in enumerate(geometry["masks"]):
        for region, stat, center in components(labels > 0, int(h * w * 0.015)):
            side = robot_entry_side(region)
            if side is None:
                continue
            observed = np.unpackbits(robots[t, side], axis=-1, count=w).astype(bool)
            value = float((observed & region).sum() / region.sum())
            if not np.isfinite(coverage[t, side]) or value < coverage[t, side]:
                coverage[t, side] = value
    results = []
    for side in [0, 1]:
        valid = np.isfinite(coverage[:, side])
        missing = valid & (coverage[:, side] < 0.6)
        fraction = float(missing.sum() / max(1, valid.sum()))
        longest = max((b - a for a, b in stable_runs(missing, 1)), default=0)
        results.append(
            dict(
                robot_id=["left", "right"][side],
                supported_frames=int(valid.sum()),
                insufficient_coverage_fraction=fraction,
                longest_insufficient_run=longest,
                passed=bool(
                    valid.sum() >= 5 and fraction <= 0.1 and longest <= fps * 0.5
                ),
            )
        )
    return dict(
        passed=all(r["passed"] for r in results),
        arms=results,
        method="entry_anchored_motion_coverage",
        minimum_coverage=0.6,
    )
