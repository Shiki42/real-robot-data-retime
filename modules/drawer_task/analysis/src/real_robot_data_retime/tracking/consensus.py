"""Spatial agreement for point observations; never fills an occluded trajectory."""

import numpy as np


def spatial_consensus(
    xy, visible, maximum_diameter, minimum_visible_fraction=0.6, minimum_points=3
):
    if (
        xy.shape != (*visible.shape, 2)
        or maximum_diameter <= 0
        or not 0 < minimum_visible_fraction <= 1
        or minimum_points < 3
    ):
        raise ValueError("invalid point consensus geometry")
    accepted = np.zeros_like(visible, dtype=bool)
    centers = np.full((len(xy), 2), np.nan)
    for t, points in enumerate(xy):
        ids = np.flatnonzero(visible[t] & np.isfinite(points).all(axis=-1))
        minimum = max(minimum_points, int(np.ceil(len(ids) * minimum_visible_fraction)))
        if len(ids) < minimum:
            continue
        distance = np.linalg.norm(points[ids, None] - points[None, ids], axis=-1)
        # A radius of half the diameter bounds every pair in the retained set.
        support = distance <= maximum_diameter / 2
        cluster = support[np.argmax(support.sum(axis=1))]
        if cluster.sum() < minimum:
            continue
        accepted[t, ids[cluster]] = True
        centers[t] = np.median(points[ids[cluster]], axis=0)
    filtered = np.asarray(xy).copy()
    filtered[~accepted] = np.nan
    return filtered, accepted, centers
