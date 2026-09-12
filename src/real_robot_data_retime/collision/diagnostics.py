"""Exact mesh distance at sampled poses, plus explicitly sampled violation spans."""

import numpy as np


def minimum_mesh_distance(checker, left, right):
    lm, lc, le, _, _ = left
    rm, rc, re, _, _ = right
    gap = np.maximum(np.abs(lc[:, None] - rc[None, :]) - le[:, None] - re[None, :], 0)
    bounds = np.maximum(
        np.linalg.norm(gap, axis=-1),
        np.linalg.norm(lc[:, None] - rc[None, :], axis=-1)
        - checker.radii[:, None]
        - checker.radii[None, :],
    )
    best = float("inf")
    pair = None
    for flat in np.argsort(bounds, axis=None):
        i, j = np.unravel_index(flat, bounds.shape)
        if bounds[i, j] >= best:
            break
        d = float(
            checker.fcl.distance(
                checker.geometry[i],
                checker.fcl.Transform3f(lm[i, :3, :3], lm[i, :3, 3]),
                checker.geometry[j],
                checker.fcl.Transform3f(rm[j, :3, :3], rm[j, :3, 3]),
                checker.fcl.DistanceRequest(),
                checker.fcl.DistanceResult(),
            )
        )
        if not np.isfinite(d):
            raise ValueError("nonfinite mesh distance")
        # FCL may return a sentinel for penetration, not a penetration depth.
        d = max(0.0, d)
        if d < best:
            best = d
            pair = (int(i), int(j))
    return best, pair


def violation_intervals(times, distances, threshold):
    times = np.asarray(times)
    distances = np.asarray(distances)
    if (
        times.ndim != 1
        or times.shape != distances.shape
        or not np.isfinite([times, distances]).all()
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError("expected ordered finite distance samples")
    bad = distances < threshold
    spans = np.flatnonzero(np.diff(np.r_[False, bad, False].astype(int))).reshape(-1, 2)
    return [
        dict(
            first_below_seconds=float(times[a]),
            last_below_seconds=float(times[b - 1]),
            minimum_distance_m=float(distances[a:b].min()),
            minimum_time_seconds=float(times[a + np.argmin(distances[a:b])]),
        )
        for a, b in spans
    ]
