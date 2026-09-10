"""Optional idle shortcuts: original source poses remain available to the search."""

import time
from heapq import heappop, heappush

import numpy as np


def edges_from_runs(runs, fps, retained_seconds):
    guard = max(2, round(retained_seconds * fps / 2))
    edges = {}
    for lo, hi in runs:
        a, b = lo + guard - 1, hi - guard
        if b > a + 1:
            edges.setdefault(a, []).append(b)
    return edges


def optional_schedule(
    n,
    left_edges,
    right_edges,
    common,
    dependency,
    safe,
    left_wait,
    right_wait,
    max_nodes=400000,
):
    raw = np.asarray(common)
    if raw.ndim != 1 or raw.size < 2 or raw.dtype.kind not in "iu":
        raise ValueError("Baseline must be a nonempty integer source path")
    common = raw.astype(np.int64)
    if n < 2 or common[0] != 0 or common[-1] != n - 1 or np.any(np.diff(common) <= 0):
        raise ValueError("Invalid baseline")
    left_wait = np.asarray(left_wait, bool)
    right_wait = np.asarray(right_wait, bool)
    if left_wait.shape != (n,) or right_wait.shape != (n,):
        raise ValueError("Wait masks must match the source length")
    if max_nodes < 1:
        raise ValueError("Node budget must be positive")
    if any(not dependency(int(t), int(t)) for t in common):
        raise ValueError("Baseline violates dependency constraints")
    if any(
        not safe(int(a), int(a), int(b), int(b))
        for a, b in zip(common[:-1], common[1:])
    ):
        raise ValueError("Baseline violates transition constraints")
    for edges in [left_edges, right_edges]:
        if any(not 0 <= i < j < n for i, ends in edges.items() for j in ends):
            raise ValueError("Invalid shortcut")

    def lower(edges):
        d = np.zeros(n, dtype=int)
        for i in range(n - 2, -1, -1):
            d[i] = 1 + min([d[i + 1]] + [d[j] for j in edges.get(i, [])])
        return d

    lh, rh = lower(left_edges), lower(right_edges)
    bound = len(common) - 1
    goal = (n - 1, n - 1)
    cost = {(0, 0): 0}
    parent = {}
    serial = 0
    heap = [(max(lh[0], rh[0]), 0, 0, serial, 0, 0)]
    nodes = 0
    tic = time.time()
    found = False
    while heap and nodes < max_nodes:
        f, negprogress, skew, _, l, r = heappop(heap)
        g = cost[l, r]
        if f != g + max(lh[l], rh[r]):
            continue
        if f >= bound:
            break
        nodes += 1
        if (l, r) == goal:
            found = True
            break
        ls = (
            ([l] if left_wait[l] else [])
            + ([l + 1] if l < n - 1 else [])
            + left_edges.get(l, [])
        )
        rs = (
            ([r] if right_wait[r] else [])
            + ([r + 1] if r < n - 1 else [])
            + right_edges.get(r, [])
        )
        for nl in ls:
            for nr in rs:
                if (nl, nr) == (l, r) or not dependency(nl, nr):
                    continue
                ng = g + 1
                nf = ng + max(lh[nl], rh[nr])
                if nf >= bound or ng >= cost.get((nl, nr), n * n):
                    continue
                if not safe(l, r, nl, nr):
                    continue
                cost[nl, nr] = ng
                parent[nl, nr] = (l, r)
                serial += 1
                heappush(heap, (nf, -nl - nr, abs(nl - nr), serial, nl, nr))
    stats = {
        "expanded_nodes": nodes,
        "seconds": time.time() - tic,
        "node_budget_exhausted": nodes >= max_nodes,
        "search_scope": "optional per-arm idle shortcuts; contact and supplied projection constraints",
        "found_strict_improvement": found,
    }
    if not found:
        return common, common, stats
    points = [goal]
    while points[-1] != (0, 0):
        points.append(parent[points[-1]])
    x = np.asarray(points[::-1])
    return x[:, 0], x[:, 1], stats
