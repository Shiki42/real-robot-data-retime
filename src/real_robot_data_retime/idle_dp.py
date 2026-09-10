"""Globally choose non-overlapping bounded idle shortcuts, with source guards."""

import numpy as np


def compact_idle_optimal(q, a, fps, qt, at, protected, config):
    q = np.asarray(q, float)
    a = np.asarray(a, float)
    protected = np.asarray(protected, bool)
    n = len(q)
    if n < 2:
        return np.arange(n, dtype=np.int64), []
    guard = max(2, round(config.retained_idle_seconds * fps / 2))
    minimum = max(2 * guard + 1, round(config.minimum_idle_seconds * fps))
    qv = np.maximum(np.abs(np.diff(q, axis=0)).max(0), 1e-9)
    av = np.maximum(np.abs(np.diff(a, axis=0)).max(0), 1e-9)
    qa = np.maximum(np.abs(np.diff(q, n=2, axis=0)).max(0), 1e-9) if n > 2 else qv * 2
    aa = np.maximum(np.abs(np.diff(a, n=2, axis=0)).max(0), 1e-9) if n > 2 else av * 2
    next_protected = np.full(n, n, int)
    boundary = n
    for i in range(n - 1, -1, -1):
        if protected[i]:
            boundary = i
        next_protected[i] = boundary
    cost = np.arange(n, -1, -1, dtype=np.int64)
    choice = np.arange(1, n + 1, dtype=np.int64)
    for i in range(n - 1, -1, -1):
        cost[i] = 1 + cost[i + 1]
        limit = int(next_protected[i])
        if limit - i < minimum:
            continue
        qr = np.maximum.accumulate(q[i:limit], axis=0) - np.minimum.accumulate(
            q[i:limit], axis=0
        )
        ar = np.maximum.accumulate(a[i:limit], axis=0) - np.minimum.accumulate(
            a[i:limit], axis=0
        )
        bad = np.flatnonzero(
            np.any(qr > qt * 1.000001, axis=1) | np.any(ar > at * 1.000001, axis=1)
        )
        stop = i + int(bad[0]) if len(bad) else limit
        ends = np.arange(i + minimum, stop + 1)
        if not len(ends):
            continue
        left = i + guard - 1
        right = ends - guard
        dq = q[right] - q[left]
        da = a[right] - a[left]
        ok = (np.abs(dq) <= qv * 1.000001).all(1) & (np.abs(da) <= av * 1.000001).all(1)
        ok &= (np.abs(dq - (q[left] - q[left - 1])) <= qa * 1.000001).all(1)
        ok &= (np.abs((q[right + 1] - q[right]) - dq) <= qa * 1.000001).all(1)
        ok &= (np.abs(da - (a[left] - a[left - 1])) <= aa * 1.000001).all(1)
        ok &= (np.abs((a[right + 1] - a[right]) - da) <= aa * 1.000001).all(1)
        ends = ends[ok]
        if not len(ends):
            continue
        values = 2 * guard + cost[ends]
        k = int(np.argmin(values))
        if values[k] < cost[i]:
            cost[i] = values[k]
            choice[i] = ends[k]
    out = []
    runs = []
    i = 0
    while i < n:
        end = int(choice[i])
        if end == i + 1:
            out.append(i)
        else:
            out.extend(range(i, i + guard))
            out.extend(range(end - guard, end))
            runs.append((i, end))
        i = end
    return np.asarray(out, dtype=np.int64), runs
