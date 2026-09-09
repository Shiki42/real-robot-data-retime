from dataclasses import dataclass
from typing import Callable
import numpy as np


@dataclass(frozen=True)
class Schedule:
    left: np.ndarray
    right: np.ndarray
    left_waits: int
    right_waits: int


def schedule_sources(length_left: int, length_right: int,
                     safe: Callable[[int, int, int, int], bool], *,
                     dependency: Callable[[int, int], bool] = lambda i, j: True,
                     left_priority: bool = True) -> Schedule:
    """Shortest monotone path through original poses, including swept edges.

    Never skip source frames or invent poses. Under left priority the left arm
    advances every frame until finished; backward reachability chooses right
    waiting positions that remain safe throughout the left arm's future motion.
    With task dependencies either arm may wait, and duration is minimized.
    `safe(i,j,ni,nj)` must check the complete transition, including endpoints.
    """
    if min(length_left, length_right) < 1:
        raise ValueError('source trajectories must be nonempty')
    n, m = length_left, length_right
    unreachable = np.iinfo(np.int32).max // 2
    cost = np.full((n, m), unreachable, dtype=np.int32)
    choice = np.zeros((n, m), dtype=np.uint8)
    if not dependency(n-1, m-1) or not safe(n-1, m-1, n-1, m-1):
        raise ValueError('terminal configuration is unsafe or violates dependency')
    cost[-1, -1] = 0
    for i in range(n-1, -1, -1):
        for j in range(m-1, -1, -1):
            if (i == n-1 and j == m-1) or not dependency(i, j):
                continue
            # Ties favor simultaneous progress, then left-only progress.
            edges = [(1, 1, 1), (1, 0, 2)]
            if not left_priority or i == n-1:
                edges.append((0, 1, 3))
            for di, dj, code in edges:
                ni, nj = i+di, j+dj
                if ni >= n or nj >= m or cost[ni, nj] == unreachable:
                    continue
                candidate = 1 + cost[ni, nj]
                if candidate < cost[i, j] and safe(i, j, ni, nj):
                    cost[i, j], choice[i, j] = candidate, code
    if cost[0, 0] == unreachable:
        raise ValueError('no safe schedule preserving source trajectories and priority')
    points = [(0, 0)]
    increments = {1: (1, 1), 2: (1, 0), 3: (0, 1)}
    while points[-1] != (n-1, m-1):
        i, j = points[-1]
        di, dj = increments[int(choice[i, j])]
        points.append((i+di, j+dj))
    values = np.asarray(points, dtype=np.int64)
    return Schedule(values[:, 0], values[:, 1],
                    int(np.sum(np.diff(values[:, 0]) == 0)),
                    int(np.sum(np.diff(values[:, 1]) == 0)))
