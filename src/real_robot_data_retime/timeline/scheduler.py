from collections.abc import Callable
from dataclasses import dataclass
from heapq import heappop, heappush

import numpy as np


class NoSafeSchedule(ValueError):
    """No path satisfies the supplied clearance and precedence constraints."""


@dataclass(frozen=True)
class Schedule:
    left: np.ndarray
    right: np.ndarray
    left_waits: int
    right_waits: int


def schedule_sources(
    length_left: int,
    length_right: int,
    safe: Callable[[int, int, int, int], bool],
    *,
    dependency: Callable[[int, int], bool] = lambda i, j: True,
    left_priority: bool = True,
    can_wait: Callable[[int, int], bool] = lambda side, index: True,
    tie_break: Callable[[int, int], float] = lambda i, j: 0.0,
    remaining_lower_bound: Callable[[int, int], int] | None = None,
    earliest_right_admissions: tuple[int, ...] = (),
) -> Schedule:
    """Shortest monotone path through recorded poses, including swept edges.

    A* uses remaining source-frame counts as an admissible duration bound. It
    searches complete paths, so a tempting right pose with no safe future escape
    is rejected. Left priority forbids left waits before its trajectory ends.
    Optional right hold indices minimize their admission times lexicographically,
    then total duration. Each prefix cost counts time until that admission.
    Every edge advances one or both source indices; source poses are never skipped.
    `safe(i,j,ni,nj)` checks the complete transition, including both endpoints.
    """
    if min(length_left, length_right) < 1:
        raise ValueError("source trajectories must be nonempty")
    n, m = length_left, length_right
    remaining = remaining_lower_bound or (lambda i, j: max(n - 1 - i, m - 1 - j))
    goal = (n - 1, m - 1)
    if not dependency(*goal) or not safe(*goal, *goal):
        raise ValueError("terminal configuration is unsafe or violates dependency")
    if not dependency(0, 0):
        raise NoSafeSchedule(
            "no safe schedule preserving source trajectories and priority"
        )
    admissions = tuple(earliest_right_admissions)
    if (
        any(not 0 <= x < m - 1 for x in admissions)
        or tuple(sorted(set(admissions))) != admissions
    ):
        raise ValueError("admission indices must be ordered unique right hold indices")

    def heuristic(i, j):
        return tuple(max(0, stop + 1 - j) for stop in admissions) + (remaining(i, j),)

    def add(a, b):
        return tuple(x + y for x, y in zip(a, b))

    zero = (0,) * (len(admissions) + 1)
    frontier = [(heuristic(0, 0), tie_break(0, 0), 0, 0, 0, 0)]
    costs = {(0, 0): zero}
    parent = {}
    reached = False
    while frontier:
        priority, _, _, _, i, j = heappop(frontier)
        g = costs[i, j]
        if priority != add(g, heuristic(i, j)):
            continue
        if (i, j) == goal:
            reached = True
            break
        moves = [(1, 1), (1, 0)]
        if not left_priority or i == n - 1:
            moves.append((0, 1))
        for di, dj in moves:
            if (di == 0 and not can_wait(0, i)) or (dj == 0 and not can_wait(1, j)):
                continue
            ni, nj = i + di, j + dj
            if ni >= n or nj >= m or not dependency(ni, nj):
                continue
            new_cost = add(g, tuple(int(j <= stop) for stop in admissions) + (1,))
            if (ni, nj) in costs and new_cost >= costs[ni, nj]:
                continue
            if not safe(i, j, ni, nj):
                continue
            costs[ni, nj] = new_cost
            parent[ni, nj] = (i, j)
            lower_bound = heuristic(ni, nj)
            heappush(
                frontier,
                (add(new_cost, lower_bound), tie_break(ni, nj), -ni - nj, -nj, ni, nj),
            )
    if not reached:
        raise NoSafeSchedule(
            "no safe schedule preserving source trajectories and priority"
        )
    points = [goal]
    while points[-1] != (0, 0):
        points.append(parent[points[-1]])
    values = np.asarray(points[::-1], dtype=np.int64)
    return Schedule(
        values[:, 0],
        values[:, 1],
        int(np.sum(np.diff(values[:, 0]) == 0)),
        int(np.sum(np.diff(values[:, 1]) == 0)),
    )
