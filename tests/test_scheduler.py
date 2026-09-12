import itertools

import numpy as np
import pytest

from real_robot_data_retime.timeline.scheduler import schedule_sources


def test_independent_runs_at_full_speed():
    plan = schedule_sources(4, 6, lambda *edge: True)
    assert plan.left.tolist() == [0, 1, 2, 3, 3, 3]
    assert plan.right.tolist() == list(range(6))


def test_right_waits_before_entering_future_collision():
    # Right pose 1 is safe now but cannot remain there while left passes pose 2.
    forbidden = {(2, 1), (2, 2)}

    def safe(i, j, ni, nj):
        return (i, j) not in forbidden and (ni, nj) not in forbidden

    plan = schedule_sources(5, 4, safe)
    assert plan.left.tolist() == [0, 1, 2, 3, 4, 4]
    assert plan.right.tolist() == [0, 0, 0, 1, 2, 3]


def test_drawer_gate_allows_cube_to_approach_then_wait():
    plan = schedule_sources(
        6,
        7,
        lambda *e: True,
        dependency=lambda i, j: i <= 2 or j >= 5,
        left_priority=False,
    )
    assert np.all((plan.left <= 2) | (plan.right >= 5))
    assert len(plan.left) == 8
    assert 2 in plan.left[2:5]


def test_endpoint_clear_but_swept_collision_rejected():
    def safe(i, j, ni, nj):
        return not (ni > i and nj > j)

    plan = schedule_sources(3, 3, safe)
    assert len(plan.left) == 5
    assert not np.any((np.diff(plan.left) > 0) & (np.diff(plan.right) > 0))


def test_no_safe_left_priority_path_fails_explicitly():
    with pytest.raises(ValueError, match="no safe schedule"):
        schedule_sources(3, 2, lambda i, j, ni, nj: ni != 1)


def test_astar_matches_exhaustive_small_schedules():
    from collections import deque

    rng = np.random.default_rng(23)
    for priority in [False, True]:
        for _ in range(30):
            n, m = 4, 4
            blocked = rng.random((n, m)) < 0.2
            blocked[0, 0] = False
            blocked[-1, -1] = False

            def safe(i, j, ni, nj, blocked=blocked):
                return not blocked[i, j] and not blocked[ni, nj]

            queue = deque([(0, 0, 0)])
            seen = {(0, 0)}
            optimum = None
            while queue:
                i, j, cost = queue.popleft()
                if (i, j) == (n - 1, m - 1):
                    optimum = cost + 1
                    break
                for di, dj in [(1, 1), (1, 0), (0, 1)]:
                    if priority and di == 0 and i < n - 1:
                        continue
                    a, b = i + di, j + dj
                    if a < n and b < m and (a, b) not in seen and safe(i, j, a, b):
                        seen.add((a, b))
                        queue.append((a, b, cost + 1))
            if optimum is None:
                with pytest.raises(ValueError, match="no safe schedule"):
                    schedule_sources(n, m, safe, left_priority=priority)
            else:
                plan = schedule_sources(n, m, safe, left_priority=priority)
                assert len(plan.left) == optimum


def test_earliest_admission_can_prefer_longer_complete_schedule():
    fast = [(0, 0), (1, 1), (2, 1), (3, 2), (4, 3), (5, 4)]
    early = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (3, 2), (4, 3), (5, 4)]
    edges = {(*a, *b) for path in (fast, early) for a, b in itertools.pairwise(path)}
    edges.add((5, 4, 5, 4))
    safe = lambda i, j, ni, nj: (i, j, ni, nj) in edges
    shortest = schedule_sources(6, 5, safe, left_priority=False)
    earliest = schedule_sources(
        6, 5, safe, left_priority=False, earliest_right_admissions=(1,)
    )
    assert len(shortest.left) == len(fast)
    assert list(zip(earliest.left, earliest.right)) == early


def test_earliest_admission_rejects_dead_end():
    path = [(0, 0), (1, 1), (2, 1), (3, 2), (4, 3), (5, 4)]
    edges = {(*a, *b) for a, b in itertools.pairwise(path)}
    edges.update([(0, 0, 0, 1), (0, 1, 0, 2), (5, 4, 5, 4)])
    result = schedule_sources(
        6,
        5,
        lambda *edge: edge in edges,
        left_priority=False,
        earliest_right_admissions=(1,),
    )
    assert list(zip(result.left, result.right)) == path
