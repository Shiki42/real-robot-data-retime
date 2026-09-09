import numpy as np
from real_robot_data_retime.timeline.bounds import drawer_remaining_bound


def test_critical_path_bound_never_exceeds_exhaustive_remaining_distance():
    rng = np.random.default_rng(102)
    for _ in range(100):
        n, m = rng.integers(6, 13, size=2)
        gate = int(rng.integers(1, n - 2))
        withdrawal = int(rng.integers(gate + 1, n))
        opening = int(rng.integers(1, m - 2))
        closing = int(rng.integers(opening + 1, m))
        markers = []
        if rng.random() < 0.5:
            markers = [
                (
                    int(rng.integers(gate, withdrawal)),
                    int(rng.integers(opening, closing)),
                )
            ]

        def valid(i, j):
            return (
                (i < gate or j >= opening)
                and (j < closing or i >= withdrawal)
                and all(
                    not ((i < u and j > v) or (i > u and j < v) or (i == u and j != v))
                    for u, v in markers
                )
            )

        distance = np.full((n, m), np.inf)
        distance[-1, -1] = 0
        for i in range(n - 1, -1, -1):
            for j in range(m - 1, -1, -1):
                if (i, j) == (n - 1, m - 1) or not valid(i, j):
                    continue
                for di, dj in [(1, 1), (1, 0), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if ni < n and nj < m and valid(ni, nj):
                        distance[i, j] = min(distance[i, j], 1 + distance[ni, nj])
        bound = drawer_remaining_bound(
            n, m, gate, opening, withdrawal, closing, markers
        )
        for i, j in np.ndindex(n, m):
            if valid(i, j) and np.isfinite(distance[i, j]):
                assert 0 <= bound(i, j) <= distance[i, j]
