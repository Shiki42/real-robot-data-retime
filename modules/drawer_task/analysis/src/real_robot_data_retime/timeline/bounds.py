"""Admissible critical-path bounds for the drawer precedence graph."""

from functools import lru_cache


def drawer_remaining_bound(n, m, gate, opening, withdrawal, closing, markers=()):
    def basic(i, j):
        value = max(n - 1 - i, m - 1 - j)
        ready = max(0, withdrawal - i)
        if i < gate and j < opening:
            arrival = max(gate - i, opening - j)
            value = max(value, arrival + n - 1 - gate)
            ready = max(ready, arrival + withdrawal - gate)
        if j < closing:
            value = max(value, max(ready, closing - j) + m - 1 - closing)
        return value

    @lru_cache(None)
    def remaining(i, j):
        elapsed = 0
        a, b = i, j
        for u, v in markers:
            if a >= u and b >= v:
                continue
            elapsed += max(u - a, v - b)
            a, b = u, v
        return max(basic(i, j), elapsed + basic(a, b))

    return remaining
