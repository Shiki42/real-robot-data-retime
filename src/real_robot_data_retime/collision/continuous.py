"""Conservative mesh separation between piecewise-linear joint samples."""

from functools import lru_cache
import numpy as np

from .piperx import PiperXClearance


class ContinuousClearance:
    """Adaptive midpoint certificates for piecewise-linear joint interpolation.

    A midpoint farther than threshold plus half the segment motion bound
    certifies its whole segment. Unresolved numerical-limit cases are rejected.
    """

    def __init__(self, left, right, urdf, mesh_root, clearance_m=0.0155):
        if not np.isfinite(clearance_m) or clearance_m <= 0:
            raise ValueError("clearance must be positive finite metres")
        self.clearance_m = clearance_m
        self.checker = PiperXClearance(
            left, right, urdf, mesh_root, margin_m=clearance_m + 1e-6
        )
        self.checked_samples = 0

    @classmethod
    def on_source_clocks(
        cls, base, clocks, pose_cache, configuration_cache, clearance_m
    ):
        """Reuse original-source FK and pair checks across timing candidates."""
        from copy import copy
        from collections import OrderedDict
        from ..timeline.smooth import sample_rows

        result = cls.__new__(cls)
        result.clearance_m = clearance_m
        result.checked_samples = 0
        c = copy(base)
        c.margin = clearance_m + 1e-6
        c.values = tuple(sample_rows(v, clock) for v, clock in zip(base.values, clocks))
        c.poses = []
        for side, clock in enumerate(clocks):
            arm = []
            for source, row in zip(clock, c.values[side]):
                source = float(source)
                if source not in pose_cache[side]:
                    pose_cache[side][source] = base._pose(row, side)
                arm.append(pose_cache[side][source])
            c.poses.append(arm)

        def configuration_safe(i, j):
            key = (float(clocks[0][i]), float(clocks[1][j]))
            if key not in configuration_cache:
                configuration_cache[key] = c._clear(c.poses[0][i], c.poses[1][j])
            return configuration_cache[key]

        c.configuration_safe = configuration_safe
        c._interpolation_cache = OrderedDict()
        result.checker = c
        return result

    @lru_cache(maxsize=65536)
    def __call__(self, i, j, ni, nj):
        c = self.checker
        if not c.configuration_safe(i, j) or not c.configuration_safe(ni, nj):
            return False
        differences = [
            c.values[s][b] - c.values[s][a] for s, a, b in [(0, i, ni), (1, j, nj)]
        ]
        bound = sum(
            np.abs(d[:6]).sum() * np.pi / 180 * c.max_reach_m + abs(d[6]) / 2000
            for d in differences
        )
        lc, le = c.poses[0][i][1:3]
        rc, re = c.poses[1][j][1:3]
        gap = np.maximum(
            np.abs(lc[:, None] - rc[None, :]) - le[:, None] - re[None, :] - bound, 0
        )
        if np.all(
            np.linalg.norm(gap, axis=-1)
            > c.margin + c.poses[0][i][3] + c.poses[1][j][3]
        ):
            return True
        stack = [(0, 0)]
        while stack:
            index, depth = stack.pop()
            denominator = 2 ** (depth + 1)
            numerator = 2 * index + 1
            left = c._interpolated_pose(0, i, ni, numerator, denominator)
            right = c._interpolated_pose(1, j, nj, numerator, denominator)
            self.checked_samples += 1
            if c._clear(left, right, margin_m=c.margin + bound / denominator):
                continue
            if not c._clear(left, right) or depth >= 24:
                return False
            stack.extend([(2 * index, depth + 1), (2 * index + 1, depth + 1)])
        return True
