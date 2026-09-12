"""Conservative mesh separation between piecewise-linear joint samples."""

from functools import lru_cache
import numpy as np

from .piperx import PiperXClearance


class ContinuousClearance:
    """Require > clearance, reserving a Lipschitz bound between sampled poses.

    With sample travel <= 2 * slack, every point is within slack of a checked
    sample. Checked separation includes slack and a strict numerical guard.
    The model's existing reach/angle/aperture bound covers relative mesh motion.
    """

    def __init__(self, left, right, urdf, mesh_root, clearance_m=0.02, slack_m=0.001):
        if (
            not np.isfinite([clearance_m, slack_m]).all()
            or min(clearance_m, slack_m) <= 0
        ):
            raise ValueError("clearance and slack must be positive finite metres")
        self.clearance_m = clearance_m
        self.slack_m = slack_m
        self.checker = PiperXClearance(
            left, right, urdf, mesh_root, margin_m=clearance_m + slack_m + 1e-6
        )
        self.checked_samples = 0

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
        steps = max(1, int(np.ceil(bound / (2 * self.slack_m))))
        for k in range(1, steps):
            self.checked_samples += 1
            if not c._clear(
                c._interpolated_pose(0, i, ni, k, steps),
                c._interpolated_pose(1, j, nj, k, steps),
            ):
                return False
        return True
