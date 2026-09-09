from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np


class PiperXClearance:
    """Cross-arm mesh clearance using RoboVisualize FK and its exact meshes.

    Margin is in metres. Swept transitions are sampled at at most 0.25 degrees
    or 0.25 mm per joint. This is a discretized geometric audit, not a physical
    safety certificate. Scene and held-object constraints are supplied separately.
    """

    def __init__(
        self,
        left,
        right,
        urdf: Path,
        mesh_root: Path,
        margin_m=0.02,
        base_spacing_m=0.49,
    ):
        from robo_visualize.arms.piperx.model import PiperXModel, GRIPPER_STROKE_M
        import hppfcl

        if (
            not np.isfinite([margin_m, base_spacing_m]).all()
            or margin_m <= 0
            or base_spacing_m <= 0
        ):
            raise ValueError("clearance and spacing must be positive finite metres")
        self.values = tuple(np.asarray(x, dtype=float) for x in (left, right))
        if any(
            x.ndim != 2 or x.shape[1] != 7 or not len(x) or not np.isfinite(x).all()
            for x in self.values
        ):
            raise ValueError("expected finite nonempty N x 7 trajectories")
        self.fcl, self.margin = hppfcl, margin_m
        self.base_spacing_m = base_spacing_m
        with TemporaryDirectory() as temp:
            resolved = Path(temp) / "model.urdf"
            resolved.write_text(
                Path(urdf)
                .read_text()
                .replace("package://", str(Path(mesh_root).resolve()) + "/")
            )
            self.model = PiperXModel(resolved)
        self.gripper_stroke_m = GRIPPER_STROKE_M
        self.geometry = [g.geometry for g in self.model.visual_model.geometryObjects]
        for g in self.geometry:
            g.computeLocalAABB()
        self.centers = np.array([g.aabb_center for g in self.geometry])
        self.radii = np.array([g.aabb_radius for g in self.geometry])
        self.half_extents = np.array(
            [(g.aabb_local.max_ - g.aabb_local.min_) / 2 for g in self.geometry]
        )
        self.poses = [
            [self._pose(row, side) for row in values]
            for side, values in enumerate(self.values)
        ]

    def _pose(self, row, side):
        self.model.base_placement.translation = np.array(
            [0.0, (0.5 - side) * self.base_spacing_m, 0.0]
        )
        # Dataset gripper.pos is total jaw aperture in mm; URDF joint7/8
        # are individual opposing finger translations (each 35 mm maximum).
        finger_state = np.array(row, copy=True)
        finger_state[6] *= 0.5
        self.model.update(finger_state)
        matrices = np.array(
            [self.model.transform_matrix(k) for k in range(len(self.geometry))]
        )
        centers = (
            np.einsum("nij,nj->ni", matrices[:, :3, :3], self.centers)
            + matrices[:, :3, 3]
        )
        extents = np.einsum(
            "nij,nj->ni", np.abs(matrices[:, :3, :3]), self.half_extents
        )
        limit_excess = max(
            0.0, float(row[6]) / 2000 - self.gripper_stroke_m, -float(row[6]) / 2000
        )
        return matrices, centers, extents, limit_excess

    def _clear(self, left, right):
        lm, lc, le, left_excess = left
        rm, rc, re, right_excess = right
        margin = self.margin + left_excess + right_excess
        bounds = (
            np.linalg.norm(lc[:, None] - rc[None, :], axis=-1)
            - self.radii[:, None]
            - self.radii[None, :]
        )
        aabb_gap = np.maximum(
            np.abs(lc[:, None] - rc[None, :]) - le[:, None] - re[None, :], 0.0
        )
        close = (bounds < margin) & (np.linalg.norm(aabb_gap, axis=-1) < margin)
        for a, b in np.argwhere(close):
            ta = self.fcl.Transform3f(lm[a, :3, :3], lm[a, :3, 3])
            tb = self.fcl.Transform3f(rm[b, :3, :3], rm[b, :3, 3])
            result = self.fcl.DistanceResult()
            distance = self.fcl.distance(
                self.geometry[a],
                ta,
                self.geometry[b],
                tb,
                self.fcl.DistanceRequest(),
                result,
            )
            if distance < margin:
                return False
        return True

    @lru_cache(maxsize=262144)
    def configuration_safe(self, i, j):
        return self._clear(self.poses[0][i], self.poses[1][j])

    def __call__(self, i, j, ni, nj):
        if not self.configuration_safe(i, j) or not self.configuration_safe(ni, nj):
            return False
        differences = [
            self.values[s][b] - self.values[s][a]
            for s, a, b in ((0, i, ni), (1, j, nj))
        ]
        steps = max(1, int(np.ceil(max(np.max(np.abs(d)) for d in differences) / 0.25)))
        for fraction in np.arange(1, steps) / steps:
            poses = [
                self._pose(self.values[s][a] + fraction * differences[s], s)
                for s, a in ((0, i), (1, j))
            ]
            if not self._clear(*poses):
                return False
        return True

    def arm_clears_volume(self, side, source_index, volume, margin=0.02):
        """Exact arm meshes against a conservative oriented scene volume."""
        rotation, lo, hi = volume
        geometry = self.fcl.Box(*(hi - lo))
        placement = self.fcl.Transform3f(rotation, rotation @ ((lo + hi) / 2))
        matrices = self.poses[side][source_index][0]
        margin += self.poses[side][source_index][3]
        for mesh, matrix in zip(self.geometry, matrices):
            pose = self.fcl.Transform3f(matrix[:3, :3], matrix[:3, 3])
            distance = self.fcl.distance(
                mesh,
                pose,
                geometry,
                placement,
                self.fcl.DistanceRequest(),
                self.fcl.DistanceResult(),
            )
            if distance < margin:
                return False
        return True
