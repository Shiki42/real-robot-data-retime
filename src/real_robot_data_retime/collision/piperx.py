from pathlib import Path
from collections import OrderedDict
from math import gcd
from tempfile import TemporaryDirectory
import numpy as np


class PiperXClearance:
    """Cross-arm mesh clearance using RoboVisualize FK and its exact meshes.

    Margin is in metres. A conservative reach/angle motion bound spaces samples
    so between-sample relative travel is at most half the margin. This audits
    the supplied geometric model, not physical calibration accuracy. Scene and
    held-object constraints are supplied separately.
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
        self.max_reach_m = sum(
            np.linalg.norm(j.translation) for j in self.model.model.jointPlacements
        ) + max(
            np.linalg.norm(g.placement.translation)
            + np.linalg.norm(g.geometry.aabb_center)
            + g.geometry.aabb_radius
            for g in self.model.visual_model.geometryObjects
        )
        self._configuration_cache = {}
        self._interpolation_cache = OrderedDict()
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
        return (
            matrices,
            centers,
            extents,
            limit_excess,
            self.model.tcp_transform().translation.copy(),
        )

    def _clear(self, left, right, *, margin_m=None):
        lm, lc, le, left_excess, _ = left
        rm, rc, re, right_excess, _ = right
        margin = (
            (self.margin if margin_m is None else margin_m) + left_excess + right_excess
        )
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
            if not np.isfinite(distance):
                raise ValueError("non-finite mesh distance")
            if distance < margin:
                return False
        return True

    def configuration_safe(self, i, j):
        key = (i, j)
        if key not in self._configuration_cache:
            self._configuration_cache[key] = self._clear(
                self.poses[0][i], self.poses[1][j]
            )
        return self._configuration_cache[key]

    def __call__(self, i, j, ni, nj):
        if not self.configuration_safe(i, j) or not self.configuration_safe(ni, nj):
            return False
        differences = [
            self.values[s][b] - self.values[s][a]
            for s, a, b in ((0, i, ni), (1, j, nj))
        ]
        bounds = [
            np.abs(d[:6]).sum() * np.pi / 180 * self.max_reach_m + abs(d[6]) / 2000
            for d in differences
        ]
        motion_bound = sum(bounds)
        left_pose, right_pose = self.poses[0][i], self.poses[1][j]
        lc, le = left_pose[1:3]
        rc, re = right_pose[1:3]
        separation = np.maximum(
            np.abs(lc[:, None] - rc[None, :])
            - le[:, None]
            - re[None, :]
            - motion_bound,
            0.0,
        )
        if np.all(
            np.linalg.norm(separation, axis=-1)
            >= self.margin + left_pose[3] + right_pose[3]
        ):
            return True
        steps = max(1, int(np.ceil(motion_bound / (self.margin * 0.5))))
        for k in range(1, steps):
            poses = [
                self._interpolated_pose(side, a, b, k, steps)
                for side, a, b in [(0, i, ni), (1, j, nj)]
            ]
            if not self._clear(*poses):
                return False
        return True

    def _interpolated_pose(self, side, start, stop, numerator, denominator):
        if start == stop or numerator == 0:
            return self.poses[side][start]
        if numerator == denominator:
            return self.poses[side][stop]
        divisor = gcd(numerator, denominator)
        key = (side, start, stop, numerator // divisor, denominator // divisor)
        if key in self._interpolation_cache:
            self._interpolation_cache.move_to_end(key)
            return self._interpolation_cache[key]
        fraction = numerator / denominator
        row = self.values[side][start] + fraction * (
            self.values[side][stop] - self.values[side][start]
        )
        pose = self._pose(row, side)
        self._interpolation_cache[key] = pose
        if len(self._interpolation_cache) > 65536:
            self._interpolation_cache.popitem(last=False)
        return pose

    def arm_clears_volume(self, side, source_index, volume, margin=0.02):
        """Exact arm meshes against a conservative oriented scene volume."""
        return self.pose_clears_volume(self.poses[side][source_index], volume, margin)

    def pose_clears_volume(self, pose, volume, margin):
        rotation, lo, hi = volume
        matrices, centers, extents, excess, _ = pose
        margin += excess
        box_center = rotation @ ((lo + hi) / 2)
        box_extent = np.abs(rotation) @ ((hi - lo) / 2)
        gap = np.maximum(np.abs(centers - box_center) - extents - box_extent, 0.0)
        candidates = np.flatnonzero(np.linalg.norm(gap, axis=1) < margin)
        if not len(candidates):
            return True
        geometry = self.fcl.Box(*(hi - lo))
        placement = self.fcl.Transform3f(rotation, box_center)
        for index in candidates:
            mesh, matrix = self.geometry[index], matrices[index]
            pose = self.fcl.Transform3f(matrix[:3, :3], matrix[:3, 3])
            distance = self.fcl.distance(
                mesh,
                pose,
                geometry,
                placement,
                self.fcl.DistanceRequest(),
                self.fcl.DistanceResult(),
            )
            if not np.isfinite(distance):
                raise ValueError("non-finite mesh distance")
            if distance < margin:
                return False
        return True
