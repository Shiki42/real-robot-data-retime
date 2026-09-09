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
    def __init__(self, left, right, urdf: Path, mesh_root: Path,
                 margin_m=0.02, base_spacing_m=0.49):
        from robo_visualize.arms.piperx.model import PiperXModel
        import hppfcl
        if not np.isfinite([margin_m, base_spacing_m]).all() or margin_m <= 0 or base_spacing_m <= 0:
            raise ValueError('clearance and spacing must be positive finite metres')
        self.values = tuple(np.asarray(x, dtype=float) for x in (left, right))
        if any(x.ndim != 2 or x.shape[1] != 7 or not len(x) or not np.isfinite(x).all() for x in self.values):
            raise ValueError('expected finite nonempty N x 7 trajectories')
        self.fcl, self.margin = hppfcl, margin_m
        self.base_spacing_m = base_spacing_m
        with TemporaryDirectory() as temp:
            resolved = Path(temp) / 'model.urdf'
            resolved.write_text(Path(urdf).read_text().replace('package://', str(Path(mesh_root).resolve())+'/'))
            self.model = PiperXModel(resolved)
        self.geometry = [g.geometry for g in self.model.visual_model.geometryObjects]
        for g in self.geometry:
            g.computeLocalAABB()
        self.centers = np.array([g.aabb_center for g in self.geometry])
        self.radii = np.array([g.aabb_radius for g in self.geometry])
        self.poses = [[self._pose(row, side) for row in values] for side, values in enumerate(self.values)]

    def _pose(self, row, side):
        self.model.base_placement.translation = np.array([0., (0.5-side)*self.base_spacing_m, 0.])
        # Dataset gripper.pos is total jaw aperture in mm; URDF joint7/8
        # are individual opposing finger translations (each 35 mm maximum).
        finger_state = np.array(row, copy=True)
        finger_state[6] *= 0.5
        self.model.update(finger_state)
        matrices = np.array([self.model.transform_matrix(k) for k in range(len(self.geometry))])
        centers = np.einsum('nij,nj->ni', matrices[:, :3, :3], self.centers)+matrices[:, :3, 3]
        return matrices, centers

    def _clear(self, left, right):
        lm, lc = left
        rm, rc = right
        bounds = np.linalg.norm(lc[:,None]-rc[None,:], axis=-1)-self.radii[:,None]-self.radii[None,:]
        for a, b in np.argwhere(bounds < self.margin):
            ta = self.fcl.Transform3f(lm[a,:3,:3], lm[a,:3,3])
            tb = self.fcl.Transform3f(rm[b,:3,:3], rm[b,:3,3])
            result = self.fcl.DistanceResult()
            distance = self.fcl.distance(self.geometry[a], ta, self.geometry[b], tb,
                                         self.fcl.DistanceRequest(), result)
            if distance < self.margin:
                return False
        return True

    @lru_cache(maxsize=262144)
    def configuration_safe(self, i, j):
        return self._clear(self.poses[0][i], self.poses[1][j])

    def __call__(self, i, j, ni, nj):
        if not self.configuration_safe(i, j) or not self.configuration_safe(ni, nj):
            return False
        differences = [self.values[s][b]-self.values[s][a] for s,a,b in ((0,i,ni),(1,j,nj))]
        steps = max(1, int(np.ceil(max(np.max(np.abs(d)) for d in differences)/0.25)))
        for fraction in np.arange(1,steps)/steps:
            poses = [self._pose(self.values[s][a]+fraction*differences[s],s) for s,a in ((0,i),(1,j))]
            if not self._clear(*poses):
                return False
        return True
