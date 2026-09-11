"""Conservative drawer sweep proxy inferred from handle TCP motion."""

import numpy as np


from dataclasses import dataclass, asdict
from numbers import Real


@dataclass(frozen=True)
class DrawerGeometry:
    drawer_width_m: float = 0.24
    drawer_depth_m: float = 0.20
    drawer_height_m: float = 0.07
    held_object_radius_m: float = 0.035

    def __post_init__(self):
        values = list(asdict(self).values())
        if (
            any(isinstance(v, bool) or not isinstance(v, Real) for v in values)
            or not np.isfinite(values).all()
            or min(values) <= 0
        ):
            raise ValueError("drawer geometry must contain positive finite metres")

    @classmethod
    def from_mapping(cls, values=None):
        if values is None:
            return cls()
        if not isinstance(values, dict) or set(values) != set(cls.__dataclass_fields__):
            raise ValueError(
                "scene_geometry requires drawer_width_m, drawer_depth_m, drawer_height_m and held_object_radius_m"
            )
        return cls(**values)

    @property
    def box_kwargs(self):
        return dict(
            half_width=self.drawer_width_m / 2,
            depth=self.drawer_depth_m,
            half_height=self.drawer_height_m / 2,
        )

    def to_dict(self):
        return asdict(self)


def drawer_sweep(
    handle_positions,
    pull_start,
    open_frame,
    *,
    half_width=0.12,
    depth=0.20,
    half_height=0.035,
):
    points = np.asarray(handle_positions, float)
    # The handle advances along the drawer axis. The body extends behind it.
    start = points[pull_start]
    end = points[open_frame]
    direction = end - start
    direction[2] = 0
    length = np.linalg.norm(direction)
    if length < 0.03:
        raise ValueError("drawer handle translation is too small to infer sweep")
    axis = direction / length
    lateral = np.array([-axis[1], axis[0], 0.0])
    samples = points[pull_start : open_frame + 1]
    # Only use the outward pull segment, excluding the approach from home.
    near = np.linalg.norm(samples - end, axis=1) < length * 1.5
    rotation = np.column_stack([axis, lateral, [0.0, 0.0, 1.0]])
    local = samples[near] @ rotation
    lo = local.min(axis=0) - np.array([depth, half_width, half_height])
    hi = local.max(axis=0) + np.array([0.0, half_width, half_height])
    return rotation, lo, hi


def outside_box(point, bounds, radius=0.035):
    point = np.asarray(point, float)
    rotation, lo, hi = bounds
    local = point @ rotation
    gap = np.maximum(np.maximum(lo - local, local - hi), 0.0)
    return bool(np.linalg.norm(gap) > radius)


def safe_wait_before_entry(tcp, pickup, image_gate, bounds):
    """Latest original held pose clear of the entire inferred moving drawer."""
    candidates = [
        i for i in range(pickup, image_gate + 1) if outside_box(tcp[i], bounds)
    ]
    if not candidates:
        raise ValueError("no recorded held pose clears the drawer sweep")
    return max(candidates)


def drawer_body(position, rotation, *, half_width=0.12, depth=0.20, half_height=0.035):
    """Instantaneous solid exclusion volume before insertion is permitted."""
    local = np.asarray(position) @ rotation
    return (
        rotation,
        local - np.array([depth, half_width, half_height]),
        local + np.array([0.0, half_width, half_height]),
    )
