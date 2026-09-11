"""Select held wait poses that admit the sampled right-arm opening motion."""

from functools import lru_cache
import numpy as np
from ..background.clean_plate import dilate
from ..compositing.ownership import arm_foreground


def opening_wait_candidates(
    eligible, robots, objects, events, left_start, opening, position, fps
):
    # Both half-interval variants must use one peak and one A duration. The first
    # half has the earliest right clock when A completes; the other is a subset.
    base = round(position if position < 0.5 else position - 0.5, 12)
    brake = round(fps * 0.5)
    added = brake - round(brake / 2)

    @lru_cache(None)
    def opening_support(start):
        return dilate(robots[start : opening + 1, 1].any(axis=0), 2)

    allowed = np.asarray(eligible, bool).copy()
    for frame in np.flatnonzero(allowed):
        duration = frame - left_start + added
        # Include a frame of quantization uncertainty and the incoming edge.
        right_start = max(0, int(np.floor(base * (duration + opening))) - 1)
        if right_start > opening:
            continue
        held = dilate(arm_foreground(robots, objects, events, 0, int(frame)), 2)
        if np.any(held & opening_support(right_start)):
            allowed[frame] = False
    return allowed


def prepare_uniform_lift(
    state,
    action,
    event,
    motion,
    robots,
    objects,
    events,
    urdf,
    mesh_root,
    fps,
    position,
    scene_geometry=None,
):
    """Shared expensive evidence/geometry for searching BOTH variants' wait pose."""
    from pathlib import Path
    from types import SimpleNamespace
    from ..collision.piperx import PiperXClearance
    from ..collision.drawer import drawer_sweep, outside_box, DrawerGeometry
    from .smooth import held_grasp_interval, select_lift_peak
    from .holds import compress_static_spans, stationary_pose_mask

    geometry = DrawerGeometry.from_mapping(scene_geometry)
    checker = PiperXClearance(
        state[:, :7], state[:, 7:], Path(urdf), Path(mesh_root), margin_m=0.005
    )
    tcp = np.array([[pose[4] for pose in arm] for arm in checker.poses])
    opening, closing = motion["open_frame"], motion["close_start"]
    volume = drawer_sweep(tcp[1], motion["pull_start"], opening, **geometry.box_kwargs)
    grasp, end, aperture = held_grasp_interval(state, action, event, fps)
    eligible = np.zeros(len(state), bool)
    down = round(round(fps * 0.5) / 2)
    up = round(round(fps * 0.3) / 2)
    for t in range(grasp, end):
        if t - down < event["approach_start"] or t + up >= end:
            continue
        if max(state[t, 6], action[t, 6]) > aperture:
            continue
        commanded = checker._pose(action[t, :7], 0)
        eligible[t] = (
            outside_box(tcp[0, t], volume, radius=geometry.held_object_radius_m)
            and checker.arm_clears_volume(0, t, volume, margin=0.005)
            and outside_box(commanded[4], volume, radius=geometry.held_object_radius_m)
            and checker.pose_clears_volume(commanded, volume, margin=0.005)
        )
    recorded_peak_height = float(np.max(tcp[0, grasp:end, 2]))
    # A must finish at the recorded lift apex, not at a convenient low grasp
    # pose. Search only the established two-millimetre peak band.
    eligible &= tcp[0, :, 2] >= recorded_peak_height - 0.002
    eligible = opening_wait_candidates(
        eligible,
        robots,
        objects,
        events,
        event["approach_start"],
        opening,
        position,
        fps,
    )
    remaining = eligible.copy()
    candidates = []
    while remaining.any():
        peak, _ = select_lift_peak(tcp[0], remaining, grasp, end - 1)
        candidates.append(peak)
        remaining[peak] = False
    if not candidates:
        raise ValueError("no safe held lift peak before drawer entry")
    right = compress_static_spans(
        state[:, 7:],
        action[:, 7:],
        [(opening + 1, closing)],
        fps,
        minimum_seconds=3 / fps,
        guard_seconds=0.2,
    )
    # Retain the recorded joint timeline through C. This lets the scheduler
    # replay existing paired occlusions verbatim instead of inventing pixels.
    right = np.union1d(
        right, np.arange(max(opening + 1, min(candidates) - 2), len(state))
    )
    return SimpleNamespace(
        checker=checker,
        geometry=geometry,
        tcp=tcp,
        volume=volume,
        grasp=grasp,
        held_end=end,
        aperture=aperture,
        eligible=eligible,
        candidates=candidates,
        right_sources=right,
        recorded_peak_height=recorded_peak_height,
        right_quiet=stationary_pose_mask(state[:, 7:], action[:, 7:], fps),
    )
