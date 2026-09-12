"""Estimate an end effector from a whole-arm mask without selecting cables."""

import cv2
import numpy as np
from skimage.graph import MCP_Geometric

from .video import components, pixel_kernel


def end_effector(robot_mask, side, *, topology_mask=None):
    h, w = robot_mask.shape
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pixel_kernel(5, w),) * 2)
    topology = robot_mask if topology_mask is None else robot_mask & topology_mask
    thick = cv2.morphologyEx(topology.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    # Reflective wrist collars can split the visible hand from the forearm.
    # Bridge only the distance-map topology; output masks retain source pixels.
    bridge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pixel_kernel(11, w),) * 2)
    thick = cv2.morphologyEx(thick, cv2.MORPH_CLOSE, bridge)
    regions = components(thick, 20)
    if not regions:
        return None
    core, _stat, _ = max(regions, key=lambda z: z[1][4])
    yy, xx = np.where(core)
    entry = xx < xx.min() + 4 if side == 0 else xx > xx.max() - 4
    touches_side = xx.min() < w * 0.08 if side == 0 else xx.max() > w * 0.92
    if not touches_side and yy.min() < h * 0.04:
        entry = yy < yy.min() + 4
    anchor = np.array([np.median(xx[entry]), np.median(yy[entry])])
    points = np.column_stack([xx, yy])
    costs = np.where(core, 1.0, np.inf)
    seeds = list(zip(yy[entry].tolist(), xx[entry].tolist()))
    distance_map, _ = MCP_Geometric(costs, fully_connected=True).find_costs(seeds)
    distance = distance_map[yy, xx]
    finite = np.isfinite(distance)
    distal = points[finite & (distance >= np.percentile(distance[finite], 95))]
    center = np.median(distal, axis=0)
    direction = center - anchor
    direction /= max(np.linalg.norm(direction), 1.0)
    perpendicular = np.array([-direction[1], direction[0]])
    spread = np.percentile(distal @ perpendicular, 95) - np.percentile(
        distal @ perpendicular, 5
    )
    x0, y0 = np.maximum(np.rint(center - w * 0.11).astype(int), 0)
    x1, y1 = np.minimum(np.rint(center + w * 0.11).astype(int), [w, h])
    mask = np.zeros((h, w), bool)
    mask[y0:y1, x0:x1] = robot_mask[y0:y1, x0:x1]
    return center, float(spread), mask


def motion_supported_grippers(grippers, geometry, frame_shape):
    """Reject stationary scene branches as tips without changing source robot pixels."""
    h, w = frame_shape
    result = dict(grippers)
    for key in ("centers", "apertures", "masks"):
        result[key] = grippers[key].copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pixel_kernel(21, w),) * 2)
    radius = max(3, round(w * 0.025))
    changed = [0, 0]
    for t in range(len(grippers["centers"])):
        for side in (0, 1):
            point = grippers["centers"][t, side]
            motion = geometry["masks"][t] == side + 1
            supported = False
            if np.isfinite(point).all():
                x, y = np.rint(point).astype(int)
                supported = (
                    np.count_nonzero(
                        motion[
                            max(0, y - radius) : min(h, y + radius + 1),
                            max(0, x - radius) : min(w, x + radius + 1),
                        ]
                    )
                    >= 20
                )
            if supported:
                continue
            robot = np.unpackbits(
                grippers["robot_masks"][t, side], axis=-1, count=w
            ).astype(bool)
            support = cv2.dilate(motion.astype(np.uint8), kernel).astype(bool)
            estimate = end_effector(robot, side, topology_mask=support)
            result["centers"][t, side] = np.nan
            result["apertures"][t, side] = np.nan
            result["masks"][t, side] = False
            if estimate is not None:
                center, spread, mask = estimate
                result["centers"][t, side] = center
                result["apertures"][t, side] = spread
                result["masks"][t, side] = mask
            changed[side] += 1
    result["localization_audit"] = {
        "method": "motion_supported_tip_topology",
        "reestimated_frames_by_arm": changed,
    }
    return result
