"""Estimate an end effector from a whole-arm mask without selecting cables."""

import cv2
import numpy as np
from skimage.graph import MCP_Geometric
from .video import components, pixel_kernel


def end_effector(robot_mask, side):
    h, w = robot_mask.shape
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pixel_kernel(5, w),) * 2)
    thick = cv2.morphologyEx(robot_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    regions = components(thick, 20)
    if not regions:
        return None
    core, stat, _ = max(regions, key=lambda z: z[1][4])
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
