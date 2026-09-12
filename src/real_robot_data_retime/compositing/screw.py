"""Whole-arm overlays over independently clocked screw workspaces."""

import cv2
import numpy as np

from ..background.clean_plate import (
    dilate,
    match_background_colors,
    real_patch,
    temporal_plate,
)
from ..interaction.video import components


class ScrewStageCompositor:
    def __init__(self, frames, robots, split_x, left_scene_boxes):
        self.original_frames = np.asarray(frames)
        self.frames = self.original_frames
        robots = np.asarray(robots, bool)
        n, h, w = self.frames.shape[:3]
        if robots.shape != (n, 2, h, w) or not 0 < split_x < w:
            raise ValueError("stage masks/scene geometry do not match video")
        self.left_scene = np.zeros((h, w), bool)
        self.left_scene[:, :split_x] = True
        for x, y, bw, bh in left_scene_boxes:
            if not 0 <= x < x + bw <= w or not 0 <= y < y + bh <= h:
                raise ValueError("scene ownership box outside video")
            self.left_scene[y : y + bh, x : x + bw] = True
        # Preserve disconnected gripper pieces visible beyond an occluder.
        # Entry-edge filtering would remove those real foreground fragments.
        self.layers = np.array([[dilate(mask, 3) for mask in row] for row in robots])
        excluded = self.layers.any(axis=1)
        moving_scene = np.zeros((h, w), bool)
        moving_scene[int(h * 0.65) :] = True
        for x, y, bw, bh in left_scene_boxes:
            moving_scene[y : y + bh, x : x + bw] = True
        self.frames, self.color_fits = match_background_colors(
            self.frames, excluded, moving_scene
        )
        self.plate, self.coverage = temporal_plate(self.frames, excluded)
        # SAM's whole-arm mask may omit a protruding held screw. Keep observed
        # dark changed components connected to the arm, never synthesize pixels.
        for t, frame in enumerate(self.frames):
            changed = (np.max(cv2.absdiff(frame, self.plate), axis=-1) > 30) & (
                frame.max(axis=-1) < 155
            )
            changed[: int(h * 0.3)] = False
            for side in range(2):
                nearby = dilate(self.layers[t, side], 5)
                for mask, _, _ in components(changed, 8):
                    if (mask & nearby).any() and not (
                        mask & self.layers[t, 1 - side]
                    ).any():
                        self.layers[t, side] |= dilate(mask, 1)
        self.excluded = np.array([dilate(m.any(axis=0), 2) for m in self.layers])
        self.cache = {}
        self.missing_pixels = 0
        self.overlap_pixels = 0
        self.paired_frames = 0

    def _scene(self, source, side):
        key = (int(source), side)
        if key in self.cache:
            return self.cache[key]
        scene = self.left_scene if side == 0 else ~self.left_scene
        wrong = self.layers[source, 1 - side] & scene
        out = self.frames[source].copy()
        if wrong.any():
            yy, xx = np.where(wrong)
            ys = slice(int(yy.min()), int(yy.max()) + 1)
            xs = slice(int(xx.min()), int(xx.max()) + 1)
            patch, missing = real_patch(
                self.frames[:, ys, xs],
                source,
                wrong[ys, xs],
                self.excluded[:, ys, xs],
                self.plate[ys, xs],
            )
            target = out[ys, xs]
            mask = wrong[ys, xs]
            target[mask] = patch[mask]
            self.missing_pixels += missing
        self.cache[key] = out
        return out

    def frame(self, left, right):
        left, right = int(left), int(right)
        if left == right:
            self.paired_frames += 1
            return self.original_frames[left].copy()
        images = [self.frames[left], self.frames[right]]
        out = self._scene(left, 0).copy()
        scene = self._scene(right, 1)
        out[~self.left_scene] = scene[~self.left_scene]
        masks = [self.layers[left, 0], self.layers[right, 1]]
        self.overlap_pixels += int((masks[0] & masks[1]).sum())
        # Reuse the established RGB-only right-foreground ordering. No metric
        # depth is available in this dataset; this is a visual pilot.
        for im, mask in zip(images, masks):
            out[mask] = im[mask]
        return out

    def report(self):
        return {
            "paired_source_frames": self.paired_frames,
            "arm_overlap_pixel_frames": self.overlap_pixels,
            "unresolved_scene_patch_pixels": self.missing_pixels,
            "real_plate_coverage_fraction": float((self.coverage > 0).mean()),
            "unknown_depth_order": "stable_right_foreground",
            "background_color_matching": "shared_stationary_pixels_to_stage_end",
            "held_object_method": "observed_connected_dark_motion",
            "metric_collision_checked": False,
        }
