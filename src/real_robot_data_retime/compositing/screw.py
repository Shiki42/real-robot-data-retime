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
from .automatic_repair import clean_background_masks, repair_occlusions
from .foreground_reference import register_foreground
from .interpolation import FlowFrames


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
        self.foreground_images = {}
        self.foreground_pairs = {}
        self.foreground_references = []
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
        robots, self.mask_cleanup = clean_background_masks(
            self.frames, robots, self.plate, self.coverage
        )
        self.observed_packed = np.packbits(robots, axis=-1)
        self.layers = np.array([[dilate(mask, 3) for mask in row] for row in robots])
        self.automatic_repair = None
        # SAM's whole-arm mask may omit a protruding held screw. Keep observed
        # dark changed components connected to the arm, never synthesize pixels.
        for t, frame in enumerate(self.frames):
            changed = (np.max(cv2.absdiff(frame, self.plate), axis=-1) > 30) & (
                frame.max(axis=-1) < 155
            )
            changed[: int(h * 0.3)] = False
            regions = components(changed, 8)
            for side in range(2):
                nearby = dilate(self.layers[t, side], 5)
                for mask, _, _ in regions:
                    if (mask & nearby).any() and not (
                        mask & self.layers[t, 1 - side]
                    ).any():
                        self.layers[t, side] |= dilate(mask, 1)
        self.excluded = np.array([dilate(m.any(axis=0), 2) for m in self.layers])
        self.flow_frames = FlowFrames(self.frames)
        self.interpolated_frames = [0, 0]
        self.cache = {}
        self.missing_pixels = 0
        self.overlap_pixels = 0
        self.paired_frames = 0

    def repair_automatically(self, state, source_pairs):
        self.automatic_repair = repair_occlusions(self, state, source_pairs)
        return self.automatic_repair

    def restore_foreground(
        self, side, start, stop, reference, state, *, register=False
    ):
        """Reveal an occluded arm with a reviewed real foreground reference.

        This only repairs that arm's foreground. Original synchronized frames,
        the other arm's pixels and both motion clocks remain untouched.
        """
        n = len(self.frames)
        state = np.asarray(state)
        if side not in (0, 1) or not 0 <= start < stop <= n or not 0 <= reference < n:
            raise ValueError("foreground reference outside stage")
        if state.shape != (n, 7) or not np.isfinite(state).all():
            raise ValueError("invalid measured foreground poses")
        delta = np.abs(state[start:stop] - state[reference])
        limit = [0.05] * 4 + ([3.0, 3.0] if register else [0.05, 0.05]) + [0.1]
        if np.any(delta > np.array(limit)):
            raise ValueError("foreground donor exceeds measured arm pose bounds")
        observed = np.unpackbits(
            self.observed_packed, axis=-1, count=self.frames.shape[2]
        ).astype(bool)
        visible = observed[reference, side] & ~dilate(observed[reference, 1 - side], 1)
        repaired = []
        for t in range(start, stop):
            observed = np.unpackbits(
                self.observed_packed[t], axis=-1, count=self.frames.shape[2]
            ).astype(bool)
            donor, support = self.frames[reference], visible
            registration = None
            if register:
                target_visible = observed[side] & ~self.layers[t, 1 - side]
                donor, support, registration = register_foreground(
                    donor, self.frames[t], visible, target_visible
                )
            region = support & self.layers[t, 1 - side] & ~observed[side]
            if not region.any():
                continue
            key = (side, t)
            image = self.foreground_images.get(key, self.frames[t]).copy()
            image[region] = donor[region]
            self.foreground_images[key] = image
            self.layers[t, side] |= region
            repaired.append(
                {
                    "source_frame": t,
                    "pixels": int(region.sum()),
                    "registration": registration,
                }
            )
        self.foreground_pairs.clear()
        self.foreground_references.append(
            {
                "side": side,
                "registered": register,
                "reference_frame": reference,
                "max_measured_pose_difference": delta.max(axis=0).tolist(),
                "repairs": repaired,
            }
        )

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
        sources = [float(left), float(right)]
        if (
            not np.isfinite(sources).all()
            or min(sources) < 0
            or max(sources) > len(self.frames) - 1
        ):
            raise ValueError("source clocks outside stage video")
        times = [int(np.floor(source)) for source in sources]
        if left == right and left == times[0]:
            self.paired_frames += 1
            return self.original_frames[times[0]].copy()
        out = self._scene(times[0], 0).copy()
        scene = self._scene(times[1], 1)
        out[~self.left_scene] = scene[~self.left_scene]
        images, masks = [], []
        for side, source in enumerate(sources):
            lo, hi = int(np.floor(source)), int(np.ceil(source))
            if lo == hi:
                image = self.foreground_images.get((side, lo), self.frames[lo])
                mask = self.layers[lo, side]
            else:
                masks_pair = [self.layers[lo, side], self.layers[hi, side]]
                if (side, lo) in self.foreground_images or (
                    side,
                    hi,
                ) in self.foreground_images:
                    key = (side, lo)
                    if key not in self.foreground_pairs:
                        pair = np.stack(
                            [
                                self.foreground_images.get((side, t), self.frames[t])
                                for t in (lo, hi)
                            ]
                        )
                        self.foreground_pairs[key] = FlowFrames(pair)
                    image, mask = self.foreground_pairs[key].sample(
                        source - lo, masks_pair
                    )
                else:
                    image, mask = self.flow_frames.sample(source, masks_pair)
                self.interpolated_frames[side] += 1
            images.append(image)
            masks.append(mask)
        self.overlap_pixels += int((masks[0] & masks[1]).sum())
        for image, mask in zip(images, masks):
            out[mask] = image[mask]
        return out

    def report(self):
        return {
            "paired_source_frames": self.paired_frames,
            "foreground_references": self.foreground_references,
            "mask_cleanup": self.mask_cleanup,
            "automatic_repair": self.automatic_repair,
            "arm_overlap_pixel_frames": self.overlap_pixels,
            "unresolved_scene_patch_pixels": self.missing_pixels,
            "real_plate_coverage_fraction": float((self.coverage > 0).mean()),
            "unknown_depth_order": "stable_right_foreground",
            "background_color_matching": "shared_stationary_pixels_to_stage_end",
            "held_object_method": "arm_payload_masks_with_connected_motion_refinement",
            "metric_collision_checked": False,
            "interpolated_frames": {
                "left": self.interpolated_frames[0],
                "right": self.interpolated_frames[1],
            },
            "interpolation_method": "bidirectional_DIS_flow",
        }
