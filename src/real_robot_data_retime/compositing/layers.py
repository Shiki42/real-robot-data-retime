"""Object-state and whole-arm compositing in a registered camera view."""

from pathlib import Path
import json
import cv2
import numpy as np
from ..background.clean_plate import (
    dilate,
    temporal_plate,
    real_patch,
    match_background_colors,
    blend_scene_patch,
)
from ..interaction.video import write_video, components
from ..tasks.workpiece import discover_bins
from .verification import OriginAudit


def drawer_region(frames, open_frame):
    h, w = frames.shape[1:3]
    candidates = []
    indices = np.unique(
        np.r_[
            0,
            open_frame,
            len(frames) - 1,
            np.linspace(0, len(frames) - 1, 20).astype(int),
        ]
    )
    for t in indices:
        hsv = cv2.cvtColor(frames[t], cv2.COLOR_BGR2HSV)
        red = (
            ((hsv[:, :, 0] < 12) | (hsv[:, :, 0] > 170))
            & (hsv[:, :, 1] > 120)
            & (hsv[:, :, 2] > 70)
        )
        regions = components(red, int(h * w * 0.005))
        if regions:
            candidates.append(max(regions, key=lambda a: a[1][4])[1])
    if not candidates:
        raise ValueError("drawer cabinet not visible in source video")
    area = np.percentile([s[4] for s in candidates], 90)
    boxes = [s for s in candidates if s[4] >= area * 0.6]
    x0 = min(int(s[0]) - 20 for s in boxes)
    y0 = min(int(s[1]) - 10 for s in boxes)
    x1 = max(int(s[0] + s[2] * 1.65) for s in boxes)
    y1 = max(int(s[1] + s[3] + h * 0.45) for s in boxes)
    mask = np.zeros((h, w), bool)
    mask[max(0, y0) : min(h, y1), max(0, x0) : min(w, x1)] = True
    return mask


def composite(
    frames, timeline, segmentation, left, right, output, debug_dir, *, depth=None
):
    n, h, w = frames.shape[:3]
    left, right = np.asarray(left), np.asarray(right)
    if left.shape != right.shape or left.ndim != 1 or not len(left):
        raise ValueError("source maps must be nonempty and equally sized")
    if min(left.min(), right.min()) < 0 or max(left.max(), right.max()) >= n:
        raise ValueError("source maps outside video")
    robots = np.unpackbits(segmentation["robots"], axis=-1, count=w).astype(bool)
    objects = np.unpackbits(segmentation["objects"], axis=-1, count=w).astype(bool)
    if robots.shape != (n, 2, h, w) or objects.shape[1:] != (n, h, w):
        raise ValueError("segmentation must match registered video dimensions")
    selected_ids = sorted({event["object_id"] for event in timeline["episodes"]})
    moving_objects = objects[selected_ids].any(axis=0)
    excluded = np.array(
        [
            dilate(a, 5) | dilate(b, 12)
            for a, b in zip(robots.any(axis=1), moving_objects)
        ]
    )
    dynamic_scene = (
        drawer_region(frames, timeline["drawer_motion"]["open_frame"])
        if timeline["task"] == "drawer"
        else None
    )
    frames, color_fits = match_background_colors(frames, excluded, dynamic_scene)
    reconstructed, coverage = temporal_plate(frames, excluded)
    plate = blend_scene_patch(
        frames[0], reconstructed, excluded[0], feather=8, color_match=True
    )
    debug = Path(debug_dir)
    debug.mkdir(parents=True, exist_ok=True)
    np.save(debug / "background_color_fits.npy", color_fits)
    cv2.imwrite(str(debug / "clean_plate.png"), plate)
    cv2.imwrite(
        str(debug / "clean_plate_coverage.png"),
        np.minimum(coverage * 4, 255).astype(np.uint8),
    )
    scene_regions = []
    if timeline["task"] == "workpiece":
        bins = discover_bins(frames[0])
        if len(bins) != 2:
            raise ValueError("both destination bins must be detected for compositing")
        for side, bin in enumerate(bins):
            x, y, bw, bh = map(int, bin["bbox"])
            mask = np.zeros((h, w), bool)
            mask[
                max(0, y - 3) : min(h, y + bh + 3), max(0, x - 3) : min(w, x + bw + 3)
            ] = True
            scene_regions.append((side, mask))
    drawer = None
    if timeline["task"] == "drawer":
        drawer = dynamic_scene.copy()
        for event in timeline["episodes"]:
            origin = objects[event["object_id"], : max(1, event["approach_start"])].any(
                axis=0
            )
            drawer &= ~dilate(origin, 6)
    # Copying source-time destination pixels preserves actual rims and occlusion.
    scene_excluded = np.array([dilate(m, 3) for m in robots.any(axis=1)])
    origins = []
    for event in timeline["episodes"]:
        origin_mask = objects[event["object_id"], 0]
        yy, xx = np.where(origin_mask)
        if not len(xx):
            raise ValueError("object origin mask is missing")
        region = np.zeros((h, w), bool)
        region[
            max(0, yy.min() - 10) : min(h, yy.max() + 11),
            max(0, xx.min() - 10) : min(w, xx.max() + 11),
        ] = True
        origins.append((event, region))
    patch_cache = {}
    overlap_pixels = 0
    metric_overlap_pixels = 0
    uncovered_patch_pixels = 0

    def patch(source, region, key, allowed=None):
        nonlocal uncovered_patch_pixels
        cache_key = (key, int(source))
        if cache_key not in patch_cache:
            result, missing = real_patch(
                frames, int(source), region, scene_excluded, plate, allowed
            )
            patch_cache[cache_key] = result
            uncovered_patch_pixels += missing
        return patch_cache[cache_key]

    audit = OriginAudit(frames, objects, timeline["episodes"])

    def render():
        nonlocal overlap_pixels, metric_overlap_pixels
        for l, r in zip(left, right):
            times = [int(l), int(r)]
            out = plate.copy()
            for side, region in scene_regions:
                im = patch(times[side], region, side)
                out[region] = im[region]
            if drawer is not None:
                motion = timeline["drawer_motion"]
                a, b = motion["open_frame"], motion["close_start"]
                if r < a or r >= b:
                    # Right owns moving drawer; preserve its real handle occlusion.
                    source = int(r)
                    im = frames[source]
                    other = drawer & dilate(robots[source, 0], 3)
                    out[drawer] = im[drawer]
                    if other.any():
                        restored = patch(source, drawer, "moving_drawer")
                        out[other] = restored[other]
                else:
                    source = int(np.clip(l, a, b - 1))
                    im = patch(source, drawer, "open_drawer", np.arange(a, b))
                    out[drawer] = im[drawer]
            for event, region in origins:
                side = ["left", "right"].index(event["robot_id"])
                source = times[side]
                pickup = event["pickup_frame"]
                allowed = (
                    np.arange(0, pickup) if source < pickup else np.arange(pickup, n)
                )
                clean_source = source if source < pickup else n - 1
                im = patch(
                    clean_source, region, ("origin", event["object_id"]), allowed
                )
                out = blend_scene_patch(out, im, region, feather=7, color_match=True)
            layers = []
            for side, t in enumerate(times):
                mask = robots[t, side].copy()
                for event in timeline["episodes"]:
                    if event["robot_id"] != ["left", "right"][side]:
                        continue
                    object_mask = objects[event["object_id"], t]
                    # Actual source pixels retain grasp-transition and release
                    # occlusions; no artificial object-coordinate interpolation.
                    mask |= object_mask
                layers.append(mask)
            overlap = layers[0] & layers[1]
            overlap_pixels += int(overlap.sum())
            if depth is None:
                # Stable image ordering; explicit report distinguishes this from
                # metric depth and collision verification.
                order = [0, 1]
                for side in order:
                    out[layers[side]] = frames[times[side]][layers[side]]
            else:
                z = [depth(t) for t in times]
                if any(a.shape != (h, w) for a in z):
                    raise ValueError("registered depth must match RGB shape")
                valid = [(a > 0) & np.isfinite(a) for a in z]
                metric_overlap_pixels += int((overlap & valid[0] & valid[1]).sum())
                right_front = ~(valid[0] & valid[1] & (z[0] < z[1]))
                lm = layers[0] & (~layers[1] | ~right_front)
                rm = layers[1] & (~layers[0] | right_front)
                out[lm] = frames[times[0]][lm]
                out[rm] = frames[times[1]][rm]
            audit.observe(out, frames, times, robots[times[0], 0] | robots[times[1], 1])
            yield out

    write_video(output, render(), timeline["fps"])
    report = dict(
        output=str(output),
        output_frames=len(left),
        clean_plate_method="masked_temporal_real_frames",
        real_plate_coverage_fraction=float((coverage > 0).mean()),
        inpainted_background_pixels=int((coverage == 0).sum()),
        inpainting_used=bool((coverage == 0).any()),
        unresolved_scene_patch_pixels=uncovered_patch_pixels,
        arm_overlap_pixel_frames=overlap_pixels,
        valid_metric_overlap_pixel_frames=metric_overlap_pixels,
        unknown_depth_order="stable_right_foreground",
        occlusion_method="aligned_metric_depth"
        if depth is not None
        else "stable_right_foreground",
        visual_validation="pending",
        automatic_origin_audit=audit.report(),
    )
    (debug / "compositing_report.json").write_text(json.dumps(report, indent=2))
    return report
