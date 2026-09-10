"""Object-state and whole-arm compositing in a registered camera view."""

import json
from pathlib import Path

import cv2
import numpy as np

from ..background.clean_plate import (
    blend_scene_patch,
    dilate,
    match_background_colors,
    real_patch,
    temporal_plate,
)
from ..interaction.video import components, write_video
from ..tasks.workpiece import discover_bins
from .verification import OriginAudit
from .interpolation import FlowFrames
from .ownership import arm_foreground, is_carried


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
    if min(left.min(), right.min()) < 0 or max(left.max(), right.max()) > n - 1:
        raise ValueError("source maps outside video")
    if not np.isfinite([left, right]).all():
        raise ValueError("nonfinite source clock")
    fractional = bool(
        np.any(left != np.floor(left)) or np.any(right != np.floor(right))
    )
    flow_frames = None
    robots = np.unpackbits(segmentation["robots"], axis=-1, count=w).astype(bool)
    objects = np.unpackbits(segmentation["objects"], axis=-1, count=w).astype(bool)
    if robots.shape != (n, 2, h, w) or objects.shape[1:] != (n, h, w):
        raise ValueError("segmentation must match registered video dimensions")
    if timeline["task"] == "drawer":
        for event in timeline["episodes"]:
            side = ["left", "right"].index(event["robot_id"])
            release = event["release_frame"]
            # Enforce scene ownership before both patch exclusion and layering;
            # a stale SAM arm mask must not erase a placed cube from scene donors.
            robots[release:, side] &= ~objects[event["object_id"], release:]
    entry = np.zeros((2, h, w), bool)
    entry[0, int(h * 0.4) :, : max(1, round(w * 0.08))] = True
    entry[1, int(h * 0.4) :, round(w * 0.92) :] = True
    # Keep observed boundary fragments when an arm is mostly outside the view.
    anchors = (robots[0] | robots[-1]) & entry
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
    if fractional:
        flow_frames = FlowFrames(frames)
    reconstructed, coverage = temporal_plate(frames, excluded)
    plate_source = 0
    plate_region = excluded[plate_source]
    plate_match_valid = ~plate_region & (coverage > 0)
    if dynamic_scene is not None:
        plate_match_valid &= ~dynamic_scene
    plate_color_match = (
        dilate(plate_region, 8) & ~plate_region & plate_match_valid
    ).sum() >= 20
    # Keep the transition inside the exclusion margin so old arm pixels cannot bleed through.
    plate = blend_scene_patch(
        frames[plate_source],
        reconstructed,
        plate_region,
        feather=4,
        color_match=bool(plate_color_match),
        color_reference_mask=plate_match_valid,
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
    paired_source_frames = 0
    uncovered_patch_pixels = 0
    skipped_origin_color_matches = 0

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
        nonlocal \
            overlap_pixels, \
            metric_overlap_pixels, \
            paired_source_frames, \
            skipped_origin_color_matches
        for l, r in zip(left, right):
            times = [int(l), int(r)]
            if l == r and l == int(l):
                # An unchanged clock pair needs no spatial reconstruction.
                out = frames[int(l)].copy()
                audit.observe(out, frames, times, robots[int(l)].any(axis=0))
                paired_source_frames += 1
                yield out
                continue
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
                    if r != source:
                        im, _ = flow_frames.sample(
                            float(r), [np.ones((h, w), bool)] * 2
                        )
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
                match_valid = ~(
                    scene_excluded[clean_source]
                    | scene_excluded[times[0]]
                    | scene_excluded[times[1]]
                    | moving_objects[clean_source]
                    | moving_objects[times[0]]
                    | moving_objects[times[1]]
                )
                if dynamic_scene is not None:
                    match_valid &= ~dynamic_scene
                for _, scene_region in scene_regions:
                    match_valid &= ~scene_region
                local_match = (dilate(region, 8) & ~region & match_valid).sum() >= 20
                skipped_origin_color_matches += int(not local_match)
                out = blend_scene_patch(
                    out,
                    im,
                    region,
                    feather=7,
                    color_match=bool(local_match),
                    color_reference_mask=match_valid,
                )
            # Stationary objects are scene content, never a right/left foreground
            # override. Either arm can occlude them at its independently mapped time.
            for event in timeline["episodes"]:
                side = ("left", "right").index(event["robot_id"])
                t = times[side]
                if is_carried(event, t):
                    continue
                if drawer is not None and t >= event["release_frame"]:
                    continue
                visible = objects[event["object_id"], t] & ~robots[t].any(axis=0)
                out[visible] = frames[t][visible]
            layers = [
                arm_foreground(robots, objects, timeline["episodes"], side, t)
                | anchors[side]
                for side, t in enumerate(times)
            ]
            layer_images = [frames[t] for t in times]
            for side, source in enumerate((l, r)):
                if source != int(source):
                    lo, hi = int(np.floor(source)), int(np.ceil(source))
                    endpoint_masks = [
                        arm_foreground(robots, objects, timeline["episodes"], side, t)
                        | anchors[side]
                        for t in (lo, hi)
                    ]
                    layer_images[side], layers[side] = flow_frames.sample(
                        float(source), endpoint_masks
                    )
            overlap = layers[0] & layers[1]
            if fractional and depth is None and overlap.any():
                raise ValueError("interpolated foregrounds violate projected clearance")
            overlap_pixels += int(overlap.sum())
            if depth is None:
                # Stable image ordering; explicit report distinguishes this from
                # metric depth and collision verification.
                order = [0, 1]
                for side in order:
                    out[layers[side]] = layer_images[side][layers[side]]
            else:
                z = []
                for source in (l, r):
                    lo, hi = int(np.floor(source)), int(np.ceil(source))
                    if lo == hi:
                        z.append(depth(lo))
                    else:
                        values = [depth(lo), depth(hi)]
                        interpolated, valid_depth = flow_frames.sample(
                            float(source), [v > 0 for v in values], values=values
                        )
                        interpolated[~valid_depth] = 0
                        z.append(interpolated)
                if any(a.shape != (h, w) for a in z):
                    raise ValueError("registered depth must match RGB shape")
                valid = [(a > 0) & np.isfinite(a) for a in z]
                metric_overlap_pixels += int((overlap & valid[0] & valid[1]).sum())
                right_front = ~(valid[0] & valid[1] & (z[0] < z[1]))
                lm = layers[0] & (~layers[1] | ~right_front)
                rm = layers[1] & (~layers[0] | right_front)
                out[lm] = layer_images[0][lm]
                out[rm] = layer_images[1][rm]
            audit.observe(out, frames, times, robots[times[0], 0] | robots[times[1], 1])
            yield out

    write_video(output, render(), timeline["fps"])
    report = dict(
        output=str(output),
        output_frames=len(left),
        interpolated_right_frames=int(np.sum(right != np.floor(right))),
        interpolated_left_frames=int(np.sum(left != np.floor(left))),
        interpolation_method="bidirectional_DIS_flow" if fractional else "none",
        paired_source_frames=paired_source_frames,
        clean_plate_method="masked_temporal_real_frames",
        clean_plate_reference_frame=plate_source,
        initial_plate_color_adjusted=bool(plate_color_match),
        skipped_origin_color_matches=skipped_origin_color_matches,
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
