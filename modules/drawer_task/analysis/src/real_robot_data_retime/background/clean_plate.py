"""Reconstruct registered real pixels, retaining explicit coverage evidence."""

import cv2
import numpy as np


def dilate(mask, radius=3):
    return cv2.dilate(
        np.asarray(mask, np.uint8), np.ones((radius * 2 + 1, radius * 2 + 1), np.uint8)
    ).astype(bool)


def temporal_plate(frames, excluded, *, samples=61):
    n, h, w = frames.shape[:3]
    ids = np.unique(np.linspace(0, n - 1, min(n, samples)).astype(int))
    output = np.empty((h, w, 3), np.uint8)
    coverage = np.zeros((h, w), np.int32)
    # Tile rows to bound temporary float memory on native-resolution videos.
    for y in range(0, h, 24):
        valid = ~excluded[ids, y : y + 24]
        values = frames[ids, y : y + 24].astype(np.float32)
        counts = valid.sum(axis=0)
        # Sort masked observations after real ones, then select a real median.
        values[~valid] = 256
        values.sort(axis=0)
        median = np.take_along_axis(
            values, np.maximum(0, (counts - 1) // 2)[None, :, :, None], axis=0
        )[0]
        output[y : y + 24] = np.minimum(median, 255).astype(np.uint8)
        coverage[y : y + 24] = counts
    missing = coverage == 0
    if missing.any():
        output[missing] = 0
        output = cv2.inpaint(output, missing.astype(np.uint8), 3, cv2.INPAINT_TELEA)
    return output, coverage


def real_patch(frames, source, region, excluded, base, allowed=None):
    """Fill a requested scene patch with nearest unoccluded real observations."""
    image = frames[source].copy()
    missing = region & excluded[source]
    ids = np.arange(len(frames)) if allowed is None else np.asarray(allowed)
    for t in ids[np.argsort(np.abs(ids - source), kind="stable")]:
        fill = missing & ~excluded[t]
        image[fill] = frames[t][fill]
        missing[fill] = False
        if not missing.any():
            break
    image[missing] = base[missing]
    return image, int(missing.sum())


def match_background_colors(frames, excluded, scene_region=None):
    """Robust per-channel exposure fit from shared stationary background pixels."""
    reference = frames[-1].astype(float)
    normalized = np.empty_like(frames)
    fits = []
    for t, frame in enumerate(frames):
        valid = ~excluded[t] & ~excluded[-1]
        if scene_region is not None:
            valid &= ~scene_region
        # Deterministic sparse samples spread across the whole image.
        yy, xx = np.where(valid)
        step = max(1, len(xx) // 6000)
        yy, xx = yy[::step], xx[::step]
        if len(xx) < 100:
            raise ValueError("insufficient common background for exposure registration")
        source = frame[yy, xx].astype(float)
        target = reference[yy, xx]
        result = frame.astype(float)
        coefficients = []
        for channel in range(3):
            x = source[:, channel]
            y = target[:, channel]
            keep = (x > 5) & (x < 250) & (y > 5) & (y < 250)
            gain, offset = 1.0, 0.0
            for _ in range(3):
                if keep.sum() < 50 or np.std(x[keep]) < 5:
                    break
                gain, offset = np.linalg.lstsq(
                    np.column_stack([x[keep], np.ones(keep.sum())]), y[keep], rcond=None
                )[0]
                residual = y - (gain * x + offset)
                center = np.median(residual[keep])
                mad = np.median(np.abs(residual[keep] - center))
                keep &= np.abs(residual - center) < max(5, 3 * mad)
            gain = float(np.clip(gain, 0.65, 1.5))
            offset = float(np.clip(offset, -50, 50))
            result[:, :, channel] = frame[:, :, channel] * gain + offset
            coefficients.append([gain, offset])
        normalized[t] = np.clip(result, 0, 255).astype(np.uint8)
        fits.append(coefficients)
    return normalized, np.asarray(fits)


def blend_scene_patch(base, patch, region, *, feather=6, color_match=False):
    """Blend only empty scene-patch boundaries, keeping object interiors opaque."""
    result = base.copy()
    pixels = patch.astype(float)
    if color_match:
        ring = dilate(region, 8) & ~region
        if ring.sum() >= 20:
            offset = np.median(base[ring].astype(float) - pixels[ring], axis=0)
            pixels = np.clip(pixels + offset, 0, 255)
    distance = cv2.distanceTransform(region.astype(np.uint8), cv2.DIST_L2, 3)
    alpha = np.minimum(distance / max(1, feather), 1.0)[:, :, None]
    result = np.clip(base * (1 - alpha) + pixels * alpha, 0, 255).astype(np.uint8)
    return result
