"""Visual drawer state and precedence gates; no manually supplied event frames."""

from dataclasses import dataclass
import numpy as np
from scipy.ndimage import median_filter
from ..interaction.evidence import stable_runs


@dataclass(frozen=True)
class DrawerMotion:
    open_frame: int
    close_start: int
    handle_candidate: int
    confidence: float
    pull_start: int
    reference_candidate: int


def discover_drawer_motion(candidate_tracks, right_gripper, fps, image_width):
    """Rank handles by persistent outward motion, dwell, then return.

    Coordinates must already be camera-registered. The open dwell must last at
    least 0.6 seconds; a missing observed return is not silently inferred.
    """
    hypotheses = []
    for k, track in enumerate(candidate_tracks):
        for reference_id in [-1, *[j for j in range(len(candidate_tracks)) if j != k]]:
            points = np.asarray(track, float)
            reference = (
                np.asarray(candidate_tracks[reference_id], float)
                if reference_id >= 0
                else np.zeros_like(points)
            )
            relative = points - reference
            valid = np.isfinite(relative).all(axis=1)
            if valid.sum() < max(10, int(fps * 2)):
                continue
            ids = np.flatnonzero(valid)
            origin = np.median(relative[ids[: max(3, int(fps * 0.3))]], axis=0)
            centered = relative[valid] - origin
            _, _, axes = np.linalg.svd(centered, full_matrices=False)
            coordinate = np.full(len(points), np.nan)
            coordinate[valid] = centered @ axes[0]
            if np.nanpercentile(coordinate, 90) < -np.nanpercentile(coordinate, 10):
                coordinate = -coordinate
            span = np.nanpercentile(coordinate, 95)
            if span < image_width * 0.04:
                continue
            smooth = median_filter(
                np.interp(np.arange(len(points)), ids, coordinate[valid]), size=7
            )
            speed = np.abs(np.gradient(smooth))
            dwell = (smooth > span * 0.85) & (speed < image_width * 0.0015) & valid
            for start, stop in stable_runs(dwell, max(5, int(fps * 0.6))):
                before = valid[:start] & (smooth[:start] < span * 0.1)
                after = valid[stop:] & (smooth[stop:] < span * 0.3)
                if not before.any() or not after.any():
                    continue
                # Use visible contact throughout the outward motion, not color name.
                approach_start = np.flatnonzero(before)[-1]
                contact = np.linalg.norm(
                    points[approach_start : start + 1]
                    - right_gripper[approach_start : start + 1],
                    axis=1,
                )
                observed = np.isfinite(contact)
                if observed.sum() < 3:
                    continue
                proximity = float(np.mean(contact[observed] < image_width * 0.13))
                confidence = 0.6 * proximity + 0.4 * float(
                    valid[approach_start:stop].mean()
                )
                hypotheses.append(
                    DrawerMotion(
                        start, stop, k, confidence, int(approach_start), reference_id
                    )
                )
    return max(hypotheses, key=lambda x: x.confidence) if hypotheses else None


def precedence_gate(
    left_source,
    right_source,
    *,
    safe_approach_end,
    open_frame,
    withdrawal_frame,
    close_start,
):
    """Both dependencies must hold at each scheduled pair of source frames."""
    return (left_source <= safe_approach_end or right_source >= open_frame) and (
        right_source < close_start or left_source >= withdrawal_frame
    )


def discover_insertion_gate(
    open_frame_image, object_track, pickup_frame, release_frame
):
    """Find the last recorded approach pose outside the visible open interior.

    Task prior: colored cabinet and a pale drawer interior. Returns the mask as
    evidence; this image-space exclusion is additional to the 3-D arm audit.
    """
    import cv2
    from ..interaction.video import components

    image = open_frame_image
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red = (
        ((hsv[:, :, 0] < 12) | (hsv[:, :, 0] > 170))
        & (hsv[:, :, 1] > 120)
        & (hsv[:, :, 2] > 70)
    )
    cabinets = components(red, int(h * w * 0.005))
    if not cabinets:
        raise ValueError("colored cabinet not visible at open state")
    _, stat, _ = max(cabinets, key=lambda z: z[1][4])
    x, y, bw, bh, area = stat
    interior = (hsv[:, :, 1] < 100) & (hsv[:, :, 2] > 130)
    roi = np.zeros((h, w), bool)
    roi[
        max(0, y + bh + int(h * 0.07)) : min(h, y + bh + int(h * 0.4)),
        max(0, x - int(bw * 0.15)) : min(w, x + int(bw * 1.55)),
    ] = True
    options = components(interior & roi, int(h * w * 0.01))
    if not options:
        raise ValueError("open drawer interior not discovered")
    mask, _, _ = max(options, key=lambda z: z[1][4])
    expanded = cv2.dilate(mask.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(
        bool
    )
    track = np.asarray(object_track, float)
    inside = []
    for t in range(pickup_frame, release_frame + 1):
        c = track[t]
        if np.isfinite(c).all():
            xx, yy = np.rint(c).astype(int)
            if 0 <= xx < w and 0 <= yy < h and expanded[yy, xx]:
                inside.append(t)
    if not inside:
        raise ValueError("held object never approaches detected drawer interior")
    entry = min(inside)
    earlier = [t for t in range(pickup_frame, entry) if np.isfinite(track[t]).all()]
    if not earlier:
        raise ValueError("no observed safe approach pose before drawer entry")
    return max(earlier), mask


def drawer_area_motion(frames, fps, right_gripper):
    """Independent drawer state from interior area, with cabinet occlusion gating."""
    import cv2
    from ..interaction.video import components

    n, h, w = frames.shape[:3]
    ratio = np.full(n, np.nan)
    reference = None
    for t, image in enumerate(frames):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        red = (
            ((hsv[:, :, 0] < 12) | (hsv[:, :, 0] > 170))
            & (hsv[:, :, 1] > 120)
            & (hsv[:, :, 2] > 70)
        )
        cabinets = components(red, int(h * w * 0.005))
        if not cabinets:
            continue
        _, stat, _ = max(cabinets, key=lambda z: z[1][4])
        x, y, bw, bh, area = stat
        if reference is None:
            reference = area
        if area < reference * 0.6:
            continue
        roi = np.zeros((h, w), bool)
        roi[
            y + int(bh * 0.75) : min(h, y + bh + int(h * 0.45)),
            x : min(w, x + int(bw * 1.6)),
        ] = True
        cream = (
            roi
            & (hsv[:, :, 0] > 10)
            & (hsv[:, :, 0] < 40)
            & (hsv[:, :, 1] > 20)
            & (hsv[:, :, 1] < 110)
            & (hsv[:, :, 2] > 130)
        )
        regions = components(cream, 100)
        if regions:
            ratio[t] = max(z[1][4] for z in regions) / reference
    valid = np.isfinite(ratio)
    if valid.sum() < fps * 2:
        return None, ratio
    ids = np.flatnonzero(valid)
    smooth = median_filter(np.interp(np.arange(n), ids, ratio[valid]), size=7)
    baseline = float(np.median(smooth[: max(3, round(fps))]))
    span = float(np.percentile(ratio[valid], 90) - baseline)
    if span < 0.25:
        return None, ratio
    high = smooth > baseline + span * 0.8
    runs = stable_runs(high, max(5, round(fps * 0.6)))
    hypotheses = []
    for start, stop in runs:
        before = np.flatnonzero(
            valid[:start] & (smooth[:start] < baseline + span * 0.2)
        )
        after = np.flatnonzero(valid[stop:] & (smooth[stop:] < baseline + span * 0.3))
        if len(before) and len(after):
            hypotheses.append((stop - start, start, stop, int(before[-1])))
    if not hypotheses:
        return None, ratio
    _, start, stop, pull = max(hypotheses)
    velocity = np.linalg.norm(np.diff(right_gripper, axis=0), axis=1)
    settled = stable_runs(
        (velocity < w * 0.0015) & np.isfinite(velocity), max(3, round(fps * 0.2))
    )
    available = [a for a, b in settled if start <= a < stop]
    if not available:
        return None, ratio
    start = min(available)
    return dict(
        open_frame=start,
        close_start=stop,
        pull_start=pull,
        confidence=float(valid[pull:stop].mean()),
        method="cabinet_normalized_interior_area",
    ), ratio
