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


def discover_drawer_motion(
    candidate_tracks, right_gripper, fps, image_width, *, release_frame=None
):
    """Rank handles by persistent outward motion, dwell, then return.

    Coordinates must already be camera-registered. The open dwell must last at
    least 0.6 seconds; a missing observed return is not silently inferred.
    """
    points_all = np.asarray(candidate_tracks, float)
    support = np.zeros(len(points_all), int)
    for i in range(len(points_all)):
        for j in range(i):
            relative = points_all[i] - points_all[j]
            valid = np.isfinite(relative).all(axis=1)
            if valid.mean() < 0.4 or valid[-max(3, round(fps)) :].mean() < 0.5:
                continue
            residual = relative[valid] - np.median(relative[valid], axis=0)
            if (
                np.percentile(np.linalg.norm(residual, axis=1), 95)
                < image_width * 0.015
            ):
                support[i] += 1
                support[j] += 1
    references = (
        np.flatnonzero(support == support.max()).tolist()
        if support.max(initial=0) > 0
        else list(range(len(points_all)))
    )
    hypotheses = []
    for k, track in enumerate(candidate_tracks):
        for reference_id in [-1, *[j for j in references if j != k]]:
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
                returned = stop + int(np.flatnonzero(after)[0])
                falling = stable_runs(
                    np.gradient(smooth) < -span * 0.002, max(3, round(fps * 0.15))
                )
                closing = [
                    a
                    for a, b in falling
                    if start < a < returned and smooth[a] > span * 0.7
                ]
                if not closing:
                    continue
                close_start = closing[-1]
                if (
                    release_frame is not None
                    and not start < release_frame < close_start
                ):
                    continue
                # Quiet-pose jitter is not closing: require the sustained inward
                # motion leading to the observed return of the drawer handle.
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
                    valid[approach_start:close_start].mean()
                )
                hypotheses.append(
                    DrawerMotion(
                        start,
                        close_start,
                        k,
                        confidence,
                        int(approach_start),
                        reference_id,
                    )
                )
    return (
        max(
            hypotheses,
            key=lambda x: (
                x.confidence + 0.05 * (x.reference_candidate >= 0),
                -x.open_frame,
            ),
        )
        if hypotheses
        else None
    )


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
    settled_start = settled_open_frame(right_gripper, start, stop, fps, w)
    if settled_start is None:
        return None, ratio
    start = settled_start
    stop = observed_drawer_return(
        right_gripper, smooth, valid, start, pull, baseline, span, fps, w
    )
    if stop is None:
        return None, ratio
    return dict(
        open_frame=start,
        close_start=stop,
        pull_start=pull,
        confidence=float(valid[pull:stop].mean()),
        method="interior_area_with_observed_handle_return",
    ), ratio


def settled_open_frame(gripper, start, stop, fps, width):
    """First observed quiet interval overlapping the visually open phase."""
    gripper = np.asarray(gripper, float)
    observed = np.isfinite(gripper).all(axis=1)
    ids = np.flatnonzero(observed)
    if len(ids) < 2:
        return None
    filled = np.column_stack(
        [np.interp(np.arange(len(gripper)), ids, gripper[ids, a]) for a in [0, 1]]
    )
    smooth = median_filter(filled, size=(5, 1), mode="nearest")
    velocity = np.linalg.norm(np.diff(smooth, axis=0), axis=1)
    minimum = max(3, round(fps * 0.2))
    quiet = (velocity < width * 0.0015) & observed[:-1] & observed[1:]
    available = [
        max(a, start)
        for a, b in stable_runs(quiet, minimum)
        if min(b, stop) - max(a, start) >= minimum
    ]
    return min(available) if available else None


def observed_drawer_return(
    gripper, area, area_valid, opening, pull, baseline, span, fps, width
):
    """Require inward handle motion with a sustained loss of exposed interior.

    Interior area alone can collapse when the left arm occludes an open drawer.
    The right gripper supplies an independent observed motion witness.
    """
    gripper = np.asarray(gripper, float)
    valid = np.isfinite(gripper).all(axis=1)
    ids = np.flatnonzero(valid)
    window = max(3, round(fps * 0.2))
    opened = gripper[opening : opening + window]
    opened = opened[np.isfinite(opened).all(axis=1)]
    if not len(opened) or not valid[pull] or len(ids) < 2:
        return None
    origin = gripper[pull]
    displacement = np.median(opened, axis=0) - origin
    distance = np.linalg.norm(displacement)
    if distance < width * 0.025:
        return None
    projection = (gripper - origin) @ displacement / distance**2
    smooth = median_filter(
        np.interp(np.arange(len(gripper)), ids, projection[valid]), size=7
    )
    falling = (np.gradient(smooth) < -0.002) & valid
    for a, b in stable_runs(falling, max(3, round(fps * 0.15))):
        if a <= opening or smooth[a] < 0.7 or smooth[a] - smooth[b - 1] < 0.15:
            continue
        end = min(len(gripper), b + round(fps * 2))
        returned = valid[b:end] & (smooth[b:end] < 0.3)
        # This establishes closing onset, not successful complete closure.
        # Some source recordings end with a partly open drawer.
        closed = area_valid[b:end] & (area[b:end] < baseline + span * 0.65)
        if not (returned & closed).any():
            continue
        arrival = b + int(np.flatnonzero(returned & closed)[0])
        if np.max(smooth[b : arrival + 1]) > smooth[a] + 0.1:
            continue
        return int(a)
    return None
