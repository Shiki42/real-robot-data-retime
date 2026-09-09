"""Image evidence independent of the propagated candidate identity."""

import cv2
import numpy as np


def origin_presence(frames, proposal, gripper, start, stop):
    """Measure whether an alleged picked object actually remains at its origin.

    Returns observations only when the gripper is clear of the origin. A high
    match supports rejecting a swapped track; missing observations are unknown.
    Frames must share the registered reference coordinate system.
    """
    x, y, w, h = map(int, proposal["bbox"])
    height, width = frames.shape[1:3]
    pad = 3
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(width, x + w + pad), min(height, y + h + pad)
    template = cv2.cvtColor(frames[0, y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    if min(template.shape) < 3 or template.std() < 3:
        return []
    observations = []
    center = np.asarray(proposal["origin"])
    radius = max(w, h) * 1.5
    for t in range(start, min(stop, len(frames)), 3):
        if (
            not np.isfinite(gripper[t]).all()
            or np.linalg.norm(gripper[t] - center) < radius
        ):
            continue
        search = cv2.cvtColor(
            frames[
                t,
                max(0, y0 - 4) : min(height, y1 + 4),
                max(0, x0 - 4) : min(width, x1 + 4),
            ],
            cv2.COLOR_BGR2GRAY,
        )
        scores = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
        observations.append((t, float(scores.max())))
    return observations


def validate_origin_departure(frames, proposal, gripper, pickup, release):
    observations = origin_presence(frames, proposal, gripper, pickup + 5, release)
    if len(observations) < 3:
        return dict(
            verified=False,
            reason="origin_occluded_or_unobservable",
            observations=observations,
        )
    remaining = np.mean([score > 0.85 for _, score in observations])
    return dict(
        verified=bool(remaining < 0.4),
        reason="origin_departed" if remaining < 0.4 else "object_still_at_origin",
        unchanged_fraction=float(remaining),
        observations=observations,
    )


def pickup_interval(object_track, gripper_track, proposal, confirmed_pickup, fps):
    """Bound pickup using last origin evidence and first persistent departure.

    The confirmed association can occur after an occlusion. Report the uncertainty
    interval instead of pretending that all intermediate frames were observed.
    """
    obj = np.asarray(object_track)
    grip = np.asarray(gripper_track)
    origin = np.asarray(proposal["origin"])
    radius = max(proposal["bbox"][2:]) * 0.3
    distance = np.linalg.norm(obj - origin, axis=1)
    visible = np.isfinite(obj).all(axis=1) & np.isfinite(grip).all(axis=1)
    begin = max(0, confirmed_pickup - round(fps * 2))
    at_origin = (
        np.flatnonzero(
            visible[begin:confirmed_pickup]
            & (distance[begin:confirmed_pickup] < radius)
        )
        + begin
    )
    last = int(at_origin[-1]) if len(at_origin) else begin
    first = confirmed_pickup
    for t in range(last + 1, confirmed_pickup + 1):
        end = min(len(obj), t + round(fps))
        measured = visible[t:end]
        departed = distance[t:end] > radius
        attached = (
            np.linalg.norm(obj[t:end] - grip[t:end], axis=1)
            < max(proposal["bbox"][2:]) * 3
        )
        if (
            visible[t]
            and measured.sum() >= 3
            and np.mean(departed[measured]) > 0.9
            and np.mean(attached[measured]) > 0.6
        ):
            first = t
            break
    return dict(
        pickup_frame=int(first),
        last_confirmed_at_origin=last,
        first_confirmed_attachment=int(confirmed_pickup),
        uncertainty_frames=[last + 1, int(confirmed_pickup)],
    )


def attachment_visibility(
    object_track, gripper_track, robot_masks, start, stop, *, object_radius_px
):
    """Explain missing observations using a latent attachment projection.

    Latent coordinates only test occlusion against robot pixels. They are never
    returned as measured points or used in motion-correlation scoring. Partial
    footprint overlap is only occlusion-consistent evidence, assigned half the
    weight of an observation or a projection inside the robot silhouette.
    """
    obj = np.asarray(object_track)
    grip = np.asarray(gripper_track)
    visible = np.isfinite(obj).all(axis=1)
    jointly_visible = visible & np.isfinite(grip).all(axis=1)
    ids = np.flatnonzero(
        jointly_visible & (np.arange(len(obj)) >= start) & (np.arange(len(obj)) < stop)
    )
    supported = np.zeros(len(obj), bool)
    partial = np.zeros(len(obj), bool)
    if len(ids) >= 2:
        offset = obj[ids] - grip[ids]
        predicted = grip + np.column_stack(
            [np.interp(np.arange(len(obj)), ids, offset[:, axis]) for axis in [0, 1]]
        )
        h, w = robot_masks.shape[1:]
        for t in range(start, stop):
            if visible[t] or not np.isfinite(predicted[t]).all():
                continue
            x, y = np.rint(predicted[t]).astype(int)
            if 0 <= x < w and 0 <= y < h:
                neighborhood = robot_masks[
                    t, max(0, y - 4) : min(h, y + 5), max(0, x - 4) : min(w, x + 5)
                ]
                supported[t] = neighborhood.mean() > 0.6
                if not supported[t] and robot_masks[t].any():
                    distance = cv2.distanceTransform(
                        (~robot_masks[t]).astype(np.uint8), cv2.DIST_L2, 3
                    )[y, x]
                    partial[t] = distance < object_radius_px

    return dict(
        visible_fraction=float(visible[start:stop].mean()),
        robot_occluded_fraction=float(supported[start:stop].mean()),
        partial_occlusion_consistent_fraction=float(partial[start:stop].mean()),
        explained_fraction=float((visible | supported | partial)[start:stop].mean()),
        evidence_confidence=float(
            (visible.astype(float) + supported + 0.5 * partial)[start:stop].mean()
        ),
        object_radius_px=float(object_radius_px),
    )


def color_support(hsv, mask, proposal):
    """Visible chromatic support, independent of SAM's predicted object centre."""
    hue = np.abs(hsv[:, :, 0].astype(float) - proposal["color"][0])
    hue = np.minimum(hue, 180 - hue)
    matching = mask & (hue < 12) & (hsv[:, :, 1] > 40) & (hsv[:, :, 2] > 20)
    return int(matching.sum()), float(matching.sum() / max(1, mask.sum()))


def validate_track_colors(frames, proposals, tracks, robots):
    """Do not count a mask on a different surface as an observed colored object."""
    n, h, w = frames.shape[:3]
    issues = np.zeros((len(tracks), n), bool)
    visible = np.ones((len(tracks), n), bool)
    for t, frame in enumerate(frames):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        robot = np.unpackbits(robots[t], axis=-1, count=w).astype(bool).any(axis=0)
        for k, (proposal, track) in enumerate(zip(proposals, tracks)):
            if proposal["appearance_model"] != "hue":
                continue
            mask = np.unpackbits(track["packed_masks"][t], axis=-1, count=w).astype(
                bool
            )
            count, fraction = color_support(hsv, mask, proposal)
            observed = count >= 8 and fraction >= 0.05
            visible[k, t] = observed
            if observed:
                continue
            track["centers"][t] = np.nan
            if mask.any():
                occluded = (mask & robot).sum() / mask.sum() >= 0.5
                if not occluded:
                    issues[k, t] = True
                    track["packed_masks"][t] = 0
                    if "areas" in track:
                        track["areas"][t] = 0
    return issues, visible
