"""Posterior interaction evidence from measured image trajectories."""

import numpy as np
from .discovery import InteractionConfig


def stable_runs(mask, minimum):
    changes = np.flatnonzero(np.diff(np.r_[False, np.asarray(mask, dtype=bool), False]))
    return [(int(a), int(b)) for a, b in changes.reshape(-1, 2) if b - a >= minimum]


def score_hypothesis(
    object_xy,
    gripper_xy,
    aperture,
    frame,
    scale,
    config=InteractionConfig(),
    contact_distance=None,
):
    """Require future object motion and attachment, not closing alone.

    Missing coordinates remain missing; interpolation must not manufacture
    posterior evidence. Returns component scores and explicit rejection reasons.
    """
    obj = np.asarray(object_xy, float)
    grip = np.asarray(gripper_xy, float)
    aperture = np.asarray(aperture, float)
    n = len(obj)
    window = config.evidence_frames
    if obj.shape != (n, 2) or grip.shape != (n, 2) or aperture.shape != (n,):
        raise ValueError("trajectory shape mismatch")
    if not 0 < frame < n - 2 or not np.isfinite(scale) or scale <= 0:
        raise ValueError("invalid candidate frame or image scale")
    begin = max(0, frame - window)
    end = min(n, frame + window + 1)
    valid = np.isfinite(obj).all(axis=1) & np.isfinite(grip).all(axis=1)
    edge_valid = valid[1:] & valid[:-1]
    vo = np.diff(obj, axis=0)
    vg = np.diff(grip, axis=0)
    speed = np.linalg.norm(vo, axis=1)
    disagreement = np.linalg.norm(vo - vg, axis=1)
    distance = np.linalg.norm(obj - grip, axis=1)
    proximity_distance = (
        distance if contact_distance is None else np.asarray(contact_distance, float)
    )
    pre = np.arange(begin, max(begin, frame - 2))
    post = np.arange(frame, end - 1)
    pre = pre[edge_valid[pre]]
    post = post[edge_valid[post]]
    reasons = []
    if len(pre) < 3:
        reasons.append("insufficient_pre_grasp_visibility")
    if len(post) < min(10, max(5, window // 3)):
        reasons.append("insufficient_future_visibility")
    anchor = np.median(obj[pre[:5]], axis=0) if len(pre) else obj[frame]
    displacement = np.linalg.norm(obj[1:] - anchor, axis=1)
    moving = (
        (speed > scale * 0.0015)
        & (disagreement < scale * 0.012)
        & edge_valid
        & (displacement >= scale * 0.01)
    )
    runs = stable_runs(moving[frame : end - 1], 3)
    if not runs:
        reasons.append("no_stable_correlated_pickup")
    near = (
        np.nanmin(proximity_distance[max(0, frame - 3) : frame + 4])
        if valid[max(0, frame - 3) : frame + 4].any()
        else np.inf
    )
    proximity = float(np.exp(-near / (scale * 0.08)))
    contact = float(np.exp(-near / (scale * 0.035)))
    static = (
        float(np.exp(-np.median(speed[pre]) / (scale * 0.002))) if len(pre) else 0.0
    )
    finite_ap = aperture[np.isfinite(aperture)]
    span = float(np.ptp(finite_ap)) if len(finite_ap) else 0.0
    before = aperture[max(0, frame - 8) : frame]
    after = aperture[frame : min(n, frame + 8)]
    closure = (
        (float(np.nanmedian(before) - np.nanmedian(after)) / max(span, 1))
        if np.isfinite(before).any() and np.isfinite(after).any()
        else 0.0
    )
    closure_observed = closure >= 0.1
    pickup = frame + runs[0][0] + 1 if runs else None
    # Release needs opening, object stillness AND gripper departure.
    release = None
    release_opening_observed = False
    if pickup is not None:
        opening = aperture - np.r_[np.repeat(aperture[0], 8), aperture[:-8]] > max(
            1.0, span * 0.08
        )
        stationary = (speed < scale * 0.002) & edge_valid
        departed = (np.linalg.norm(vg, axis=1) > scale * 0.003) & (
            disagreement > scale * 0.002
        )
        for a, b in stable_runs(stationary & departed, 3):
            initial = obj[pre[0]] if len(pre) else obj[frame]
            displaced = np.linalg.norm(obj[a] - initial) > scale * 0.03
            if a > pickup + 3 and displaced:
                # Stable deposition and separation can occur with tiny jaw
                # changes (e.g. letters). Opening remains corroborating evidence.
                release = a + 1
                release_opening_observed = bool(
                    opening[max(pickup, a - window) : min(n, b + 8)].any()
                )
                break
    if release is None:
        reasons.append("release_not_verified")
    transport_start = pickup if pickup is not None else frame
    transport_stop = min(end - 1, release if release is not None else end - 1)
    transport = np.arange(transport_start, transport_stop)
    transport = transport[edge_valid[transport]]
    active = transport[
        (speed[transport] > scale * 0.0015)
        | (np.linalg.norm(vg[transport], axis=1) > scale * 0.0015)
    ]
    motion = (
        float(np.clip(np.median(speed[active]) / (scale * 0.006), 0, 1))
        if len(active)
        else 0.0
    )
    correlation = (
        float(np.mean(np.exp(-disagreement[active] / (scale * 0.008))))
        if len(active)
        else 0.0
    )
    attachment = (
        float(np.mean(proximity_distance[transport] < scale * 0.12))
        if len(transport)
        else 0.0
    )
    if attachment < 0.5:
        reasons.append("insufficient_persistent_attachment")
    scores = dict(
        proximity=proximity,
        contact=contact,
        static_before=static,
        motion_onset=motion,
        motion_correlation=correlation,
        attachment=attachment,
        release_consistency=float(release is not None),
    )
    score = float(np.dot(config.weights, list(scores.values())))
    if score < config.minimum_confidence:
        reasons.append("low_posterior_score")
    return dict(
        score=score,
        components=scores,
        closure_score=closure,
        closure_observed=closure_observed,
        release_opening_observed=release_opening_observed,
        pickup_frame=pickup,
        release_frame=release,
        accepted=not reasons,
        rejection_reasons=reasons,
    )
