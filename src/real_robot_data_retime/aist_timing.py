"""Conservative raw-unit timing proposals for AIST.

Taxonomy is not an input. Gripper-derived protected intervals are hypotheses,
not calibrated contact labels. Projection checks are not physical collision tests.
"""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter

from .timeline.scheduler import schedule_sources


@dataclass(frozen=True)
class TimingConfig:
    minimum_idle_seconds: float = 0.75
    retained_idle_seconds: float = 0.4
    contact_pre_guard_seconds: float = 0.5
    contact_post_guard_seconds: float = 2.0
    quantum_multiplier: float = 2.0
    relative_tolerance: float = 0.001
    active_range_ratio: float = 10.0
    minimum_relative_arm_span: float = 0.1


def ranges(mask):
    d = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(int))
    return [
        (int(a), int(b))
        for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))
    ]


def estimate_tolerance(x, config=TimingConfig()):
    x = np.asarray(x, float)
    span = np.ptp(x, axis=0)
    delta = np.abs(np.diff(x, axis=0))
    result = []
    for j in range(x.shape[1]):
        positive = delta[:, j][delta[:, j] > max(1e-9, span[j] * 1e-5)]
        quantum = float(np.quantile(positive, 0.1)) if len(positive) else 0.0
        result.append(
            max(
                1e-8,
                quantum * config.quantum_multiplier,
                span[j] * config.relative_tolerance,
            )
        )
    return np.array(result)


def contact_hypotheses(q, fps, config=TimingConfig()):
    g = median_filter(np.asarray(q)[:, [6, 13]], size=(5, 1), mode="nearest")
    hi = np.quantile(g, 0.95, axis=0)
    span = np.ptp(g, axis=0)
    potentially_closed = g < hi - np.maximum(span * 0.05, 1e-5)
    raw = [
        (a, b)
        for a, b in ranges(potentially_closed.all(1))
        if b - a >= max(3, round(fps * 0.1))
    ]
    protected = np.zeros(len(q), bool)
    pre = round(config.contact_pre_guard_seconds * fps)
    post = round(config.contact_post_guard_seconds * fps)
    for a, b in raw:
        protected[max(0, a - pre) : min(len(q), b + post)] = True
    if np.any(span < 1e-8):
        protected[:] = True
    return protected, {
        "possible_both_closed": raw,
        "protected": ranges(protected),
        "gripper_open_reference": hi.tolist(),
        "source": "relative recorded gripper apertures with conservative initial-state protection",
        "contact_verified": False,
        "constant_gripper_ambiguity": bool(np.any(span < 1e-8)),
    }


def bounded_runs(q, a, qt, at, protected):
    runs = []
    start = 0
    while start < len(q):
        if protected[start]:
            start += 1
            continue
        end = start + 1
        qmin = q[start].copy()
        qmax = qmin.copy()
        amin = a[start].copy()
        amax = amin.copy()
        while end < len(q) and not protected[end]:
            nmin = np.minimum(qmin, q[end])
            nmax = np.maximum(qmax, q[end])
            mmin = np.minimum(amin, a[end])
            mmax = np.maximum(amax, a[end])
            if np.any(nmax - nmin > qt) or np.any(mmax - mmin > at):
                break
            qmin, qmax, amin, amax = nmin, nmax, mmin, mmax
            end += 1
        runs.append((start, end))
        start = end
    return runs


def compact_idle_greedy(q, a, fps, qt, at, protected, config=TimingConfig()):
    q = np.asarray(q)
    a = np.asarray(a)
    keep = np.ones(len(q), bool)
    minimum = max(3, round(config.minimum_idle_seconds * fps))
    guard = max(1, round(config.retained_idle_seconds * fps / 2))
    qstep = np.maximum(np.max(np.abs(np.diff(q, axis=0)), axis=0), 1e-9)
    astep = np.maximum(np.max(np.abs(np.diff(a, axis=0)), axis=0), 1e-9)
    removed = []
    for lo, hi in bounded_runs(q, a, qt, at, protected):
        if hi - lo < minimum or hi - lo <= 2 * guard:
            continue
        keep[lo + guard : hi - guard] = False
        anchor = lo + guard - 1
        # Retain jitter/drift anchors when needed to preserve peak per-axis steps.
        for t in range(anchor + 1, hi - guard + 1):
            if np.any(np.abs(q[t] - q[anchor]) > qstep * 1.0001) or np.any(
                np.abs(a[t] - a[anchor]) > astep * 1.0001
            ):
                keep[t - 1] = True
                anchor = t - 1
        removed.append((lo, hi))
    keep[0] = keep[-1] = True
    return np.flatnonzero(keep), removed


def compact_idle(q, a, fps, qt, at, protected, config=TimingConfig()):
    from .idle_dp import compact_idle_optimal

    return compact_idle_optimal(q, a, fps, qt, at, protected, config)


def validate_map(q, a, index, qt, at, protected):
    index = np.asarray(index)
    if (
        len(index) < 1
        or index[0] != 0
        or index[-1] != len(q) - 1
        or np.any(np.diff(index) < 0)
    ):
        raise ValueError("Source map must retain endpoints and be monotone")
    for lo, hi in zip(index[:-1], index[1:]):
        if hi - lo <= 1:
            continue
        if protected[lo + 1 : hi].any():
            raise ValueError("Protected source frames were skipped")
        if np.any(np.ptp(q[lo : hi + 1], axis=0) > qt * 1.0001) or np.any(
            np.ptp(a[lo : hi + 1], axis=0) > at * 1.0001
        ):
            raise ValueError("Non-idle motion was skipped")


def build_proposal(
    q, a, fps, projection=None, config=TimingConfig(), shared_clock_required=None
):
    q = np.asarray(q, float)
    a = np.asarray(a, float)
    if (
        q.shape != a.shape
        or q.ndim != 2
        or q.shape[1] != 14
        or len(q) < 2
        or not np.isfinite(q).all()
        or not np.isfinite(a).all()
    ):
        raise ValueError("Expected matching finite N x 14 arrays")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid FPS")
    qt = estimate_tolerance(q, config)
    at = estimate_tolerance(a, config)
    protected, contact = contact_hypotheses(q, fps, config)
    smooth = median_filter(q, size=(5, 1), mode="nearest")
    spans = np.array([np.ptp(smooth[:, s : s + 6], axis=0).sum() for s in [0, 7]])
    active = [
        bool(
            np.any(
                np.ptp(smooth[:, s : s + 6], axis=0)
                > qt[s : s + 6] * config.active_range_ratio
            )
            and spans[k] >= max(spans.max(), 1e-9) * config.minimum_relative_arm_span
        )
        for k, s in enumerate([0, 7])
    ]
    contact["motion_span_raw"] = spans.tolist()
    contact["single_arm_hold_guard"] = False
    if sum(active) == 1:
        side = int(np.flatnonzero(active)[0])
        g = median_filter(q[:, side * 7 + 6], size=5, mode="nearest")
        hi = np.quantile(g, 0.95)
        span = np.ptp(g)
        closed = g < hi - max(span * 0.05, 1e-5)
        for lo, hi_ in ranges(closed):
            protected[
                max(0, lo - round(fps * config.contact_pre_guard_seconds)) : min(
                    len(q), hi_ + round(fps * config.contact_post_guard_seconds)
                )
            ] = True
        contact["protected"] = ranges(protected)
        contact["single_arm_hold_guard"] = True
    common, common_runs = compact_idle(q, a, fps, qt, at, protected, config)
    receipt = {
        "source_frames": len(q),
        "fps": fps,
        "config": config.__dict__,
        "q_tolerance_raw": qt.tolist(),
        "action_tolerance_raw": at.tolist(),
        "active_arms": active,
        "contact_hypotheses": contact,
        "common_clock_frames": len(common),
        "common_idle_runs": common_runs,
        "taxonomy_used": False,
        "physical_collision_verified": False,
        "camera_reconstruction_verified": False,
    }
    validate_map(q, a, common, qt, at, protected)
    if sum(active) < 2:
        receipt.update(
            output_frames=len(common), decision="common_clock_only_one_active_arm"
        )
        return common, common, common, receipt
    sources = []
    idle = []
    for s in [0, 7]:
        indices, runs_ = compact_idle(
            q[:, s : s + 7],
            a[:, s : s + 7],
            fps,
            qt[s : s + 7],
            at[s : s + 7],
            protected,
            config,
        )
        sources.append(indices)
        idle.append(runs_)
    visibility = (
        np.zeros(len(q), bool)
        if shared_clock_required is None
        else np.asarray(shared_clock_required, bool)
    )
    if visibility.shape != (len(q),):
        raise ValueError("Shared-clock mask length mismatch")
    if visibility.any():
        common_kept = np.zeros(len(q), bool)
        common_kept[common] = True
        for side in range(2):
            own = np.zeros(len(q), bool)
            own[sources[side]] = True
            own[visibility] = common_kept[visibility]
            sources[side] = np.flatnonzero(own)
    receipt["visibility_shared_clock_frames"] = int(visibility.sum())
    intervals = ranges(protected | visibility)

    def dependency(i, j):
        l, r = int(sources[0][i]), int(sources[1][j])
        for lo, hi in intervals:
            if (lo <= l < hi or lo <= r < hi) and l != r:
                return False
            if (l >= hi and r < lo) or (r >= hi and l < lo):
                return False
        return True

    def safe(i, j, ni, nj):
        l, r, nl, nr = (
            int(sources[0][i]),
            int(sources[1][j]),
            int(sources[0][ni]),
            int(sources[1][nj]),
        )
        return projection is None or projection(l, r, nl, nr)

    try:
        plan = schedule_sources(
            len(sources[0]),
            len(sources[1]),
            safe,
            dependency=dependency,
            left_priority=False,
        )
        left, right = sources[0][plan.left], sources[1][plan.right]
        for s, m in [(0, left), (7, right)]:
            validate_map(
                q[:, s : s + 7],
                a[:, s : s + 7],
                m,
                qt[s : s + 7],
                at[s : s + 7],
                protected,
            )
        for lo, hi in intervals:
            hit = ((left >= lo) & (left < hi)) | ((right >= lo) & (right < hi))
            if not np.array_equal(left[hit], right[hit]):
                raise RuntimeError("Coupled clocks diverged")
        if len(left) >= len(common):
            left = right = common
            decision = "common_clock_no_additional_gain"
        else:
            decision = (
                "candidate_projection_screened"
                if projection is not None
                else "candidate_without_projection_check"
            )
    except ValueError as error:
        left = right = common
        decision = "common_clock_fallback"
        receipt["schedule_rejection"] = str(error)
    receipt.update(
        output_frames=len(left),
        decision=decision,
        per_arm_kept_frames=list(map(len, sources)),
        per_arm_idle_runs=idle,
        additional_parallel_frames_saved=max(0, len(common) - len(left)),
    )
    return left, right, common, receipt
