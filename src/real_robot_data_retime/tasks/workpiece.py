PROFILE = dict(
    name="workpiece",
    camera="observation.images.right_environment_1",
    expected_objects=4,
    object_kind="dark",
    dependencies=[],
    priority="alternating_pickups",
)


def discover_bins(frame):
    import cv2
    from ..interaction.video import components

    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    wood = (
        (hsv[:, :, 0] > 5)
        & (hsv[:, :, 0] < 35)
        & (hsv[:, :, 1] > 50)
        & (hsv[:, :, 2] > 65)
    )
    wood[: int(h * 0.35)] = False
    bins = [
        dict(bbox=stat[:4].tolist(), mask=mask, center=center)
        for mask, stat, center in components(wood, int(h * w * 0.01))
        if stat[2] > w * 0.15 and (center[0] < w * 0.25 or center[0] > w * 0.75)
    ]
    bins = sorted(bins, key=lambda b: b["bbox"][2] * b["bbox"][3], reverse=True)[:2]
    return sorted(bins, key=lambda b: b["center"][0])


def bin_visit(gripper, pickup, stop, bin_box, fps):
    import numpy as np
    from scipy.ndimage import median_filter
    from ..interaction.evidence import stable_runs

    x, y, w, h = bin_box
    pad = max(w, h) * 0.25
    xy = np.asarray(gripper)
    finite = np.isfinite(xy).all(axis=1)
    near = (
        finite
        & (xy[:, 0] > x - pad)
        & (xy[:, 0] < x + w + pad)
        & (xy[:, 1] > y - pad)
        & (xy[:, 1] < y + h + pad)
    )
    spatial_near = near.copy()
    near[: pickup + 3] = False
    near[stop:] = False
    visits = stable_runs(near, max(3, round(fps * 0.15)))
    if not visits:
        return None
    start, end = visits[0]
    outside = np.flatnonzero(finite[end:stop] & ~spatial_near[end:stop]) + end
    if not len(outside):
        return None
    clear_frame = int(outside[0])
    velocity = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    quiet = (
        median_filter(np.where(np.isfinite(velocity), velocity, np.inf), size=5)
        < max(w, h) * 0.015
    )
    pauses = stable_runs(quiet[start : end - 1], 3)
    release = start + pauses[-1][1] if pauses else end - 1
    return dict(
        entry_frame=start,
        release_frame=release,
        clearance_frame=clear_frame,
        release_interval=[start, end],
    )


def verify_deposit(frames, robots, pickup, visit, bin_box, fps):
    import cv2
    import numpy as np

    x, y, w, h = map(int, bin_box)
    inset = max(2, round(min(w, h) * 0.03))
    roi = np.zeros(frames.shape[1:3], bool)
    roi[y + inset : y + h - inset, x + inset : x + w - inset] = True
    coverage = np.mean(robots[:, roi], axis=1)
    before = np.flatnonzero(coverage[:pickup] < 0.1)
    after = (
        np.flatnonzero(coverage[visit["clearance_frame"] :] < 0.1)
        + visit["clearance_frame"]
    )
    if not len(before) or len(after) < 3:
        return dict(verified=False, reason="bin_not_observed_clear")
    reference = int(before[-1])
    samples = after[:: max(1, round(fps * 0.2))][:5]
    if len(samples) < 3:
        return dict(verified=False, reason="insufficient_post_deposit_evidence")
    first = frames[reference].astype(float)
    changed = []
    for t in samples:
        current = frames[t].astype(float)
        correction = np.median(current[roi] - first[roi], axis=0)
        darkening = np.mean(first + correction - current, axis=-1)
        change = (darkening > 20) & roi & ~robots[t]
        change = cv2.morphologyEx(
            change.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
        ).astype(bool)
        changed.append(change)
    persistent = np.sum(changed, axis=0) >= max(3, len(changed) - 1)
    pixels = int(persistent.sum())
    return dict(
        verified=pixels >= 12,
        changed_pixels=pixels,
        reference_frame=reference,
        observed_frames=samples.tolist(),
        method="persistent_bin_appearance_change",
        reason="destination_changed"
        if pixels >= 12
        else "no_persistent_destination_change",
    )


def alternating_pickups(events):
    """Return the four source pickup milestones in shared-workspace order."""
    own = [
        sorted(
            (e for e in events if e["robot_id"] == side),
            key=lambda e: e["pickup_frame"],
        )
        for side in ("left", "right")
    ]
    if [len(group) for group in own] != [2, 2]:
        raise ValueError("workpiece scheduling requires two pickups per arm")
    return [
        (side, own[side][cycle]["pickup_frame"])
        for cycle in range(2)
        for side in range(2)
    ]


def pickup_precedence(left, right, milestones):
    clocks = (left, right)
    return all(
        clocks[side] < frame or clocks[previous_side] > previous_frame
        for (previous_side, previous_frame), (side, frame) in zip(
            milestones, milestones[1:]
        )
    )


def dark_object_masks(value, minimum_area, maximum_area):
    """Separate dark objects joined only by a brighter shadow saddle.

    Normal-size components retain their original support. Oversized workspace
    components need two independently large dark cores before being divided;
    nearest-core ownership retains their original object pixels for SAM prompts.
    """
    import numpy as np
    from scipy.ndimage import distance_transform_edt

    from ..interaction.video import components

    h, w = value.shape
    mask = value < 95
    divided = []
    for region, stat, center in components(mask, minimum_area, maximum_area):
        if not (
            w * 0.2 < center[0] < w * 0.8
            and center[1] > h * 0.48
            and max(stat[2:4]) > w * 0.15
        ):
            continue
        for threshold in np.unique(value[region])[::-1]:
            cores = [
                core
                for core, bounds, _ in components(
                    region & (value < threshold), minimum_area, maximum_area
                )
                if max(bounds[2:4]) <= w * 0.15
            ]
            if len(cores) < 2:
                continue
            labels = np.zeros((h, w), np.int32)
            for index, core in enumerate(cores, 1):
                labels[core] = index
            nearest = distance_transform_edt(
                labels == 0, return_distances=False, return_indices=True
            )
            owners = labels[tuple(nearest)]
            divided.extend(
                region & (owners == index) for index in range(1, len(cores) + 1)
            )
            mask[region] = False
            break
    return [mask, *divided]


def bin_aware_audit_support(frames, robots, motion, support):
    """Exclude independently observed stationary bin appearance from robot evidence."""
    import numpy as np
    from ..background.clean_plate import match_background_colors

    n, h, w = frames.shape[:3]
    bins = discover_bins(frames[0])
    if len(bins) != 2:
        raise ValueError("both destination bins must be observed for scene auditing")
    result = support.copy()
    motion = motion.copy()
    records = []
    for bin in bins:
        x, y, bw, bh = map(int, bin["bbox"])
        # Include the same three-pixel rim used by the scene compositor.
        x1, y1 = min(w, x + bw + 3), min(h, y + bh + 3)
        x, y = max(0, x - 3), max(0, y - 3)
        bw, bh = x1 - x, y1 - y
        bin_robots = np.array(
            [
                np.unpackbits(robots[t], axis=-1, count=w)[:, y:y1, x:x1].any(axis=0)
                for t in range(n)
            ]
        )
        clear = np.flatnonzero(bin_robots.mean(axis=(1, 2)) < 0.01)
        if len(clear) < 5:
            records.append(
                dict(bbox=[x, y, bw, bh], reference_frames=[], excluded_pixel_frames=0)
            )
            continue
        observed = clear[-5:]
        common = (~bin_robots & ~bin_robots[observed[-1]]).sum(axis=(1, 2))
        eligible = np.flatnonzero(common >= 100)
        indices = np.r_[eligible, observed[-1]]
        normalized, _ = match_background_colors(
            frames[indices, y:y1, x:x1], bin_robots[indices]
        )
        references = normalized[np.searchsorted(eligible, observed)].astype(float)
        reference = np.median(references, axis=0)
        stable = np.max(np.ptp(references, axis=0), axis=-1) <= 12
        excluded = 0
        for position, t in enumerate(eligible):
            stationary = stable & (
                np.max(np.abs(normalized[position].astype(float) - reference), axis=-1)
                <= 12
            )
            region = result[t, y:y1, x:x1]
            excluded += int((region & stationary).sum())
            region[stationary] = False
            motion[t, y:y1, x:x1][stationary] = 0
        records.append(
            dict(
                bbox=[x, y, bw, bh],
                reference_frames=observed.tolist(),
                excluded_pixel_frames=excluded,
            )
        )
    return motion, result, records
