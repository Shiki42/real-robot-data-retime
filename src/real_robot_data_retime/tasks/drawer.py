PROFILE = dict(
    name="drawer",
    camera="observation.images.top",
    expected_objects=1,
    object_kind="saturated",
    dependencies=["open_before_insert", "withdraw_before_close"],
)


def destination_bounds(frames):
    """Locate the open-tray band below the colored cabinet, not its blue knob."""
    import cv2
    import numpy as np
    from ..interaction.video import components

    n, h, w = frames.shape[:3]
    bounds = np.full((n, 4), np.nan)
    for t, frame in enumerate(frames):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        red = (
            ((hsv[:, :, 0] < 12) | (hsv[:, :, 0] > 170))
            & (hsv[:, :, 1] > 120)
            & (hsv[:, :, 2] > 70)
        )
        regions = components(red, int(h * w * 0.005))
        if not regions:
            continue
        _, stat, _ = max(regions, key=lambda item: item[1][4])
        x, y, bw, bh, _ = stat
        bounds[t] = [
            max(0, x - bw * 0.2),
            y + bh + h * 0.03,
            min(w, x + bw * 1.7),
            min(h, y + bh + h * 0.45),
        ]
    return bounds


def release_confirmations(frames, tracks, proposals, contact_distances, fps):
    """Confirm a proposed release with a visible, separated cube in the tray."""
    import cv2
    import numpy as np
    from ..interaction.verification import color_support
    from ..interaction.evidence import stable_runs

    n, h, w = frames.shape[:3]
    bounds = destination_bounds(frames)
    good = np.zeros((len(tracks), n, 2), bool)
    in_destination = np.zeros((len(tracks), n), bool)
    for t, frame in enumerate(frames):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        x0, y0, x1, y1 = bounds[t]
        for k, (track, proposal) in enumerate(zip(tracks, proposals)):
            x, y = track["centers"][t]
            if not (x0 <= x < x1 and y0 <= y < y1):
                continue
            in_destination[k, t] = True
            mask = np.unpackbits(track["packed_masks"][t], axis=-1, count=w).astype(
                bool
            )
            count, fraction = color_support(hsv, mask, proposal)
            if count >= 8 and fraction >= 0.15:
                good[k, t] = contact_distances[k, t] > w * 0.01
    confirmations = np.full((len(tracks), n, 2), -1, np.int64)
    horizon = max(3, round(fps * 0.5))
    for k, track in enumerate(tracks):
        speed = np.linalg.norm(np.diff(track["centers"], axis=0), axis=1)
        stationary = np.r_[False, speed < w * 0.002]
        for side in [0, 1]:
            for start, stop in stable_runs(good[k, :, side] & stationary, 3):
                for t in range(max(0, start - horizon), stop):
                    if not in_destination[k, t]:
                        continue
                    confirmed = max(t, start)
                    if confirmations[k, t, side] < 0:
                        confirmations[k, t, side] = confirmed
    return confirmations
