"""Initialize object tracking at visually observed origin-departure events."""

import numpy as np
import cv2
from .origin_events import origin_departure_interval
from .neural_tracks import mask_measurements


def event_seeded_tracks(frames, proposals, robot_masks, sam, fps):
    n, h, w = frames.shape[:3]
    tracks = []
    events = []
    for k, p in enumerate(proposals):
        event, signal = origin_departure_interval(frames, p, robot_masks, fps)
        centers = np.full((n, 2), np.nan)
        areas = np.zeros(n)
        packed = np.zeros((n, h, (w + 7) // 8), np.uint8)
        seed = (
            max(0, event["last_observed_at_origin"] - round(fps * 0.1)) if event else 0
        )
        prefix_visible = (
            (signal[:seed] > 0.55) if signal is not None else np.zeros(seed, bool)
        )
        x, y, bw, bh = map(int, p["bbox"])
        reference = np.median(frames[: min(8, n)], axis=0).astype(np.uint8)[
            y : y + bh, x : x + bw
        ]
        for t in range(seed):
            clear = robot_masks[t][p["mask"]].mean() < 0.2
            if prefix_visible[t] and clear:
                sx, sy = max(0, x - 6), max(0, y - 6)
                search = frames[t, sy : min(h, y + bh + 6), sx : min(w, x + bw + 6)]
                correlation = cv2.matchTemplate(search, reference, cv2.TM_CCOEFF_NORMED)
                _, score, _, offset = cv2.minMaxLoc(correlation)
                if score < 0.65:
                    continue
                dx, dy = offset[0] + sx - x, offset[1] + sy - y
                centers[t] = p["origin"] + [dx, dy]
                areas[t] = p["area"]
                mask = cv2.warpAffine(
                    p["mask"].astype(np.uint8),
                    np.array([[1, 0, dx], [0, 1, dy]], float),
                    (w, h),
                )
                packed[t] = np.packbits(mask, axis=-1)
        # The static-origin hypotheses are independently checked by per-frame
        # photometric evidence and visibility; hidden frames remain unmeasured.
        prompt = dict(p)
        prompt["negative_points"] = [
            other["origin"].tolist() for j, other in enumerate(proposals) if j != k
        ]
        for t, masks in sam.propagate(frames, [prompt], seed_frame=seed):
            c, area = mask_measurements(masks[0])
            if area <= p["area"] * 4:
                centers[t] = c
                areas[t] = area
                packed[t] = np.packbits(masks[0], axis=-1)
        tracks.append(
            dict(
                proposal=p,
                centers=centers,
                areas=areas,
                packed_masks=packed,
                confidence=np.minimum(areas / max(1, p["area"]), 1.0),
            )
        )
        events.append(dict(object_id=k, seed_frame=seed, origin_event=event))
    return tracks, events
