"""Scene objects stay below both arms; only carried objects share an arm layer."""


def is_carried(event, frame):
    return event["pickup_frame"] <= frame < event["release_frame"]


def arm_foreground(robots, objects, events, side, frame):
    mask = robots[frame, side].copy()
    for event in events:
        if event["robot_id"] == ("left", "right")[side] and is_carried(event, frame):
            mask |= objects[event["object_id"], frame]
    return mask


def complete_drawer_origins(frame, robots, objects, events):
    """Join touching color parts at rest, without merging their carried tracks."""
    import cv2
    import numpy as np

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    result = []
    for event in events:
        identity = event["object_id"]
        seed = objects[identity, 0]
        yy, xx = np.where(seed)
        if not len(xx):
            raise ValueError("missing drawer object origin")
        pixels = hsv[seed]
        valid = pixels[:, 1] > 85
        if not valid.any():
            raise ValueError("drawer object origin has no saturated evidence")
        colors = pixels[valid]
        bright = colors[:, 2] >= np.percentile(colors[:, 2], 65)
        hue = float(np.median(colors[bright, 0]))
        delta = np.abs(hsv[:, :, 0].astype(float) - hue)
        candidate = (
            (np.minimum(delta, 180 - delta) < 25)
            & (hsv[:, :, 1] > 85)
            & (hsv[:, :, 2] > 15)
        )
        radius = max(int(np.ptp(xx)) + 1, int(np.ptp(yy)) + 1)
        roi = np.zeros(seed.shape, bool)
        roi[
            max(0, yy.min() - radius) : yy.max() + radius + 1,
            max(0, xx.min() - radius) : xx.max() + radius + 1,
        ] = True
        candidate = cv2.morphologyEx(
            (candidate & roi).astype(np.uint8),
            cv2.MORPH_CLOSE,
            np.ones((3, 3), np.uint8),
        ).astype(bool)
        candidate &= ~robots[0].any(axis=0)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            candidate.astype(np.uint8), 8
        )
        complete = seed.copy()
        for index in range(1, count):
            region = labels == index
            if (region & seed).sum() >= 3 and stats[index, 4] <= seed.sum() * 4:
                complete |= region
        added = int((complete & ~seed).sum())
        objects[identity, : event["pickup_frame"]] |= complete
        result.append(dict(object_id=identity, added_origin_pixels=added))
    return result


def exclude_placed_objects(robots, objects, events):
    """Apply identical post-release ownership before planning and rendering."""
    for event in events:
        side = ("left", "right").index(event["robot_id"])
        release = event["release_frame"]
        robots[release:, side] &= ~objects[event["object_id"], release:]
    return robots
