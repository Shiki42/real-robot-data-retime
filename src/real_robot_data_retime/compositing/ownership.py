"""Scene objects stay below both arms; only carried objects share an arm layer."""


def is_carried(event, frame):
    return event["pickup_frame"] <= frame < event["release_frame"]


def arm_foreground(robots, objects, events, side, frame):
    mask = robots[frame, side].copy()
    for event in events:
        if event["robot_id"] == ("left", "right")[side] and is_carried(event, frame):
            mask |= objects[event["object_id"], frame]
    return mask
