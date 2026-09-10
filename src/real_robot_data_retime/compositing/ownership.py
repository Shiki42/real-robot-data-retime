"""Scene objects stay below both arms; only carried objects share an arm layer."""

import numpy as np
from ..background.clean_plate import dilate
from ..interaction.video import components


def is_carried(event, frame):
    return event["pickup_frame"] <= frame < event["release_frame"]


def arm_foreground(robots, objects, events, side, frame):
    mask = robots[frame, side].copy()
    for event in events:
        if event["robot_id"] == ("left", "right")[side] and is_carried(event, frame):
            carried = objects[event["object_id"], frame]
            # Resolve a tracker that has also latched onto a still-visible
            # neighboring workpiece. Never cut a component attached to this arm.
            stationary = np.zeros_like(mask)
            for other in events:
                if (
                    other["object_id"] != event["object_id"]
                    and frame < other["pickup_frame"]
                ):
                    stationary |= (
                        objects[other["object_id"], frame]
                        & objects[other["object_id"], 0]
                    )
            stationary &= ~robots[frame].any(axis=0)
            if np.any(carried & stationary):
                carried = carried.copy()
                attachment = dilate(robots[frame, side], 3)
                for region, _, _ in components(carried, 1):
                    if np.any(region & stationary) and not np.any(region & attachment):
                        carried[region] = False
            mask |= carried
    return mask
