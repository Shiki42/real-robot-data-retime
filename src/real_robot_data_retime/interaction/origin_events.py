"""Persistent origin disappearance, robust to transient robot occlusion."""

import cv2
import numpy as np
from scipy.ndimage import median_filter

from .evidence import stable_runs


def origin_departure_interval(frames, proposal, robot_masks, fps):
    mask = proposal["mask"].astype(bool)
    outer = cv2.dilate(mask.astype(np.uint8), np.ones((13, 13), np.uint8)).astype(bool)
    inner = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    ring = outer & ~inner
    if mask.sum() < 8 or ring.sum() < 8:
        return None, None
    reference = np.median(frames[: min(8, len(frames))], axis=0)
    contrast = np.median(reference[mask], axis=0) - np.median(reference[ring], axis=0)
    energy = float(contrast @ contrast)
    if energy < 100:
        return None, None
    signal = []
    clear = []
    for frame, robot in zip(frames, robot_masks):
        delta = np.median(frame[mask], axis=0) - np.median(frame[ring], axis=0)
        signal.append(float(delta @ contrast / energy))
        clear.append(float(robot[mask].mean()) < 0.2)
    signal = median_filter(np.array(signal), size=5)
    clear = np.array(clear)
    present = (signal > 0.55) & clear
    present_runs = stable_runs(present, max(3, round(fps * 0.1)))
    if not present_runs:
        return None, signal
    last = present_runs[-1][1] - 1
    absent = (signal < 0.25) & clear
    absent_runs = stable_runs(absent, max(3, round(fps * 0.2)))
    after = [a for a, b in absent_runs if a > last]
    if not after:
        return None, signal
    first = min(after)
    contacts = stable_runs(~clear, max(3, round(fps * 0.1)))
    prior = [(a, b) for a, b in contacts if a < first and b <= first + round(fps * 0.2)]
    contact = prior[-1] if prior else None
    return dict(
        last_observed_at_origin=int(last),
        first_observed_empty=int(first),
        pickup_interval=[int(last + 1), int(first)],
        method="persistent_origin_disappearance",
        contact_start=None if contact is None else contact[0],
        contact_end=None if contact is None else contact[1],
    ), signal


def tracking_seed_frame(event, fps):
    """Seed before image-space contact/occlusion evidence, reusing the mask policy."""
    if event is None:
        return 0
    anchor = event["contact_start"]
    if anchor is None:
        anchor = event["last_observed_at_origin"]
    return max(0, int(anchor) - round(fps * 0.1))
