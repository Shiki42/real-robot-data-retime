"""Register persistent background descriptors without accumulating frame drift."""

import cv2
import numpy as np


def stabilize(frames):
    n, h, w = frames.shape[:3]
    orb = cv2.ORB_create(nfeatures=1200, fastThreshold=10, edgeThreshold=10)
    initial, descriptor = orb.detectAndCompute(
        cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY), None
    )
    transforms = np.repeat(np.eye(3)[None], n, axis=0)
    confidence = np.zeros(n)
    confidence[0] = 1.0
    if descriptor is None or len(initial) < 8:
        return frames.copy(), transforms, confidence
    origin = np.array([p.pt for p in initial], np.float32)
    trajectories = np.full((n, len(initial), 2), np.nan, np.float32)
    trajectories[0] = origin
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    for t in range(1, n):
        points, current = orb.detectAndCompute(
            cv2.cvtColor(frames[t], cv2.COLOR_BGR2GRAY), None
        )
        if current is None or len(current) < 2:
            continue
        for matches in matcher.knnMatch(descriptor, current, k=2):
            if len(matches) != 2:
                continue
            best, second = matches
            if best.distance < 64 and best.distance < 0.75 * second.distance:
                trajectories[t, best.queryIdx] = points[best.trainIdx].pt
    visible = np.isfinite(trajectories).all(axis=-1)
    persistent = visible.mean(axis=0) > 0.2
    for k in np.flatnonzero(persistent):
        xy = trajectories[visible[:, k], k]
        span = np.linalg.norm(
            np.percentile(xy, 95, axis=0) - np.percentile(xy, 5, axis=0)
        )
        if span > w * 0.04:
            persistent[k] = False
    aligned = np.empty_like(frames)
    for t in range(n):
        valid = persistent & visible[t]
        if valid.sum() >= 8:
            delta = trajectories[t, valid] - origin[valid]
            shift = np.median(delta, axis=0)
            inliers = np.linalg.norm(delta - shift, axis=1) < 1.5
            if inliers.sum() >= 8:
                transforms[t, :2, 2] = -np.median(delta[inliers], axis=0)
                confidence[t] = float(inliers.mean())
        aligned[t] = cv2.warpAffine(
            frames[t], transforms[t, :2], (w, h), borderMode=cv2.BORDER_REFLECT
        )
    return aligned, transforms, confidence
