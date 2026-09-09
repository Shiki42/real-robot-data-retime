"""Remove measured camera motion before interpreting object motion."""

import cv2
import numpy as np


def stabilize(frames):
    n, h, w = frames.shape[:3]
    aligned = np.empty_like(frames)
    aligned[0] = frames[0]
    transforms = np.repeat(np.eye(3)[None], n, axis=0)
    confidence = np.zeros(n)
    confidence[0] = 1.0
    previous = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    for t in range(1, n):
        current = cv2.cvtColor(frames[t], cv2.COLOR_BGR2GRAY)
        # Uniform spatial sampling prevents a textured robot dominating RANSAC.
        points = []
        for y in range(0, h, max(1, h // 4)):
            for x in range(0, w, max(1, w // 4)):
                crop = previous[y : min(h, y + h // 4), x : min(w, x + w // 4)]
                found = cv2.goodFeaturesToTrack(crop, 30, 0.02, 8)
                if found is not None:
                    points.append(found + np.array([[[x, y]]], dtype=np.float32))
        matrix = None
        if points:
            p = np.concatenate(points).astype(np.float32)
            q, status, error = cv2.calcOpticalFlowPyrLK(previous, current, p, None)
            back, back_status, _ = cv2.calcOpticalFlowPyrLK(current, previous, q, None)
            valid = (
                status[:, 0].astype(bool)
                & back_status[:, 0].astype(bool)
                & (np.linalg.norm(back - p, axis=-1)[:, 0] < 1.0)
            )
            if valid.sum() >= 12:
                matrix, inliers = cv2.estimateAffinePartial2D(
                    p[valid],
                    q[valid],
                    method=cv2.RANSAC,
                    ransacReprojThreshold=1.5,
                    maxIters=2000,
                )
                if matrix is not None:
                    confidence[t] = float(inliers.mean())
                    scale = np.sqrt(np.linalg.det(matrix[:, :2]))
                    if (
                        confidence[t] < 0.45
                        or not 0.98 < scale < 1.02
                        or np.linalg.norm(matrix[:, 2]) > w * 0.04
                    ):
                        matrix = None
                        confidence[t] = 0.0
        if matrix is None:
            transforms[t] = transforms[t - 1]
        else:
            step = np.eye(3)
            step[:2] = matrix
            transforms[t] = transforms[t - 1] @ np.linalg.inv(step)
        aligned[t] = cv2.warpAffine(
            frames[t], transforms[t, :2], (w, h), borderMode=cv2.BORDER_REFLECT
        )
        previous = current
    return aligned, transforms, confidence
