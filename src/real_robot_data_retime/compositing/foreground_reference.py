"""Register a nearby real arm image using only mutually visible arm features."""

import cv2
import numpy as np


class ForegroundRegistrationError(ValueError):
    """An observed reference failed explicit registration quality checks."""


def register_foreground(reference, target, reference_mask, target_mask):
    gray = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in (target, reference)]
    mask = cv2.erode(target_mask.astype(np.uint8), np.ones((7, 7), np.uint8))
    points = cv2.goodFeaturesToTrack(gray[0], 300, 0.01, 4, mask=mask)
    if points is None or len(points) < 8:
        raise ForegroundRegistrationError(
            "insufficient visible arm features for foreground registration"
        )
    tracked, valid, _ = cv2.calcOpticalFlowPyrLK(gray[0], gray[1], points, None)
    back, returned, _ = cv2.calcOpticalFlowPyrLK(gray[1], gray[0], tracked, None)
    a, b = points[:, 0], tracked[:, 0]
    h, w = target.shape[:2]
    xy = np.rint(b).astype(int)
    keep = (
        (valid[:, 0] > 0)
        & (returned[:, 0] > 0)
        & (np.linalg.norm(back[:, 0] - a, axis=1) < 0.75)
    )
    keep &= (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
    ids = np.flatnonzero(keep)
    ids = ids[reference_mask[xy[ids, 1], xy[ids, 0]]]
    if len(ids) < 8:
        raise ForegroundRegistrationError(
            "insufficient matching visible foreground features"
        )
    transform, inliers = cv2.estimateAffinePartial2D(
        b[ids], a[ids], method=cv2.RANSAC, ransacReprojThreshold=1.0
    )
    if transform is None or inliers.sum() < 8:
        raise ForegroundRegistrationError("foreground registration failed")
    ids = ids[inliers[:, 0].astype(bool)]
    error = np.linalg.norm(
        b[ids] @ transform[:, :2].T + transform[:, 2] - a[ids], axis=1
    )
    scale = np.linalg.norm(transform[:, 0])
    angle = np.degrees(np.arctan2(transform[1, 0], transform[0, 0]))
    displacement = np.linalg.norm(
        b[ids] @ transform[:, :2].T + transform[:, 2] - b[ids], axis=1
    )
    if (
        error.max() > 1.5
        or not 0.95 <= scale <= 1.05
        or abs(angle) > 5
        or displacement.max() > 12
    ):
        raise ForegroundRegistrationError(
            f"foreground registration exceeds bounds: scale={scale}, angle={angle}, translation={transform[:, 2]}, error={error.max()}"
        )
    image = cv2.warpAffine(reference, transform, (w, h), flags=cv2.INTER_LINEAR)
    support = cv2.warpAffine(
        reference_mask.astype(np.uint8), transform, (w, h), flags=cv2.INTER_NEAREST
    ).astype(bool)
    return (
        image,
        support,
        {
            "transform": transform.tolist(),
            "inlier_features": len(ids),
            "max_reprojection_error_px": float(error.max()),
            "max_visible_feature_displacement_px": float(displacement.max()),
        },
    )
