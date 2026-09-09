"""Observable silhouette aperture states, separate from grasp confirmation."""

import numpy as np
from scipy.ndimage import median_filter


def aperture_states(aperture, fps):
    values = np.asarray(aperture, float)
    states = np.full(len(values), "UNKNOWN", dtype="U7")
    valid = np.isfinite(values)
    if valid.sum() < 5:
        return states
    low, high = np.percentile(values[valid], [10, 90])
    span = high - low
    if span < 1:
        return states
    smooth = median_filter(
        np.interp(np.arange(len(values)), np.flatnonzero(valid), values[valid]), size=5
    )
    lag = min(len(values), max(1, round(fps * 0.1)))
    difference = smooth - np.r_[np.repeat(smooth[0], lag), smooth[:-lag]]
    states[valid & (smooth < (low + high) / 2)] = "CLOSED"
    states[valid & (smooth >= (low + high) / 2)] = "OPEN"
    states[valid & (difference < -span * 0.08)] = "CLOSING"
    states[valid & (difference > span * 0.08)] = "OPENING"
    return states
