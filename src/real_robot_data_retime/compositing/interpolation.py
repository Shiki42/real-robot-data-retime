"""Bidirectional optical-flow interpolation of an arm and its carried object."""

from functools import lru_cache
import cv2
import numpy as np


class FlowFrames:
    def __init__(self, frames):
        self.frames = frames
        self.flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        h, w = frames.shape[1:3]
        self.grid = np.stack(np.meshgrid(np.arange(w), np.arange(h)), axis=-1).astype(
            np.float32
        )

    @lru_cache(maxsize=4)
    def pair(self, lo):
        a, b = self.frames[lo : lo + 2]
        gray = [cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in (a, b)]
        return (
            self.flow.calc(gray[0], gray[1], None),
            self.flow.calc(gray[1], gray[0], None),
        )

    def sample(self, source, masks):
        lo, hi = int(np.floor(source)), int(np.ceil(source))
        if lo == hi:
            return self.frames[lo], masks[0].astype(bool)
        alpha = source - lo
        forward, backward = self.pair(lo)
        # Solve inverse maps with fixed-point iterations; forward flow lives
        # in the first image and backward flow lives in the second image.
        maps = []
        for flow, weight in ((forward, alpha), (backward, 1 - alpha)):
            mapping = self.grid - weight * flow
            for _ in range(3):
                sampled = cv2.remap(
                    flow,
                    mapping,
                    None,
                    cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE,
                )
                mapping = self.grid - weight * sampled
            maps.append(mapping)
        colors, support = [], []
        for frame, mask, mapping in zip(
            (self.frames[lo], self.frames[hi]), masks, maps
        ):
            support.append(
                cv2.remap(mask.astype(np.float32), mapping, None, cv2.INTER_LINEAR)
            )
            colors.append(
                cv2.remap(
                    frame.astype(np.float32) * mask[..., None],
                    mapping,
                    None,
                    cv2.INTER_LINEAR,
                )
            )
        weights = (1 - alpha) * support[0] + alpha * support[1]
        premultiplied = (1 - alpha) * colors[0] + alpha * colors[1]
        image = (
            np.rint(premultiplied / np.maximum(weights[..., None], 1e-6))
            .clip(0, 255)
            .astype(np.uint8)
        )
        return image, weights >= 0.5
