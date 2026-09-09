"""Official BootsTAPIR PyTorch inference, with explicit observation semantics."""

import cv2
import numpy as np

TAPNET_REVISION = "5d3fb48e76c5422841e38501514121e251beabb7"


def sample_object_points(proposals, points_per_object=32):
    """Deterministic interior samples; retain object membership for every track."""
    if points_per_object < 1:
        raise ValueError("points_per_object must be positive")
    queries, owners = [], []
    for owner, proposal in enumerate(proposals):
        mask = np.asarray(proposal["mask"], dtype=np.uint8)
        interior = cv2.erode(mask, np.ones((3, 3), np.uint8))
        yy, xx = np.where(interior)
        if not len(xx):
            raise ValueError(f"object {owner} has no trackable interior")
        chosen = np.linspace(0, len(xx) - 1, min(points_per_object, len(xx)), dtype=int)
        queries.extend(np.column_stack([np.zeros(len(chosen)), yy[chosen], xx[chosen]]))
        owners.extend([owner] * len(chosen))
    if not queries:
        raise ValueError("no automatic object proposals")
    return np.asarray(queries, np.float32), np.asarray(owners, np.int32)


def model_queries(queries, source_shape, resolution):
    """TAPIR uses (t,y,x), while returned tracks use (x,y)."""
    result = np.asarray(queries, np.float32).copy()
    h, w = source_shape
    result[:, 1] *= resolution / h
    result[:, 2] *= resolution / w
    return result


class BootsTapir:
    def __init__(self, checkpoint=None, device="cuda", resolution=256):
        import torch
        from huggingface_hub import hf_hub_download
        from tapnet.torch.tapir_model import TAPIR

        if resolution < 256 or resolution % 256:
            raise ValueError("resolution must be a positive multiple of 256")
        self.resolution, self.device = resolution, device
        self.checkpoint = checkpoint or hf_hub_download(
            "google/tapnet", "bootstapir_checkpoint_v2.pt", revision=TAPNET_REVISION
        )
        self.model = TAPIR(pyramid_level=1)
        self.model.load_state_dict(
            torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        )
        self.model.to(device).eval()

    def track(self, frames, queries):
        import torch

        n, h, w = frames.shape[:3]
        queries = np.asarray(queries, np.float32)
        if queries.ndim != 2 or queries.shape[1] != 3 or not len(queries):
            raise ValueError(
                "queries must have nonempty shape (points,3) in t,y,x order"
            )
        if (
            not np.isfinite(queries).all()
            or (queries < 0).any()
            or (queries[:, 0] >= n).any()
            or (queries[:, 1] >= h).any()
            or (queries[:, 2] >= w).any()
            or (queries[:, 0] != np.floor(queries[:, 0])).any()
        ):
            raise ValueError("query outside decoded video or noninteger seed frame")
        rgb = np.stack(
            [cv2.resize(f, (self.resolution,) * 2)[:, :, ::-1] for f in frames]
        )
        video = torch.from_numpy(rgb.copy()).to(self.device).float()[None] / 127.5 - 1
        points = torch.from_numpy(model_queries(queries, (h, w), self.resolution)).to(
            self.device
        )[None]
        with torch.inference_mode():
            out = self.model(video, points)
        xy = out["tracks"][0].permute(1, 0, 2).cpu().numpy()
        confidence = (
            (1 - out["occlusion"].sigmoid()) * (1 - out["expected_dist"].sigmoid())
        )[0]
        visible = confidence.T.cpu().numpy() > 0.5
        xy *= np.array([w, h]) / self.resolution
        visible &= np.isfinite(xy).all(axis=-1)
        visible &= (
            (xy[..., 0] >= 0) & (xy[..., 0] < w) & (xy[..., 1] >= 0) & (xy[..., 1] < h)
        )
        xy[~visible] = np.nan
        return xy, visible
