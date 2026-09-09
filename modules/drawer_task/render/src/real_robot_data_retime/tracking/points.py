import numpy as np
from functools import partial


def _track_points(frames, proposals, stride=3):
    """CoTracker's visibility distinguishes observation from occlusion prediction."""
    import torch
    from huggingface_hub import hf_hub_download
    from cotracker.predictor import CoTrackerPredictor

    from ..segmentation.model_versions import MODEL_REVISIONS

    checkpoint = hf_hub_download(
        "facebook/cotracker3",
        "scaled_offline.pth",
        revision=MODEL_REVISIONS["facebook/cotracker3"],
    )
    model = (
        CoTrackerPredictor(checkpoint=checkpoint, window_len=60, v2=False).cuda().eval()
    )
    # Keep CoTracker's full temporal window and predictor coordinate handling;
    # only bound its supported, frame-independent convolution feature batches.
    model.model.forward = partial(model.model.forward, fmaps_chunk_size=32)
    sampled = frames[::stride]
    points = torch.tensor([[0, *p["origin"]] for p in proposals], dtype=torch.float32)[
        None
    ].cuda()
    rgb = sampled[:, :, :, ::-1].copy()
    video = torch.from_numpy(rgb).permute(0, 3, 1, 2)[None].float().cuda()
    with torch.inference_mode():
        xy, visible = model(video, queries=points)
    xy = xy[0].cpu().numpy()
    visible = visible[0].cpu().numpy()
    xy[~visible] = np.nan
    del model, video, points
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    return xy, visible, np.arange(0, len(frames), stride)


def track_points(frames, proposals, stride=3):
    """Serialize the high-memory stage across local batch workers."""
    import fcntl
    from pathlib import Path

    lock_path = Path.home() / ".cache" / "real-robot-data-retime" / "cotracker.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _track_points(frames, proposals, stride)
