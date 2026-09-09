import numpy as np


def track_points(frames, proposals, stride=3):
    """CoTracker's visibility distinguishes observation from occlusion prediction."""
    import torch
    from huggingface_hub import hf_hub_download
    from cotracker.predictor import CoTrackerPredictor

    checkpoint = hf_hub_download("facebook/cotracker3", "scaled_offline.pth")
    model = (
        CoTrackerPredictor(checkpoint=checkpoint, window_len=60, v2=False).cuda().eval()
    )
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
    return xy, visible, np.arange(0, len(frames), stride)
