"""RobotSeg semantic-video experiment; ambiguous arm identity stays unassigned."""

import time
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np


def split_anchored_robots(mask):
    """Assign only components supported by exactly one entry edge.

    A component touching both edges is not split at the image midpoint: that
    would invent an arm boundary where the robots overlap.
    """
    from ..interaction.video import components

    h, w = mask.shape
    sides = np.zeros((2, h, w), bool)
    for region, stat, _ in components(mask, 1):
        left = stat[0] < w * 0.08
        right = stat[0] + stat[2] > w * 0.92
        if left != right:
            sides[0 if left else 1] |= region
    return sides, mask & ~sides.any(axis=0)


class RobotSegVideo:
    def __init__(self, checkpoint, device="cuda"):
        if not Path(checkpoint).is_file():
            raise FileNotFoundError(checkpoint)
        from robotseg.build_robotseg import build_robotseg_video_predictor

        self.timings = {
            "jpeg_seconds": 0.0,
            "state_init_seconds": 0.0,
            "state_count": 0,
        }
        self.device = device
        self.model = build_robotseg_video_predictor(
            "configs/robotseg-infer.yaml",
            str(checkpoint),
            device=device,
        )
        # Keep upstream mask decoding but do not require its optional CUDA
        # connected-component extension. This is explicit, not an error fallback.
        self.model.fill_hole_area = 0

    @contextmanager
    def _state(self, frames):
        import torch

        with TemporaryDirectory(prefix="retime-robotseg-") as directory:
            started = time.monotonic()
            for t, frame in enumerate(frames):
                if not cv2.imwrite(
                    str(Path(directory) / f"{t:06d}.jpg"),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 100],
                ):
                    raise OSError("RobotSeg input frame write failed")
            self.timings["jpeg_seconds"] += time.monotonic() - started
            with (
                torch.inference_mode(),
                torch.autocast(
                    "cuda", dtype=torch.bfloat16, enabled=self.device.startswith("cuda")
                ),
            ):
                started = time.monotonic()
                state = self.model.init_state(
                    video_path=directory,
                    async_loading_frames=False,
                    offload_video_to_cpu=True,
                    offload_state_to_cpu=True,
                )
                if self.device.startswith("cuda"):
                    torch.cuda.synchronize()
                self.timings["state_init_seconds"] += time.monotonic() - started
                self.timings["state_count"] += 1
                yield state

    def segment_semantic(self, frames, category="robot"):
        if category not in {"robot", "arm", "gripper"}:
            raise ValueError("unknown RobotSeg category")
        with self._state(frames) as state:
            self.model.add_new_robot(state, frame_idx=0, obj_id=0, robot=category)
            for t, ids, logits in self.model.propagate_in_video(state, robot=category):
                if list(ids) != [0]:
                    raise ValueError("unexpected RobotSeg semantic object IDs")
                yield t, (logits[0, 0] > 0).cpu().numpy()


class RobotSegPromptedVideo(RobotSegVideo):
    """Use existing automatic SAM prompts with the robot-specific checkpoint.

    Reverse propagation, like SamVideo, feeds a reversed sequence from the
    automatically selected seed. No hand-selected frame/point is introduced.
    """

    def propagate(
        self,
        frames,
        proposals,
        seed_frame=0,
        reverse=False,
        stop_frame=None,
        category="robot",
    ):
        if category not in {"robot", "arm", "gripper"}:
            raise ValueError("unknown RobotSeg category")
        if not proposals or not 0 <= seed_frame < len(frames):
            raise ValueError("missing proposals or invalid seed frame")
        endpoint = (
            (-1 if reverse else len(frames)) if stop_frame is None else stop_frame
        )
        indices = np.arange(seed_frame, endpoint, -1 if reverse else 1)
        if not len(indices):
            return
        if (indices < 0).any() or (indices >= len(frames)).any():
            raise ValueError("propagation interval outside video")
        with self._state(frames[indices]) as state:
            for k, proposal in enumerate(proposals):
                x, y, w, h = proposal["bbox"]
                positives = proposal["positive_points"]
                negatives = proposal.get("negative_points", [])
                self.model.add_new_points_or_box(
                    state,
                    frame_idx=0,
                    obj_id=k,
                    robots=category,
                    points=np.asarray([*positives, *negatives], dtype=np.float32),
                    labels=np.asarray(
                        [1] * len(positives) + [0] * len(negatives), dtype=np.int32
                    ),
                    box=np.asarray([x, y, x + w, y + h], dtype=np.float32),
                )
            for t, ids, logits in self.model.propagate_in_video(state, robot=category):
                if list(ids) != list(range(len(proposals))):
                    raise ValueError("RobotSeg changed prompted object identity order")
                yield int(indices[t]), (logits[:, 0] > 0).cpu().numpy()
