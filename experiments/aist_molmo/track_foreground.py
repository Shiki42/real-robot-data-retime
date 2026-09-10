"""Pinned Molmo pilot. Experimental; no physical-success claim."""

import hashlib
import json
import time

import cv2
import numpy as np
import torch
from _common import checkpoint_path, run_stage, work_dir
from skimage.graph import MCP_Geometric


def split_union(mask):
    small = cv2.resize(
        mask.astype(np.uint8), (160, 90), interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    yy, xx = np.indices(small.shape)
    work = small.copy()
    work[81:] = False
    seeds = []
    for side, rx in [(0, 35), (1, 120)]:
        pts = np.argwhere(work & (yy > 49) & (xx < 72 if side == 0 else xx > 88))
        if not len(pts):
            return np.zeros((2, 360, 640), bool)
        seeds.append(
            tuple(pts[np.argmin((pts[:, 0] - 76) ** 2 + (pts[:, 1] - rx) ** 2)])
        )
    costs = np.where(work, 1.0, np.inf)
    dl = MCP_Geometric(costs).find_costs(starts=[seeds[0]])[0]
    dr = MCP_Geometric(costs).find_costs(starts=[seeds[1]])[0]
    labels = np.zeros((90, 160), np.uint8)
    labels[work & np.isfinite(dl) & (dl <= dr)] = 1
    labels[work & np.isfinite(dr) & (dr < dl)] = 2
    labels[(yy >= 81) & small & (xx < 80)] = 1
    labels[(yy >= 81) & small & (xx >= 80)] = 2
    labels = cv2.resize(labels, (640, 360), interpolation=cv2.INTER_NEAREST)
    return np.array([(labels == 1) & mask, (labels == 2) & mask])


def main(args):
    """Run the pinned Molmo pilot stage; see README.md for input contracts."""
    from sam3.model.sam3_video_predictor import Sam3VideoPredictor

    P = work_dir()
    root = P / "experiments/molmo_video_v4"
    root.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    manifest = json.loads((P / "data/molmo_preview/manifest.json").read_text())
    for record in manifest["episodes"]:
        ep, n, sha = (int(record["episode"]), int(record["frames"]), record["sha256"])
        d = root / f"ep{ep}"
        d.mkdir(exist_ok=True)
        src = P / f"data/molmo_preview/ep{ep}/source.mp4"
        partial = src.with_suffix(".mp4.uploading")
        if not src.exists():
            if hashlib.sha256(partial.read_bytes()).hexdigest() != sha:
                raise RuntimeError("Source transfer hash mismatch")
            partial.rename(src)
        if hashlib.sha256(src.read_bytes()).hexdigest() != sha:
            raise RuntimeError("Final hash mismatch")
        cap = cv2.VideoCapture(str(src))
        frames = d / "source_jpeg"
        frames.mkdir(exist_ok=True)
        count = 0
        while True:
            ok, im = cap.read()
            if not ok:
                break
            if im.shape[:2] != (360, 640):
                raise ValueError("This pilot requires 640x360 clips")
            if not (frames / f"{count:05d}.jpg").exists():
                cv2.imwrite(
                    str(frames / f"{count:05d}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 95]
                )
            count += 1
        cap.release()
        if count != n:
            raise RuntimeError(f"Frame count {count} != {n}")
        if (d / "robots.npz").exists():
            continue
        seed = n // 2
        pred = Sam3VideoPredictor(
            checkpoint_path=str(checkpoint_path()), async_loading_frames=True
        )
        packed = np.zeros((n, 2, 360, 80), np.uint8)
        valid = np.zeros((n, 2), bool)
        visited = np.zeros(n, bool)
        tic = time.time()
        try:
            for direction in ["forward", "backward"]:
                sid = pred.handle_request(
                    dict(
                        type="start_session",
                        resource_path=str(frames),
                        offload_video_to_cpu=True,
                        offload_state_to_cpu=True,
                    )
                )["session_id"]
                try:
                    reply = pred.handle_request(
                        dict(
                            type="add_prompt",
                            session_id=sid,
                            frame_index=seed,
                            text="robotic arm",
                            output_prob_thresh=0.3,
                        )
                    )

                    def consume(reply):
                        t = int(reply["frame_index"])
                        visited[t] = True
                        masks = np.asarray(reply["outputs"]["out_binary_masks"], bool)
                        if len(masks):
                            masks = split_union(masks.any(0))
                            for side, m in enumerate(masks):
                                packed[t, side] = np.packbits(m, axis=-1)
                                valid[t, side] = m.sum() > 40
                        if visited.sum() % 100 == 0:
                            (d / "track_status.json").write_text(
                                json.dumps(
                                    dict(
                                        state="tracking",
                                        visited=int(visited.sum()),
                                        total=n,
                                        direction=direction,
                                        elapsed=time.time() - tic,
                                    )
                                )
                            )

                    consume(reply)
                    for reply in pred.handle_stream_request(
                        dict(
                            type="propagate_in_video",
                            session_id=sid,
                            start_frame_index=seed,
                            propagation_direction=direction,
                            output_prob_thresh=0.3,
                        )
                    ):
                        consume(reply)
                finally:
                    pred.handle_request(dict(type="close_session", session_id=sid))
        finally:
            pred.shutdown()
        if not visited.all():
            raise RuntimeError("Unvisited frames")
        np.savez_compressed(d / "robots.npz", robots=packed, valid=valid, seed=seed)
        (d / "track_status.json").write_text(
            json.dumps(
                dict(
                    state="complete",
                    frames=n,
                    coverage=float(valid.all(1).mean()),
                    elapsed=time.time() - tic,
                )
            )
        )
        print("TRACKED", ep, n, float(valid.all(1).mean()), flush=True)


if __name__ == "__main__":
    run_stage(main, renderer=False, segmentation=True)
