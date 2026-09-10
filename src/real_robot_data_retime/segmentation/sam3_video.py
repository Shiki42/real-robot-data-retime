"""Isolated forward/backward SAM3 sessions and explicit arm ID binding."""

import numpy as np


def bind_root_ids(ids, masks):
    ids = np.asarray(ids)
    masks = np.asarray(masks, bool)
    if masks.ndim != 3 or len(ids) != len(masks):
        raise ValueError("Invalid seed masks")
    edge = max(1, masks.shape[-1] // 20)
    left = masks[:, :, :edge].sum(axis=(1, 2))
    right = masks[:, :, -edge:].sum(axis=(1, 2))
    pairs = [
        (float(left[i] + right[j]), i, j)
        for i in range(len(ids))
        for j in range(len(ids))
        if i != j
    ]
    if not pairs:
        raise ValueError("Two distinct roots required")
    _, i, j = max(pairs)
    if (
        left[i] <= 0
        or right[j] <= 0
        or left[i] <= 2 * right[i]
        or right[j] <= 2 * left[j]
    ):
        raise ValueError("Ambiguous camera-edge roots")
    return {int(ids[i]): 0, int(ids[j]): 1}


def track_two_roots(
    predictor, resource_path, seed_frame, num_frames, height, width, progress=None
):
    robots = np.zeros((num_frames, 2, height, (width + 7) // 8), np.uint8)
    valid = np.zeros((num_frames, 2), bool)
    visited = np.zeros(num_frames, bool)
    receipts = []
    for direction in ["forward", "backward"]:
        sid = predictor.handle_request(
            {
                "type": "start_session",
                "resource_path": str(resource_path),
                "offload_video_to_cpu": True,
                "offload_state_to_cpu": True,
            }
        )["session_id"]
        try:
            seed = predictor.handle_request(
                {
                    "type": "add_prompt",
                    "session_id": sid,
                    "frame_index": seed_frame,
                    "text": "robotic arm",
                    "output_prob_thresh": 0.3,
                }
            )
            mapping = bind_root_ids(
                seed["outputs"]["out_obj_ids"], seed["outputs"]["out_binary_masks"]
            )
            unknown = set()
            count = 0

            def consume(reply):
                nonlocal count
                t = int(reply["frame_index"])
                if not 0 <= t < num_frames:
                    raise ValueError("Frame outside source")
                visited[t] = True
                count += 1
                o = reply["outputs"]
                for oid, mask in zip(o["out_obj_ids"], o["out_binary_masks"]):
                    oid = int(oid)
                    if oid not in mapping:
                        unknown.add(oid)
                        continue
                    mask = np.asarray(mask, bool)
                    if mask.shape != (height, width):
                        raise ValueError("Mask shape mismatch")
                    s = mapping[oid]
                    robots[t, s] = np.packbits(mask, axis=-1)
                    valid[t, s] = mask.any()
                if progress and count % 25 == 0:
                    progress(direction, count, int(visited.sum()), len(unknown))

            consume(seed)
            for reply in predictor.handle_stream_request(
                {
                    "type": "propagate_in_video",
                    "session_id": sid,
                    "start_frame_index": seed_frame,
                    "propagation_direction": direction,
                    "output_prob_thresh": 0.3,
                }
            ):
                consume(reply)
            receipts.append(
                {
                    "direction": direction,
                    "id_mapping": mapping,
                    "frames_received": count,
                    "unknown_ids": sorted(unknown),
                }
            )
        finally:
            predictor.handle_request({"type": "close_session", "session_id": sid})
    if not visited.all():
        raise RuntimeError("Unvisited source frames")
    return robots, valid, visited, receipts
