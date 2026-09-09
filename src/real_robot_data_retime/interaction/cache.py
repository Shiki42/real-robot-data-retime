"""Content-addressed expensive-stage cache with explicit implementation hashes."""

import hashlib
import marshal
import json
import uuid
from pathlib import Path
import numpy as np


def stage_key(frames, functions, parameters):
    digest = hashlib.sha256()
    digest.update(str((frames.shape, str(frames.dtype), parameters)).encode())
    digest.update(memoryview(np.ascontiguousarray(frames)))
    for function in functions:
        digest.update(marshal.dumps(function.__code__))
    return digest.hexdigest()


def gripper_cache(frames, geometry, sam, cache_dir):
    from .discovery import motion_and_grippers
    from .neural_tracks import segment_grippers
    from ..segmentation.sam_backend import SamVideo

    digest = hashlib.sha256()
    digest.update(memoryview(np.ascontiguousarray(geometry["centers"])))
    digest.update(memoryview(np.ascontiguousarray(geometry["masks"])))
    geometry_hash = digest.hexdigest()
    key = stage_key(
        frames,
        [motion_and_grippers, segment_grippers, SamVideo.propagate],
        (sam.model_id, sam.model.config._commit_hash, geometry_hash),
    )
    path = Path(cache_dir) / f"grippers-{key}.npz"
    if path.exists():
        with np.load(path) as d:
            return dict(
                seeds=json.loads(str(d["seeds"])),
                centers=d["centers"],
                apertures=d["apertures"],
                masks=np.unpackbits(d["masks"], axis=-1, count=frames.shape[2]).astype(
                    bool
                ),
            )
    result = segment_grippers(frames, geometry, sam)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + f".{uuid.uuid4().hex}.tmp.npz")
    np.savez_compressed(
        temporary,
        seeds=json.dumps(result["seeds"]),
        centers=result["centers"],
        apertures=result["apertures"],
        masks=np.packbits(result["masks"], axis=-1),
    )
    temporary.replace(path)
    return result
