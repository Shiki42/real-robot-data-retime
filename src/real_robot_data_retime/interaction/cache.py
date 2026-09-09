"""Content-addressed expensive-stage cache with explicit implementation hashes."""

import hashlib
from importlib.metadata import version
from types import CodeType
import json
import uuid
from pathlib import Path
import numpy as np


def implementation_key(code):
    constants = tuple(
        implementation_key(x) if isinstance(x, CodeType) else x for x in code.co_consts
    )
    return (
        code.co_code,
        constants,
        code.co_names,
        code.co_varnames,
        code.co_freevars,
        code.co_argcount,
        code.co_kwonlyargcount,
        code.co_flags,
    )


def stage_key(frames, functions, parameters):
    digest = hashlib.sha256()
    digest.update(str((frames.shape, str(frames.dtype), parameters)).encode())
    digest.update(memoryview(np.ascontiguousarray(frames)))
    for function in functions:
        digest.update(
            repr(
                (
                    implementation_key(function.__code__),
                    function.__defaults__,
                    function.__kwdefaults__,
                )
            ).encode()
        )
    return digest.hexdigest()


def gripper_cache(frames, geometry, sam, cache_dir):
    from .discovery import motion_and_grippers
    from .gripper_geometry import end_effector
    from .neural_tracks import segment_grippers, segment_robots, propagate_robot
    from .robot_discovery import (
        robot_prompt,
        prompt_from_robot_region,
        robot_entry_side,
    )
    from ..segmentation.sam_backend import SamVideo

    digest = hashlib.sha256()
    digest.update(memoryview(np.ascontiguousarray(geometry["centers"])))
    digest.update(memoryview(np.ascontiguousarray(geometry["masks"])))
    geometry_hash = digest.hexdigest()
    key = stage_key(
        frames,
        [
            motion_and_grippers,
            segment_grippers,
            end_effector,
            segment_robots,
            propagate_robot,
            robot_prompt,
            robot_entry_side,
            prompt_from_robot_region,
            SamVideo.propagate,
        ],
        (
            sam.model_id,
            sam.model.config._commit_hash,
            geometry_hash,
            version("torch"),
            version("transformers"),
        ),
    )
    path = Path(cache_dir) / f"grippers-{key}.npz"
    if path.exists():
        with np.load(path) as d:
            return dict(
                seeds=json.loads(str(d["seeds"])),
                robot_masks=d["robot_masks"],
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
        robot_masks=result["robot_masks"],
        centers=result["centers"],
        apertures=result["apertures"],
        masks=np.packbits(result["masks"], axis=-1),
    )
    temporary.replace(path)
    return result
