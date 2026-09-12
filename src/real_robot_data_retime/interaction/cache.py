"""Content-addressed expensive-stage cache with explicit implementation hashes."""

import hashlib
import json
import uuid
from importlib.metadata import version
from pathlib import Path
from types import CodeType

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
    from ..segmentation.sam_backend import SamVideo
    from .discovery import motion_and_grippers
    from .gripper_geometry import end_effector
    from .neural_tracks import segment_grippers, segment_robots
    from .robot_discovery import (
        prompt_from_robot_region,
        robot_entry_side,
        robot_prompt,
        supported_robot_prompt,
    )

    digest = hashlib.sha256()
    digest.update(memoryview(np.ascontiguousarray(geometry["centers"])))
    digest.update(memoryview(np.ascontiguousarray(geometry["masks"])))
    digest.update(memoryview(np.ascontiguousarray(geometry["prompt_support"])))
    geometry_hash = digest.hexdigest()
    key = stage_key(
        frames,
        [
            motion_and_grippers,
            segment_grippers,
            end_effector,
            segment_robots,
            robot_prompt,
            robot_entry_side,
            supported_robot_prompt,
            prompt_from_robot_region,
            supported_robot_prompt,
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
            return {
                "seeds": json.loads(str(d["seeds"])),
                "robot_masks": d["robot_masks"],
                "centers": d["centers"],
                "apertures": d["apertures"],
                "masks": np.unpackbits(
                    d["masks"], axis=-1, count=frames.shape[2]
                ).astype(bool),
            }
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


def photometry_cache(frames, cache_dir):
    """Cache exact photometric evidence for immutable frames and implementations."""
    import cv2

    from ..background import clean_plate
    from . import discovery, video
    from . import photometric_motion as module
    from .discovery import motion_and_grippers

    dependencies = hashlib.sha256()
    for source in (module, discovery, clean_plate, video):
        dependencies.update(Path(source.__file__).read_bytes())
    key = stage_key(
        frames,
        [module.photometric_motion, motion_and_grippers],
        (
            "photometry-v1",
            dependencies.hexdigest(),
            np.__version__,
            cv2.__version__,
            version("scipy"),
        ),
    )
    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"photometry-{key}.npz"
    if path.exists():
        with np.load(path) as archive:
            values = {name: archive[name] for name in archive.files}
        raw = {
            name[5:]: value
            for name, value in values.items()
            if name.startswith("raw__")
        }
        evidence = {
            name[11:]: value
            for name, value in values.items()
            if name.startswith("discovery__")
        }
        evidence["prompt_support"] = values["support"]
        return module.PhotometricMotion(
            raw,
            evidence,
            values["support"],
            values["ambiguous"],
            values["coefficients"],
        )
    result = module.photometric_motion(frames)
    values = {f"raw__{name}": value for name, value in result.raw.items()}
    values.update(
        {
            f"discovery__{name}": value
            for name, value in result.discovery.items()
            if name != "prompt_support"
        }
    )
    values.update(
        support=result.support,
        ambiguous=result.ambiguous,
        coefficients=result.coefficients,
    )
    temporary = directory / f"photometry-{key}-{uuid.uuid4().hex}.tmp.npz"
    np.savez_compressed(temporary, **values)
    temporary.replace(path)
    return result
