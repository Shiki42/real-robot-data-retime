"""Reusable automatic hypotheses, always reverified by current causal inference."""

import hashlib
import json
from pathlib import Path
import numpy as np
from .neural_tracks import grippers_from_robots


def inputs_fingerprint(video, frames, proposals, transforms):
    with Path(video).open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return dict(
        video_sha256=digest,
        shape=list(frames.shape),
        registration_sha256=hashlib.sha256(
            np.ascontiguousarray(transforms).tobytes()
        ).hexdigest(),
        proposals=[
            dict(
                bbox=np.asarray(p["bbox"]).tolist(),
                origin=np.asarray(p["origin"]).tolist(),
            )
            for p in proposals
        ],
    )


def producer_fingerprint():
    root = Path(__file__).parent.parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_hypotheses(directory, expected_inputs, proposals):
    directory = Path(directory)
    manifest = json.loads((directory / "measurements.json").read_text())
    if manifest["inputs"] != expected_inputs:
        raise ValueError(
            "cached automatic hypotheses do not match video, registration or proposals"
        )
    with (
        np.load(directory / "segmentation.npz") as masks,
        np.load(directory / "tracks.npz") as paths,
    ):
        h, w = map(int, masks["frame_shape"])
        robots = masks["robots"]
        objects = masks["objects"]
        centers = paths["objects"]
    if len(objects) != len(proposals) or centers.shape != (
        len(proposals),
        len(robots),
        2,
    ):
        raise ValueError("cached hypothesis dimensions differ")
    grippers = grippers_from_robots(robots, (h, w))
    tracks = []
    for k, proposal in enumerate(proposals):
        area = np.unpackbits(objects[k], axis=-1, count=w).sum(axis=(1, 2))
        tracks.append(
            dict(
                proposal=proposal,
                centers=centers[k],
                areas=area,
                packed_masks=objects[k],
                confidence=np.minimum(area / max(1, proposal["area"]), 1.0),
            )
        )
    points = None
    if (directory / "drawer_point_tracks.npz").exists():
        with np.load(directory / "drawer_point_tracks.npz") as archive:
            points = {key: archive[key] for key in archive.files}
    return grippers, tracks, points, manifest["producer"]
