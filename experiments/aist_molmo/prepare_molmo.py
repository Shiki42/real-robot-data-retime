"""Fetch the pinned 16-episode numeric shard and two source clips for the pilot."""

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from _common import run_stage, sha256, work_dir
from huggingface_hub import HfApi, hf_hub_download

REPO = "allenai/28112025-block-02"
REVISION = "957f65a1fa5c86b129b2f7c3e52a3b46ea976215"
VIDEO = "videos/observation.images.top/chunk-000/file-000.mp4"


def main(args):
    """Prepare episodes 15 and 8; the shared video is about 321 MB."""
    root = work_dir()
    data = root / "data/molmo_block_pilot.download"
    previews = root / "data/molmo_preview"
    previews.mkdir(parents=True, exist_ok=True)
    info = HfApi().dataset_info(REPO, revision=REVISION, files_metadata=True)
    inventory = {item.rfilename: item for item in info.siblings}
    needed = sorted(
        name
        for name in inventory
        if name.startswith(("data/", "meta/episodes/"))
        and name.endswith(".parquet")
        or name
        in (
            "meta/info.json",
            "meta/tasks.parquet",
            "meta/tasks_annotated.parquet",
            VIDEO,
        )
    )
    verified = []
    for name in needed:
        path = Path(
            hf_hub_download(
                REPO,
                name,
                repo_type="dataset",
                revision=REVISION,
                local_dir=data,
                etag_timeout=15,
            )
        )
        item = inventory[name]
        digest = sha256(path)
        expected = getattr(item.lfs, "sha256", None) if item.lfs else None
        if path.stat().st_size != item.size or expected and digest != expected:
            raise RuntimeError("Hub size/digest mismatch: " + name)
        verified.append({"path": name, "bytes": path.stat().st_size, "sha256": digest})
        print("Verified", name, flush=True)
    metadata = json.loads((data / "meta/info.json").read_text())
    if metadata["fps"] != 30:
        raise ValueError("This profile expects 30 Hz source data")
    episodes = pd.concat(
        [
            pq.read_table(p).to_pandas()
            for p in (data / "meta/episodes").rglob("*.parquet")
        ]
    )
    previous = previews / "manifest.json"
    manifest = json.loads(previous.read_text()) if previous.exists() else None
    if manifest and (
        manifest["source_repo"] != REPO or manifest["source_revision"] != REVISION
    ):
        raise ValueError("Existing manifest belongs to a different source")
    clips = []
    for ep in (15, 8):
        matches = episodes[episodes.episode_index == ep]
        if len(matches) != 1:
            raise ValueError("Expected one metadata row per episode")
        row = matches.iloc[0]
        key = "videos/observation.images.top"
        if row[key + "/chunk_index"] != 0 or row[key + "/file_index"] != 0:
            raise ValueError("Shared-video layout changed")
        start = float(row[key + "/from_timestamp"])
        frames = int(row["length"])
        folder = previews / f"ep{ep}"
        folder.mkdir(exist_ok=True)
        target = folder / "source.mp4"
        if target.exists():
            old = next(
                (x for x in (manifest or {}).get("episodes", []) if x["episode"] == ep),
                None,
            )
            if old is None or sha256(target) != old["sha256"]:
                raise ValueError(
                    "Existing clip lacks matching provenance; use a fresh work directory"
                )
        else:
            temporary = folder / "source.partial.mp4"
            subprocess.run(
                [
                    "ffmpeg",
                    "-loglevel",
                    "error",
                    "-y",
                    "-c:v",
                    "libdav1d",
                    "-ss",
                    str(start),
                    "-i",
                    str(data / VIDEO),
                    "-frames:v",
                    str(frames),
                    "-an",
                    "-c:v",
                    "libx264",
                    "-threads",
                    "2",
                    "-preset",
                    "fast",
                    "-crf",
                    "16",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(temporary),
                ],
                check=True,
            )
            cap = cv2.VideoCapture(str(temporary))
            count = 0
            try:
                if (
                    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                ) != (640, 360):
                    raise ValueError("Unexpected source clip dimensions")
                if not np.isclose(cap.get(cv2.CAP_PROP_FPS), 30):
                    raise ValueError("Unexpected video FPS")
                while cap.grab():
                    count += 1
            finally:
                cap.release()
            if count != frames:
                raise ValueError("Decoded clip frame count mismatch")
            temporary.replace(target)
        clips.append(
            {
                "episode": ep,
                "frames": frames,
                "sha256": sha256(target),
                "source_start_seconds": start,
                "fps": 30,
            }
        )
        previous.write_text(
            json.dumps(
                {
                    "source_repo": REPO,
                    "source_revision": REVISION,
                    "source_files": verified,
                    "episodes": clips
                    + [
                        x
                        for x in (manifest or {}).get("episodes", [])
                        if x["episode"] not in [c["episode"] for c in clips]
                    ],
                },
                indent=2,
            )
        )
    print("Prepared two verified source clips", flush=True)


if __name__ == "__main__":
    run_stage(main)
