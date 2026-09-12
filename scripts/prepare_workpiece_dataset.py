"""Extract all episodes and build a resumable manifest from pinned LeRobot data."""

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import pyarrow.parquet as pq

from real_robot_data_retime.automatic_dataset import check_output_location
from real_robot_data_retime.trim import read_episode, source_episodes


def prepare(source, work, revision, urdf, mesh_root, reuse_manifest=None):
    check_output_location(source, source, work)
    info = json.loads((source / "meta/info.json").read_text())
    rows = source_episodes(source)
    if len(rows) != info["total_episodes"]:
        raise ValueError("source episode count mismatch")
    reused = (
        {}
        if reuse_manifest is None
        else {
            r["episode"]: r for r in json.loads(reuse_manifest.read_text())["episodes"]
        }
    )
    records = []
    for row in rows:
        ep = row["episode_index"]
        root = work / f"episodes/{ep:03d}"
        root.mkdir(parents=True, exist_ok=True)
        table = read_episode(source, info, row)
        pq.write_table(table, root / "source.parquet")
        old = reused.get(ep)
        video = Path(old["source_video"]) if old else root / "source.mp4"
        if not video.exists():
            key = "observation.images.right_environment_1"
            prefix = "videos/" + key
            src = source / info["video_path"].format(
                video_key=key,
                chunk_index=row[prefix + "/chunk_index"],
                file_index=row[prefix + "/file_index"],
            )
            subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-v",
                    "error",
                    "-ss",
                    str(row[prefix + "/from_timestamp"]),
                    "-i",
                    str(src),
                    "-frames:v",
                    str(len(table)),
                    "-an",
                    "-c:v",
                    "libx264",
                    "-threads",
                    "2",
                    "-preset",
                    "fast",
                    "-crf",
                    "18",
                    "-pix_fmt",
                    "yuv420p",
                    str(video),
                ],
                check=True,
            )
        capture = cv2.VideoCapture(str(video))
        count = round(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        capture.release()
        if count != len(table) or abs(fps - info["fps"]) > 1e-4:
            raise ValueError(f"video and joint frame counts/rates differ: {ep}")
        record = {
            "episode": ep,
            "source_video": str(video),
            "joint_data": str(root / "source.parquet"),
            "analysis": old["analysis"] if old else str(root / "analysis"),
            "work_dir": str(root),
            "length": len(table),
        }
        if old and "verified_render" in old:
            record["verified_render"] = old["verified_render"]
        records.append(record)
    manifest = {
        "source": str(source),
        "source_revision": revision,
        "urdf": str(urdf),
        "mesh_root": str(mesh_root),
        "episodes": records,
    }
    (work / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ["source", "work", "urdf", "mesh-root"]:
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--reuse-manifest", type=Path)
    a = p.parse_args()
    result = prepare(
        a.source.resolve(),
        a.work.resolve(),
        a.revision,
        a.urdf.resolve(),
        a.mesh_root.resolve(),
        a.reuse_manifest,
    )
    print(
        json.dumps(
            {
                "episodes": len(result["episodes"]),
                "manifest": str(a.work / "manifest.json"),
            }
        )
    )


if __name__ == "__main__":
    main()
