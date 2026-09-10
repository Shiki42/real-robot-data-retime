"""Build an offline video/action viewer with verified per-frame presentation times."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import pyarrow.parquet as pq
import numpy as np


def build(root):
    root = Path(root)
    datasets = []
    for name, title in [("final-wait", "空中等待"), ("early", "抽屉已提前打开")]:
        folder = root / name
        table = pq.read_table(folder / "trajectories.parquet")
        timestamps = np.array(table["timestamp"].to_pylist(), float)
        action = np.array(table["action"].to_pylist(), float)
        if (
            not len(timestamps)
            or not np.isfinite(timestamps).all()
            or timestamps[0] != 0
            or np.any(np.diff(timestamps) <= 0)
        ):
            raise ValueError("action timestamps must start at zero and increase")
        with np.load(folder / "source_mapping.npz") as mapping:
            for side in ["left", "right"]:
                if not np.array_equal(
                    mapping[side], table[f"{side}_source_frame"].to_pylist()
                ):
                    raise ValueError(
                        f"{side}: action and rendered video source clocks differ"
                    )
        if action.shape != (len(timestamps), 14) or not np.isfinite(action).all():
            raise ValueError("expected finite 14-value action per frame")
        for video in ["parallel.mp4", "preview.webm"]:
            probe = json.loads(
                subprocess.check_output(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-select_streams",
                        "v:0",
                        "-show_frames",
                        "-show_entries",
                        "frame=best_effort_timestamp_time",
                        "-of",
                        "json",
                        str(folder / video),
                    ]
                )
            )
            pts = np.array(
                [float(f["best_effort_timestamp_time"]) for f in probe["frames"]]
            )
            if len(pts) != len(timestamps) or not np.allclose(
                pts, timestamps, atol=0.00051, rtol=0
            ):
                raise ValueError(f"{video}: video PTS and action timestamps differ")
            if video == "preview.webm":
                web_pts = pts.tolist()
        report = json.loads((folder / "report.json").read_text())
        datasets.append(
            dict(
                name=name,
                title=title,
                timestamps=timestamps.tolist(),
                video_pts=web_pts,
                action=action.tolist(),
                left=table["left_source_frame"].to_pylist(),
                right=table["right_source_frame"].to_pylist(),
                stages=report["plan"]["stages"],
            )
        )
    template = (
        Path(__file__).resolve().parents[1]
        / "src/real_robot_data_retime/preview/action.html"
    )
    html = template.read_text().replace(
        "__ACTION_DATA__", json.dumps(datasets, allow_nan=False)
    )
    (root / "index.html").write_text(html)
    (root / "serve.py").write_text(
        Path(__file__).with_name("serve_action_preview.py").read_text()
    )
    paths = ["index.html", "serve.py"] + [
        f"{d['name']}/{f}"
        for d in datasets
        for f in [
            "parallel.mp4",
            "preview.webm",
            "trajectories.parquet",
            "source_mapping.npz",
            "report.json",
        ]
    ]
    manifest = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in paths
    }
    (root / "download_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({d["name"]: len(d["timestamps"]) for d in datasets}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    build(parser.parse_args().root)
