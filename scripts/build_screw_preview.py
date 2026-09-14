"""Build a local preview for the verified screw pilot artifacts."""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np


def build(root):
    cases = json.loads((root / "cases.json").read_text())
    for case in cases:
        folder = root / case["name"]
        report = json.loads((folder / "report.json").read_text())
        if (
            not report["validation"]["passed"]
            or not report["plan"]["validation"]["passed"]
        ):
            raise ValueError("preview requires validated video and motion mappings")
        with np.load(folder / "source_mapping.npz") as data:
            case["left"] = data["left"].tolist()
            case["right"] = data["right"].tolist()
        case["media_version"] = str((folder / "preview.webm").stat().st_mtime_ns)
        case["stages"] = report["plan"]["stages"]
        case["fps"] = report["plan"]["fps"]
        case["review_locations"] = []
        for stage in case["stages"]:
            if stage["kind"] != "independent":
                continue
            repair = report["render"][stage["cycle"] - 1]["automatic_repair"]
            a, b = stage["output_start"], stage["output_end"] + 1
            clocks = [np.asarray(case[key][a:b]) for key in ("left", "right")]
            different = (clocks[0] != clocks[1]) | (clocks[0] != np.floor(clocks[0]))
            for side in range(2):
                sources = [
                    stage["source_start"] + item["source_frame"]
                    for item in repair["unresolved"]
                    if item["side"] == side
                ]
                flagged = different & (
                    np.isin(np.floor(clocks[side]), sources)
                    | np.isin(np.ceil(clocks[side]), sources)
                )
                starts = np.flatnonzero(np.diff(np.r_[False, flagged].astype(int)) == 1)
                for offset in starts:
                    case["review_locations"].append(
                        {
                            "frame": a + int(offset),
                            "cycle": stage["cycle"],
                            "arm": ["左臂", "右臂"][side],
                        }
                    )
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
                    str(folder / "preview.webm"),
                ]
            )
        )
        case["video_pts"] = [
            float(frame["best_effort_timestamp_time"]) for frame in probe["frames"]
        ]
        if len(case["video_pts"]) != case["frames"] or not np.allclose(
            case["video_pts"],
            np.arange(case["frames"]) / case["fps"],
            atol=0.00051,
            rtol=0,
        ):
            raise ValueError("browser video timestamps do not match the source mapping")
    template = (
        Path(__file__).parents[1] / "src/real_robot_data_retime/preview/screw.html"
    )
    original = (
        '<details><summary>查看原始示范</summary><video src="original.webm" controls preload="metadata"></video></details>'
        if (root / "original.webm").is_file()
        else ""
    )
    (root / "index.html").write_text(
        template.read_text()
        .replace("__ORIGINAL__", original)
        .replace("__CASES__", json.dumps(cases).replace("<", "\\u003c"))
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    build(parser.parse_args().root)
