"""Build a local preview for the verified screw pilot artifacts."""

import argparse
import json
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
        case["stages"] = report["plan"]["stages"]
        case["fps"] = report["plan"]["fps"]
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
