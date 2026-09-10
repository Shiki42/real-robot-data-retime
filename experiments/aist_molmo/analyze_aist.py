"""Read one AIST episode and save a conservative timing-only proposal."""

import argparse
import json
from pathlib import Path

import numpy as np

from real_robot_data_retime.aist import AistEpisode
from real_robot_data_retime.aist_timing import build_proposal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--fps",
        type=float,
        help="Explicit nominal profile only if file metadata is missing",
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be a new directory")
    with AistEpisode(args.source, fps=args.fps) as episode:
        left, right, common, report = build_proposal(
            episode.state, episode.action, episode.fps
        )
        report["source"] = episode.receipt()
    args.output.mkdir(parents=True)
    np.savez_compressed(
        args.output / "source_maps.npz", left=left, right=right, common=common
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps({"frames": report["output_frames"], "decision": report["decision"]})
    )


if __name__ == "__main__":
    main()
