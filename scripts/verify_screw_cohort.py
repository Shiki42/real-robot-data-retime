"""Recheck rendered clocks, synchronized media and final-pose waits."""

import argparse
import json
from pathlib import Path

import numpy as np

from real_robot_data_retime.screw import inputs, verify_output, write_json
from real_robot_data_retime.timeline.screw import screw_schedule
from real_robot_data_retime.timeline.smooth import sample_rows


def verify(source, work, configs):
    receipts = []
    for path in configs:
        config = json.loads(path.read_text())
        info, _, _, _, state, action, trim, identity = inputs(source, config)
        episode = config["episode_index"]
        root = work / f"ep{episode:03d}"
        for case in json.loads((root / "cases.json").read_text()):
            folder = root / case["name"]
            left, right, plan = screw_schedule(
                state,
                action,
                trim["start"],
                trim["stop"],
                config["coupled_intervals"],
                config["ready_frames"],
                config["right_retreat_ends"],
                case["position"],
                info["fps"],
                brake_seconds=config["brake_seconds"],
                restart_seconds=config["restart_seconds"],
            )
            with np.load(folder / "source_mapping.npz") as data:
                np.testing.assert_array_equal(left, data["left"])
                np.testing.assert_array_equal(right, data["right"])
            media = verify_output(folder, state, action, left, right, info["fps"])
            motion = [
                np.c_[sample_rows(v[:, :7], left), sample_rows(v[:, 7:], right)]
                for v in (state, action)
            ]
            # This is a near-still diagnostic, not proof of physical contact.
            change = np.maximum(
                *[np.max(np.abs(np.diff(v, axis=0)), axis=1) for v in motion]
            )
            rounds = []
            for stage in plan["stages"]:
                if stage["kind"] != "independent":
                    continue
                a, b = max(stage["pickup_start_output_frames"]), stage["output_end"]
                edges = np.diff(np.r_[False, change[a:b] < 0.01, False].astype(int))
                lengths = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
                rounds.append(
                    {
                        "cycle": stage["cycle"],
                        "final_left_source": stage["final_ready_source_frames"][0],
                        "insertion_source": stage["source_end"],
                        "insertion_output": b,
                        "waiter": [
                            tr["arm"]
                            for tr in stage["transitions"]
                            if tr["hold_frames"]
                        ],
                        "longest_both_near_still_frames": int(lengths.max())
                        if len(lengths)
                        else 0,
                    }
                )
            report = json.loads((folder / "report.json").read_text())
            if report["inputs"] != identity:
                raise ValueError(
                    "rendered input identity differs from current source/config"
                )
            report["plan"] = plan
            write_json(folder / "report.json", report)
            receipts.append(
                {
                    "episode": episode,
                    "case": case["name"],
                    "frames": len(left),
                    "media": media,
                    "timeline": plan["validation"],
                    "rounds": rounds,
                    "wrist_pixels": report["validation"]["wrist_pixels"],
                    "protected_main_pixels": report["validation"][
                        "protected_main_pixels"
                    ],
                }
            )
        print("verified episode", episode, flush=True)
    return receipts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--configs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_json(args.output, verify(args.source, args.work, args.configs))
