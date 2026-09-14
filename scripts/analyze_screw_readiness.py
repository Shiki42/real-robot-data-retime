"""Locate the last left adjustment before each reviewed insertion boundary."""

import argparse
import json
from pathlib import Path

from real_robot_data_retime.screw import inputs, write_json
from real_robot_data_retime.timeline.readiness import final_left_pose


def analyze(source, config):
    values = inputs(source, config)
    state, action, trim, identity = values[4:]
    evidence = []
    start = trim["start"]
    for cycle, (begin, end) in enumerate(config["coupled_intervals"]):
        final, receipt = final_left_pose(state[:, :7], action[:, :7], start, begin)
        config["ready_frames"][cycle][0] = final
        evidence.append(receipt)
        start = end
    config["final_left_pose_evidence"] = evidence
    config["readiness_source_data_sha256"] = identity["data_sha256"]
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    write_json(args.config, analyze(args.source, json.loads(args.config.read_text())))
