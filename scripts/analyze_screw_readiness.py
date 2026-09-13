"""Derive sustained readiness from measured and commanded PiperX TCP poses."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from real_robot_data_retime.model_experiment import sha256
from real_robot_data_retime.screw import inputs, write_json
from real_robot_data_retime.timeline.readiness import sustained_ready_frame


def analyze(source, config, urdf, mesh_root):
    from robo_visualize.arms.piperx.model import PiperXModel

    _info, _row, _path, _first, state, action, trim, identity = inputs(source, config)
    with TemporaryDirectory() as tmp:
        resolved = Path(tmp) / "robot.urdf"
        resolved.write_text(
            urdf.read_text().replace("package://", str(mesh_root.resolve()) + "/")
        )
        model = PiperXModel(resolved)
        poses = []
        for values in (state, action):
            stream = np.empty((2, len(values), 4, 4))
            for side in range(2):
                for t, value in enumerate(values[:, side * 7 : side * 7 + 7]):
                    value = value.copy()
                    value[6] *= 0.5
                    model.update(value)
                    stream[side, t] = model.tcp_transform().homogeneous
            poses.append(stream)
    physical, evidence = [], []
    for cycle, refs in enumerate(config["ready_frames"]):
        starts = (
            [trim["start"], trim["start"]]
            if cycle == 0
            else [
                config["coupled_intervals"][cycle - 1][1],
                config["right_retreat_ends"][cycle - 1],
            ]
        )
        frames, details = [], []
        for side, ref in enumerate(refs):
            sl = slice(side * 7, side * 7 + 7)
            ready, receipt = sustained_ready_frame(
                state[:, sl],
                action[:, sl],
                poses[0][side],
                poses[1][side],
                starts[side],
                ref,
            )
            frames.append(ready)
            details.append(receipt)
        physical.append(frames)
        evidence.append(details)
    config.update(
        preparation_frames=physical,
        readiness_evidence=evidence,
        readiness_source_data_sha256=identity["data_sha256"],
        readiness_urdf_sha256=sha256(urdf),
    )
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--mesh-root", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    write_json(args.config, analyze(args.source, config, args.urdf, args.mesh_root))
