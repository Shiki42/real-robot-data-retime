"""Render a verified interaction checkpoint through the shared renderer."""

import argparse
import json
from pathlib import Path

from real_robot_data_retime.interaction.checkpoint_render import render_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "analysis", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fixed-workspace", action="store_true")
    mode.add_argument("--diagnostic-nominal-workspace", action="store_true")
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--joint-data", type=Path)
    parser.add_argument("--urdf", type=Path)
    parser.add_argument("--mesh-root", type=Path)
    parser.add_argument("--right-delay-seconds", type=float, default=0)
    parser.add_argument("--left-delay-seconds", type=float, default=0)
    args = parser.parse_args()
    result = render_checkpoint(
        args.input,
        args.analysis,
        args.output,
        fixed_workspace=args.fixed_workspace,
        workspace=json.loads(args.workspace.read_text()) if args.workspace else None,
        diagnostic_nominal_workspace=args.diagnostic_nominal_workspace,
        joint_data=args.joint_data,
        urdf=args.urdf,
        mesh_root=args.mesh_root,
        right_delay_seconds=args.right_delay_seconds,
        left_delay_seconds=args.left_delay_seconds,
    )
    print(json.dumps(result, indent=2))
    if not (
        result["source_origin_checks_passed"]
        if args.diagnostic_nominal_workspace
        else result["automatic_checks_passed"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
