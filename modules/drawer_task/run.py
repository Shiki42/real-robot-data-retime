"""Run the accepted drawer snapshots without importing the repository's live route."""

import argparse
import hashlib
import json
import runpy
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def verify():
    manifest = json.loads((ROOT / "freeze-manifest.json").read_text())
    for relative, expected in manifest.items():
        path = ROOT / relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"frozen drawer file changed: {relative}")
    for stage in ("analysis", "render"):
        expected = json.loads((ROOT / stage / "source-manifest.json").read_text())
        actual = {
            str(p.relative_to(ROOT / stage))
            for p in (ROOT / stage / "src").rglob("*.py")
        }
        if actual != set(expected):
            raise ValueError(f"frozen {stage} source inventory changed")


def main():
    # Ignore PYTHONPATH, user site packages and the caller's working directory.
    if not sys.flags.isolated:
        result = subprocess.run(
            [sys.executable, "-I", str(Path(__file__).resolve()), *sys.argv[1:]],
            check=False,
        )
        raise SystemExit(result.returncode)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("verify")
    analyze = commands.add_parser("analyze")
    analyze.add_argument("--input", type=Path, required=True)
    analyze.add_argument("--output", type=Path, required=True)
    analyze.add_argument("--reuse-measurements", type=Path)
    render = commands.add_parser("render")
    render.add_argument("--input", type=Path, required=True)
    render.add_argument("--analysis", type=Path, required=True)
    render.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify()
    if args.command == "verify":
        print("拉抽屉任务：固定版本文件校验通过")
        return
    stage = "analysis" if args.command == "analyze" else "render"
    sys.path.insert(0, str(ROOT / stage / "src"))
    if args.command == "analyze":
        from real_robot_data_retime.interaction.pipeline import run

        if args.output.exists():
            raise FileExistsError(args.output)
        report = run(
            args.input,
            args.output,
            task="drawer",
            reuse_measurements=args.reuse_measurements,
        )
        passed = report["success"]
    else:
        timeline = json.loads((args.analysis / "interaction_timeline.json").read_text())
        if timeline["task"] != "drawer":
            raise ValueError("this frozen module only accepts drawer checkpoints")
        render_checkpoint = runpy.run_path(str(ROOT / "render/checkpoint.py"))[
            "render_checkpoint"
        ]
        report = render_checkpoint(args.input, args.analysis, args.output)
        passed = report["automatic_checks_passed"]
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
