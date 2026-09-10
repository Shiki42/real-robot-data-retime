import argparse
from pathlib import Path
from real_robot_data_retime.interaction.pipeline import run
from real_robot_data_retime.edit import edit_video


def main():
    p = argparse.ArgumentParser(description="Automatic dual-arm video editing")
    p.add_argument("video", nargs="?", type=Path)
    p.add_argument("--input", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--debug-dir", type=Path)
    p.add_argument("--task", choices=["drawer", "letters", "workpiece"])
    p.add_argument("--backend", choices=["sam2", "geometry"], default="sam2")
    p.add_argument("--analysis-width", type=int, default=640)
    p.add_argument("--analysis-only", action="store_true")
    p.add_argument("--joint-data", type=Path)
    p.add_argument("--urdf", type=Path)
    p.add_argument("--mesh-root", type=Path)
    p.add_argument("--right-delay-seconds", type=float, default=0)
    p.add_argument("--left-delay-seconds", type=float, default=0)
    a = p.parse_args()
    if bool(a.video) == bool(a.input):
        p.error("provide exactly one input video")
    source = a.video or a.input
    debug = a.debug_dir or source.parent / (source.stem + "_debug")
    options = dict(backend=a.backend, analysis_width=a.analysis_width)
    if a.analysis_only:
        result = run(source, debug, a.task, **options)
    else:
        output = a.output or source.parent / (source.stem + "_parallel.mp4")
        if output.resolve() == source.resolve():
            p.error("output must differ from input")
        result = edit_video(
            source,
            output,
            debug,
            a.task,
            **options,
            joint_data=a.joint_data,
            urdf=a.urdf,
            mesh_root=a.mesh_root,
            right_delay_seconds=a.right_delay_seconds,
            left_delay_seconds=a.left_delay_seconds,
        )
    print(result)
    if not result["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
