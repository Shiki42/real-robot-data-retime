import argparse
from pathlib import Path
from real_robot_data_retime.interaction.pipeline import run


def main():
    p = argparse.ArgumentParser(
        description="Video-only automatic interaction understanding"
    )
    p.add_argument("video", nargs="?", type=Path)
    p.add_argument("--input", type=Path)
    p.add_argument("--debug-dir", type=Path)
    p.add_argument("--task", choices=["drawer", "letters", "workpiece"])
    p.add_argument("--backend", choices=["sam2", "geometry"], default="sam2")
    p.add_argument("--analysis-width", type=int, default=640)
    a = p.parse_args()
    if bool(a.video) == bool(a.input):
        p.error("provide exactly one input video")
    source = a.video or a.input
    result = run(
        source, a.debug_dir or source.parent / (source.stem + "_debug"), a.task
    )
    print(result)


if __name__ == "__main__":
    main()
