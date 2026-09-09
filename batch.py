import argparse
import json
from pathlib import Path
from real_robot_data_retime.interaction.pipeline import run


def main():
    p = argparse.ArgumentParser(description="Unattended video interaction batch")
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    videos = sorted(a.input_dir.glob("*.mp4"))
    if not videos:
        p.error("input directory contains no MP4 videos")
    a.output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for video in videos:
        reports.append({"input": str(video), **run(video, a.output_dir / video.stem)})
        (a.output_dir / "batch_report.json").write_text(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
