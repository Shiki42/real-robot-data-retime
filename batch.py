import argparse
import json
import traceback
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
        try:
            result = run(video, a.output_dir / video.stem)
        except (ValueError, RuntimeError, OSError) as error:
            # Batch isolation is intentional: retain a full failure record and
            # a failing batch exit status while processing the remaining videos.
            traceback.print_exc()
            result = dict(
                success=False, error_type=type(error).__name__, error=str(error)
            )
            destination = a.output_dir / video.stem
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "report.json").write_text(json.dumps(result, indent=2))
        reports.append({"input": str(video), **result})
        (a.output_dir / "batch_report.json").write_text(json.dumps(reports, indent=2))
    if any(not report["success"] for report in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
