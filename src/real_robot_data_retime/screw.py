"""Reproducible five-insertion pilot with reviewed source-frame boundaries."""

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

from .automatic_validation import wrist_pixel_error
from .compositing.screw import ScrewStageCompositor
from .interaction.video import read_video, write_video
from .model_experiment import sha256
from .staged import export_trajectories
from .timeline.screw import screw_schedule
from .timeline.smooth import sample_rows
from .trim import analyze_episode, read_episode, source_episodes
from .validate import frame_at, mae
from .video import remap_video


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def inputs(source, config):
    info = json.loads((source / "meta/info.json").read_text())
    rows = source_episodes(source)
    row = next(r for r in rows if r["episode_index"] == config["episode_index"])
    table = read_episode(source, info, row)
    key = config["camera"]
    prefix = "videos/" + key
    path = source / info["video_path"].format(
        video_key=key,
        chunk_index=row[prefix + "/chunk_index"],
        file_index=row[prefix + "/file_index"],
    )
    first = round(row[prefix + "/from_timestamp"] * info["fps"])
    state = np.array(table["observation.state"].to_pylist())
    action = np.array(table["action"].to_pylist())
    trim = analyze_episode(action, info["fps"])
    data_path = source / info["data_path"].format(
        chunk_index=row["data/chunk_index"], file_index=row["data/file_index"]
    )
    identity = {
        "data_sha256": sha256(data_path),
        "video_sha256": sha256(path),
        "episode_index": row["episode_index"],
        "first_video_frame": first,
        "source_frames": len(table),
        "config": config,
    }
    return info, row, path, first, state, action, trim, identity


def mask_proposal(spec, shape):
    return {
        key: spec[key] for key in ("bbox", "positive_points", "negative_points")
    } | {"mask": np.ones(shape, bool)}


def segment_carried_screw(model, frames, robots, cursor, begin, spec):
    """Track the visible bolt head and shaft only after its recorded pickup."""
    if not cursor <= spec["pickup_frame"] <= begin or not spec["pickup_frame"] <= spec[
        "source_frame"
    ] < len(frames):
        raise ValueError("held screw prompt is outside its pickup/approach interval")
    proposal = mask_proposal(spec, frames.shape[1:3])
    for reverse, stop in [(False, begin + 1), (True, spec["pickup_frame"] - 1)]:
        for t, mask in model.propagate(
            frames,
            [proposal],
            seed_frame=spec["source_frame"],
            reverse=reverse,
            stop_frame=stop,
        ):
            if t <= begin:
                robots[t - cursor, 1] |= mask[0]


def prepare(frames, config, trim, work, identity):
    from .interaction.photometric_motion import photometric_motion
    from .interaction.robot_discovery import robot_prompt
    from .segmentation.sam_backend import SamVideo

    model = SamVideo()
    (work / "analysis.json").unlink(missing_ok=True)
    cursor = trim["start"]
    for cycle, (begin, end) in enumerate(config["coupled_intervals"]):
        block = frames[cursor : end + 1]
        evidence = (
            photometric_motion(block)
            if config["left_segmentation_prompts"][cycle] is None
            else None
        )
        robots = np.zeros((len(block), 2, *frames.shape[1:3]), bool)
        prompts = []
        for side in range(2):
            if side == 0 and config["left_segmentation_prompts"][cycle] is None:
                seed, proposal = robot_prompt(block, evidence.discovery, side)
            else:
                spec = config[
                    ["left_segmentation_prompts", "right_segmentation_prompts"][side]
                ][cycle]
                seed = spec["source_frame"] - cursor
                proposal = mask_proposal(spec, frames.shape[1:3])
            prompts.append(
                {
                    "side": side,
                    "frame": seed + cursor,
                    "bbox": proposal["bbox"],
                    "positive_points": proposal["positive_points"],
                }
            )
            for reverse in (False, True):
                for t, mask in model.propagate(
                    block, [proposal], seed_frame=seed, reverse=reverse
                ):
                    robots[t, side] = mask[0]
        for spec in config["segmentation_overrides"]:
            if spec["cycle"] != cycle:
                continue
            proposal = mask_proposal(spec, frames.shape[1:3])
            for t, mask in model.propagate(
                frames,
                [proposal],
                seed_frame=spec["source_frame"],
                stop_frame=spec["stop"],
            ):
                robots[t - cursor, spec["side"]] = mask[0]
        for spec in config["right_payload_prompts"][cycle]:
            segment_carried_screw(model, frames, robots, cursor, begin, spec)
        valid = begin - cursor + 1
        np.savez_compressed(
            work / f"masks_{cycle}.npz",
            robots=np.packbits(robots[:valid], axis=-1),
            source_start=cursor,
            source_end=begin,
        )
        write_json(work / f"prompts_{cycle}.json", prompts)
        print("segmentation complete", cycle + 1, flush=True)
        cursor = end
    write_json(work / "analysis.json", {"inputs": identity, "complete": True})


def verify_output(folder, state, action, left, right, fps):
    table = pq.read_table(folder / "trajectories.parquet")
    for name, values in [("action", action), ("observation.state", state)]:
        expected = np.column_stack(
            [sample_rows(values[:, :7], left), sample_rows(values[:, 7:], right)]
        )
        if not np.array_equal(np.array(table[name].to_pylist()), expected):
            raise ValueError("exported motion does not match video source clocks")
    counts = {}
    for name in ["parallel.mp4", "left_wrist.mp4", "right_wrist.mp4", "preview.webm"]:
        probe = json.loads(
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_frames",
                    "-show_entries",
                    "frame=best_effort_timestamp_time",
                    "-of",
                    "json",
                    str(folder / name),
                ]
            )
        )
        pts = np.array(
            [float(f["best_effort_timestamp_time"]) for f in probe["frames"]]
        )
        if len(pts) != len(left) or not np.allclose(
            pts, np.arange(len(left)) / fps, atol=0.00051, rtol=0
        ):
            raise ValueError("video PTS differs from exported trajectories")
        counts[name] = len(pts)
    return {
        "passed": True,
        "frames": counts,
        "action_state_exact": True,
        "camera_timestamps_match": True,
    }


def verify_protected_pixels(folder, frames, plan):
    capture = cv2.VideoCapture(str(folder / "parallel.mp4"))
    errors = []
    try:
        for stage in plan["stages"]:
            if stage["kind"] not in ("coupled", "final_storage"):
                continue
            for target in np.linspace(
                stage["output_start"], stage["output_end"], 5
            ).astype(int):
                source = stage["source_start"] + target - stage["output_start"]
                errors.append(mae(frame_at(capture, int(target)), frames[source]))
    finally:
        capture.release()
    if max(errors) > 6:
        raise ValueError(
            "protected main-view pixels do not match the original recording"
        )
    return {
        "samples": len(errors),
        "maximum_mean_absolute_pixel_error": max(errors),
        "passed": True,
    }


def render_frames(frames, compositors, left, right, stages):
    stage_index = 0
    for t, (l, r) in enumerate(zip(left, right)):
        while t > stages[stage_index]["output_end"]:
            stage_index += 1
        stage = stages[stage_index]
        if stage["kind"] == "independent":
            yield compositors[stage["cycle"] - 1].frame(
                l - stage["source_start"], r - stage["source_start"]
            )
        else:
            if l != r or l != int(l):
                raise ValueError(
                    "coupled render requires identical native source frames"
                )
            yield frames[int(l)]


def render(source, work, frames, config, values):
    info, row, _path, _first, state, action, trim, identity = values
    analysis = json.loads((work / "analysis.json").read_text())
    if not analysis["complete"] or analysis["inputs"] != identity:
        raise ValueError("segmentation source/config identity differs; prepare again")
    (work / "cases.json").unlink(missing_ok=True)
    compositors = []
    cursor = trim["start"]
    for cycle, (begin, end) in enumerate(config["coupled_intervals"]):
        with np.load(work / f"masks_{cycle}.npz") as data:
            if int(data["source_start"]) != cursor or int(data["source_end"]) != begin:
                raise ValueError("segmentation boundaries differ")
            robots = np.unpackbits(
                data["robots"], axis=-1, count=frames.shape[2]
            ).astype(bool)
        compositors.append(
            ScrewStageCompositor(
                frames[cursor : begin + 1],
                robots,
                config["workspace_split_x"],
                config["left_scene_boxes"],
            )
        )
        print("compositor ready", cycle + 1, flush=True)
        cursor = end
    cases = []
    for position in config["positions"]:
        name = f"screw-{round(position * 100):02d}"
        folder = work / name
        folder.mkdir(exist_ok=True)
        (folder / "report.json").unlink(missing_ok=True)
        left, right, plan = screw_schedule(
            state,
            action,
            trim["start"],
            trim["stop"],
            config["coupled_intervals"],
            config["ready_frames"],
            config["right_retreat_ends"],
            position,
            info["fps"],
            preparation_frames=config["preparation_frames"],
            brake_seconds=config["brake_seconds"],
            restart_seconds=config["restart_seconds"],
        )
        for comp in compositors:
            comp.overlap_pixels = comp.paired_frames = 0
            comp.interpolated_frames = [0, 0]

        write_video(
            folder / "parallel.mp4",
            render_frames(frames, compositors, left, right, plan["stages"]),
            info["fps"],
        )
        np.savez_compressed(folder / "source_mapping.npz", left=left, right=right)
        export_trajectories(
            folder, (state, action), left, right, info["fps"], {"stages": {}}
        )
        wrist_pixels = {}
        for side, clock in [("left", left), ("right", right)]:
            key = f"observation.images.{side}_wrist"
            prefix = "videos/" + key
            video = source / info["video_path"].format(
                video_key=key,
                chunk_index=row[prefix + "/chunk_index"],
                file_index=row[prefix + "/file_index"],
            )
            offset = round(row[prefix + "/from_timestamp"] * info["fps"])
            feature = info["features"][key]
            remap_video(
                video,
                folder / f"{side}_wrist.mp4",
                clock + offset,
                info["fps"],
                feature["shape"][1],
                feature["shape"][0],
            )
            wrist_pixels[side] = wrist_pixel_error(
                video, folder / f"{side}_wrist.mp4", clock + offset
            )
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-i",
                str(folder / "parallel.mp4"),
                "-c:v",
                "libvpx-vp9",
                "-deadline",
                "realtime",
                "-cpu-used",
                "8",
                "-crf",
                "32",
                "-b:v",
                "0",
                str(folder / "preview.webm"),
            ],
            check=True,
        )
        validation = verify_output(folder, state, action, left, right, info["fps"])
        validation["wrist_pixels"] = wrist_pixels
        validation["protected_main_pixels"] = verify_protected_pixels(
            folder, frames, plan
        )
        report = {
            "plan": plan,
            "trim": trim,
            "inputs": identity,
            "render": [c.report() for c in compositors],
            "validation": validation,
        }
        write_json(folder / "report.json", report)
        cases.append(
            {
                "name": name,
                "position": position,
                "seconds": len(left) / info["fps"],
                "frames": len(left),
            }
        )
        print("rendered", name, len(left), flush=True)
    write_json(work / "cases.json", cases)
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Regenerate whole-arm segmentation before rendering",
    )
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text())
    values = inputs(args.source, config)
    info, _row, path, first, state, _action, trim, identity = values
    frames, fps = read_video(
        path,
        first,
        first + len(state),
        width=info["features"][config["camera"]]["shape"][1],
    )
    if fps != info["fps"]:
        raise ValueError("source video and telemetry FPS differ")
    if args.prepare or args.prepare_only:
        prepare(frames, config, trim, args.work, identity)
    if not args.prepare_only:
        render(args.source, args.work, frames, config, values)


if __name__ == "__main__":
    main()
