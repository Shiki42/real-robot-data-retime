"""Unattended LeRobot processing; failed episodes never become a public dataset."""

import argparse
import hashlib
import json
import shutil
from tempfile import TemporaryDirectory
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from .interaction.pipeline import run
from .edit import native_render_inputs
from .timeline.planner import plan_joints
from .compositing.layers import composite
from .compositing.depth import AlignedDepth
from .trim import source_episodes, read_episode, image_statistics, aggregate_stats
from .stats import feature_statistics
from .video import remap_video


def analysis_identity(video, package_root):
    digest = hashlib.sha256()
    with Path(video).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    root = Path(package_root)
    for directory in ["interaction", "tracking", "segmentation", "tasks"]:
        for path in sorted((root / directory).glob("*.py")):
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(path.read_bytes())
    for name in ["timeline/episode.py", "timeline/hypotheses.py"]:
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def remap_table(table, left, right, episode):
    length = len(left)
    columns = {}
    for field in table.schema:
        key = field.name
        if key == "complementary_info.rgb_device_timestamp_ns.top":
            continue  # A composite has two physical source timestamps.
        if key in ["action", "observation.state"]:
            values = np.asarray(table[key].to_pylist())
            values = np.concatenate([values[left, :7], values[right, 7:]], axis=1)
            columns[key] = pa.array(values.tolist(), type=field.type)
        elif key in ["episode_index", "frame_index", "index", "timestamp"]:
            values = {
                "episode_index": np.full(length, episode),
                "frame_index": np.arange(length),
                "index": np.arange(length),
                "timestamp": np.arange(length),
            }[key]
            columns[key] = pa.array(values, type=field.type)
        elif key == "task_index":
            if len(set(table[key].to_pylist())) != 1:
                raise ValueError("task changes inside source episode")
            columns[key] = table[key].take(pa.array(left))
        elif key.startswith("complementary_info.left_") or key.endswith(".left_wrist"):
            columns[key] = table[key].take(pa.array(left))
        elif key.startswith("complementary_info.right_") or key.endswith(
            ".right_wrist"
        ):
            columns[key] = table[key].take(pa.array(right))
        else:
            raise ValueError(f"no explicit retime policy for field {key}")
    columns["retime.left_source_frame"] = pa.array(left, type=pa.int64())
    columns["retime.right_source_frame"] = pa.array(right, type=pa.int64())
    return pa.table(columns)


def check_output_location(source, raw_source, output):
    for original in [source.resolve(), raw_source.resolve()]:
        if output.resolve() == original or original in output.resolve().parents:
            raise ValueError("retimed output must be outside both source datasets")


def process_episode(source, raw_source, output, work_dir, urdf, mesh_root, index):
    source, raw_source, output, work_dir = map(
        Path, [source, raw_source, output, work_dir]
    )
    check_output_location(source, raw_source, output)
    info = json.loads((source / "meta/info.json").read_text())
    row = source_episodes(source)[index]
    ep = row["episode_index"]
    (output / f"meta/retime_receipts/episode_{ep:03d}.json").unlink(missing_ok=True)
    debug = work_dir / f"episode_{ep:03d}"
    debug.mkdir(parents=True, exist_ok=True)
    video = source / info["video_path"].format(
        video_key="observation.images.top", chunk_index=0, file_index=ep
    )
    identity = analysis_identity(video, Path(__file__).parent)
    identity_file = debug / "analysis_identity.txt"
    if not (
        identity_file.exists()
        and identity_file.read_text() == identity
        and (debug / "report.json").exists()
    ):
        reuse = debug if (debug / "measurements.json").exists() else None
        report = run(video, debug, "drawer", reuse_measurements=reuse)
        identity_file.write_text(identity)
    else:
        report = json.loads((debug / "report.json").read_text())
    if not report["success"]:
        raise ValueError(f"episode {ep}: automatic interaction verification failed")
    timeline = json.loads((debug / "interaction_timeline.json").read_text())
    table = read_episode(source, info, row)
    left, right, plan = plan_joints(
        np.asarray(table["observation.state"].to_pylist()),
        np.asarray(table["action"].to_pylist()),
        timeline,
        urdf,
        mesh_root,
    )
    np.savez_compressed(debug / "source_mapping.npz", left=left, right=right)
    (debug / "schedule.json").write_text(json.dumps(plan, indent=2))
    trim = json.loads((source / "trim_manifest.json").read_text())["episodes"][index]
    render = render_main(
        video, raw_source, output, debug, index, ep, timeline, left, right, trim
    )
    mapped = remap_table(table, left, right, ep)
    pos = mapped.schema.get_field_index("timestamp")
    mapped = mapped.set_column(
        pos,
        "timestamp",
        pa.array(
            np.arange(len(left)) / info["fps"],
            type=table.schema.field("timestamp").type,
        ),
    )
    synthetic = np.arange(len(left)) >= plan["synthetic_terminal_hold"]["start_frame"]
    mapped = mapped.append_column("retime.synthetic_hold", pa.array(synthetic))
    stats = {}
    for key in mapped.column_names:
        stats[key] = feature_statistics(
            np.asarray(mapped[key].to_pylist(), float).reshape(len(mapped), -1)
        )
    cameras = {k: v for k, v in info["features"].items() if v["dtype"] == "video"}
    for camera, feature in cameras.items():
        destination = output / f"videos/{camera}/chunk-000/file-{ep:03d}.mp4"
        if camera != "observation.images.top":
            side = {
                "observation.images.left_wrist": left,
                "observation.images.right_wrist": right,
            }[camera]
            original = source / info["video_path"].format(
                video_key=camera, chunk_index=0, file_index=ep
            )
            remap_video(
                original,
                destination,
                side,
                info["fps"],
                feature["shape"][1],
                feature["shape"][0],
            )
        stats[camera] = image_statistics(destination, len(left))
    data = output / f"data/chunk-000/file-{ep:03d}.parquet"
    data.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(mapped, data)
    mappings = output / "meta/retime_source_indices"
    mappings.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        mappings / f"episode_{ep:03d}.npz",
        left=left,
        right=right,
        raw_left=left + trim["start"],
        raw_right=right + trim["start"],
        synthetic_hold=synthetic,
    )
    receipt = dict(
        episode_index=ep,
        length=len(left),
        tasks=row["tasks"],
        plan=plan,
        compositing=render,
        statistics=stats,
        analysis_identity=identity,
        interaction=dict(
            timeline=timeline,
            report=report,
            measurements=json.loads((debug / "measurements.json").read_text()),
            robot_mask_audit=json.loads((debug / "robot_mask_audit.json").read_text()),
        ),
        trim=trim,
    )
    receipts = output / "meta/retime_receipts"
    receipts.mkdir(parents=True, exist_ok=True)
    (receipts / f"episode_{ep:03d}.json").write_text(json.dumps(receipt, indent=2))
    return receipt


def render_main(
    video, raw_source, output, debug, index, episode, timeline, left, right, trim
):
    with (
        np.load(debug / "tracks.npz") as tracks,
        np.load(debug / "segmentation.npz") as segmentation,
    ):
        native, masks, transforms = native_render_inputs(
            video, tracks["registration"], segmentation
        )
    raw_info = json.loads((raw_source / "meta/info.json").read_text())
    raw = read_episode(raw_source, raw_info, source_episodes(raw_source)[index])
    depth = AlignedDepth(
        raw_source,
        raw["observation.depth.top"].to_pylist()[trim["start"] : trim["stop"]],
        transforms,
        native.shape[1:3],
    )
    main = output / f"videos/observation.images.top/chunk-000/file-{episode:03d}.mp4"
    try:
        render = composite(
            native, timeline, masks, left, right, main, debug, depth=depth
        )
    finally:
        depth.close()
    if not render["automatic_origin_audit"]["passed"]:
        raise ValueError(
            f"episode {episode}: rendered object-origin verification failed"
        )
    render["output"] = main.relative_to(output).as_posix()
    from .interaction.measurements import producer_fingerprint

    render["implementation_fingerprint"] = producer_fingerprint()
    return render


def rerender_episode(source, raw_source, output, work_dir, index):
    """Rebuild the main view from an already audited source map, preserving telemetry."""
    source, raw_source, output, work_dir = map(
        Path, [source, raw_source, output, work_dir]
    )
    check_output_location(source, raw_source, output)
    row = source_episodes(source)[index]
    ep = row["episode_index"]
    debug = work_dir / f"episode_{ep:03d}"
    path = output / f"meta/retime_receipts/episode_{ep:03d}.json"
    receipt = json.loads(path.read_text())
    if receipt["analysis_identity"] != (debug / "analysis_identity.txt").read_text():
        raise ValueError("render hypotheses differ from the audited schedule")
    timeline = json.loads((debug / "interaction_timeline.json").read_text())
    with np.load(output / f"meta/retime_source_indices/episode_{ep:03d}.npz") as maps:
        left, right = maps["left"], maps["right"]
    video = source / f"videos/observation.images.top/chunk-000/file-{ep:03d}.mp4"
    # Render and measure off to the side; an interrupted publish must never
    # leave an old receipt certifying a newly replaced video.
    with TemporaryDirectory(prefix=".rerender-", dir=output) as staging:
        stage = Path(staging)
        render = render_main(
            video,
            raw_source,
            stage,
            debug,
            index,
            ep,
            timeline,
            left,
            right,
            receipt["trim"],
        )
        receipt["compositing"] = render
        receipt["statistics"]["observation.images.top"] = image_statistics(
            stage / render["output"], len(left)
        )
        temporary = path.with_suffix(".tmp.json")
        temporary.write_text(json.dumps(receipt, indent=2))
        path.unlink()
        (stage / render["output"]).replace(output / render["output"])
        temporary.replace(path)
    return receipt


def finalize(source, output, repo_id):
    source, output = Path(source), Path(output)
    info = json.loads((source / "meta/info.json").read_text())
    rows = source_episodes(source)
    receipts = [
        json.loads(
            (
                output / f"meta/retime_receipts/episode_{r['episode_index']:03d}.json"
            ).read_text()
        )
        for r in rows
    ]
    metadata = []
    total = 0
    all_stats = []
    for receipt in receipts:
        ep = receipt["episode_index"]
        n = receipt["length"]
        file = output / f"data/chunk-000/file-{ep:03d}.parquet"
        table = pq.read_table(file)
        index = table.schema.get_field_index("index")
        table = table.set_column(
            index,
            "index",
            pa.array(
                np.arange(total, total + n), type=table.schema.field("index").type
            ),
        )
        pq.write_table(table, file)
        stats = receipt["statistics"]
        stats["index"] = feature_statistics(np.arange(total, total + n).reshape(-1, 1))
        all_stats.append(stats)
        row = dict(
            episode_index=ep,
            length=n,
            tasks=receipt["tasks"],
            dataset_from_index=total,
            dataset_to_index=total + n,
        )
        row.update(
            {
                "data/chunk_index": 0,
                "data/file_index": ep,
                "meta/episodes/chunk_index": 0,
                "meta/episodes/file_index": 0,
            }
        )
        for camera, feature in info["features"].items():
            if feature["dtype"] == "video":
                row.update(
                    {
                        f"videos/{camera}/chunk_index": 0,
                        f"videos/{camera}/file_index": ep,
                        f"videos/{camera}/from_timestamp": 0.0,
                        f"videos/{camera}/to_timestamp": n / info["fps"],
                    }
                )
        for key, values in stats.items():
            row.update({f"stats/{key}/{name}": value for name, value in values.items()})
        metadata.append(row)
        total += n
    info["features"] = {
        k: v
        for k, v in info["features"].items()
        if k in table.column_names or v["dtype"] == "video"
    }
    for key, dtype in [
        ("retime.left_source_frame", "int64"),
        ("retime.right_source_frame", "int64"),
        ("retime.synthetic_hold", "bool"),
    ]:
        info["features"][key] = dict(dtype=dtype, shape=[1], names=None)
    info.update(
        repo_id=repo_id,
        total_frames=total,
        total_episodes=len(rows),
        splits={"train": f"0:{len(rows)}"},
    )
    (output / "meta/episodes/chunk-000").mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(metadata),
        output / "meta/episodes/chunk-000/file-000.parquet",
    )
    shutil.copy2(source / "meta/tasks.parquet", output / "meta/tasks.parquet")
    (output / "meta/info.json").write_text(json.dumps(info, indent=2))
    (output / "meta/stats.json").write_text(
        json.dumps(aggregate_stats(all_stats), indent=2)
    )
    return dict(episodes=len(rows), frames=total)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["source", "raw-source", "output", "work-dir", "urdf", "mesh-root"]:
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--episodes", type=int, nargs="+")
    p.add_argument("--repo-id", default="Shiki42/piperx-put-cube-in-drawer-retime")
    a = p.parse_args()
    indices = (
        a.episodes
        if a.episodes is not None
        else list(range(len(source_episodes(a.source))))
    )
    for index in indices:
        receipt = process_episode(
            a.source, a.raw_source, a.output, a.work_dir, a.urdf, a.mesh_root, index
        )
        print(json.dumps(dict(episode=index, frames=receipt["length"])), flush=True)
    if a.episodes is None:
        print(finalize(a.source, a.output, a.repo_id), flush=True)


if __name__ == "__main__":
    main()
