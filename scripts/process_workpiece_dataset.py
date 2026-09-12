"""Process a pinned workpiece manifest with bounded analysis/render workers."""

import argparse
import json
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from real_robot_data_retime.automatic_dataset import (
    check_output_location,
    finalize,
    remap_table,
)
from real_robot_data_retime.stats import feature_statistics
from real_robot_data_retime.trim import image_statistics, read_episode, source_episodes
from real_robot_data_retime.video import remap_video


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def completed_analysis(record):
    root = Path(record["analysis"])
    return (
        (root / "progress.json").exists()
        and (root / "report.json").exists()
        and json.loads((root / "progress.json").read_text())["stage"] == "complete"
        and json.loads((root / "report.json").read_text())["success"]
    )


def analyze(record):
    external = record.get("external_analysis_session")
    if external and not completed_analysis(record):
        while (
            subprocess.run(
                ["tmux", "has-session", "-t", external],
                capture_output=True,
                check=False,
            ).returncode
            == 0
        ):
            time.sleep(5)
    if not completed_analysis(record):
        from real_robot_data_retime.interaction.pipeline import run

        root = Path(record["analysis"])
        cached = (root / "measurements.json").exists()
        report = run(record["source_video"], root, "workpiece", retry_objects=False)
        if not report["success"] and not cached:
            report = run(
                record["source_video"],
                root,
                "workpiece",
                reuse_measurements=root,
                retry_objects=True,
            )
        if not report["success"]:
            return {"status": "analysis_rejected", "report": str(root / "report.json")}
    return {"status": "analyzed"}


def source_waiting_configs(cache, configs):
    """Add outside parking poses that cube inflation cannot select."""
    from itertools import product

    from real_robot_data_retime.timeline.workpiece_workspace import (
        workspace_waiting_stops,
    )

    base = None
    for config in configs:
        try:
            stops = workspace_waiting_stops(cache, config)
        except ValueError as exc:
            if str(exc) != "no outside approach for fixed-volume waiting pose":
                raise
            continue
        base = config
        break
    if base is None:
        return []
    auto = [stops[0][0], *stops[1]]
    own, fk = cache["own"], cache["fk"]
    extra = np.asarray(base["waiting_padding_extra_m"])
    radii = base["ee_radius_m"] + base["waiting_padding_m"] + extra
    lo, hi = np.asarray(base["minimum_m"]), np.asarray(base["maximum_m"])
    pools = []
    preferred = []
    owners = [
        (1, cache["starts"][1], auto[2]),
        (0, cache["starts"][0], auto[0]),
        (0, own[0][0]["release_frame"] + 1, own[0][1]["retract_end"]),
    ]
    for stage, (side, cycle) in enumerate([(0, 1), (1, 0), (1, 1)]):
        if cycle == 0:
            lower = cache["starts"][side]
        else:
            from real_robot_data_retime.interaction.evidence import stable_runs

            previous = own[side][cycle - 1]
            pickup = previous["pickup_frame"]
            clear = previous["release_evidence"]["clearance_frame"]
            aperture = np.asarray(cache["joints"][0][:, side * 7 + 6])
            held = float(np.median(aperture[pickup : min(pickup + 3, clear)]))
            openings = stable_runs(aperture[pickup : clear + 1] >= held + 5.0, 3)
            # A visible bin change plus measured jaw reopening establishes
            # release even when the forearm still occludes the bin afterwards.
            confirmed = pickup + openings[0][0] + 2 if openings else clear
            lower = max(previous["release_frame"], confirmed)
        candidates = [
            i
            for i in range(lower + 8, auto[stage] + 1)
            if not np.any(
                np.all(
                    (cache["tcp"][side, i - 8 : i + 1] >= lo - radii[stage])
                    & (cache["tcp"][side, i - 8 : i + 1] <= hi + radii[stage]),
                    axis=-1,
                )
            )
        ]
        choices = [auto[stage]]
        safest = None
        if candidates:
            _owner, begin, end = owners[stage]
            for stop in reversed(candidates):
                safe = True
                for other in range(begin, end + 1):
                    pair = (stop, other) if side == 0 else (other, stop)
                    if pair not in cache["configuration_cache"]:
                        cache["configuration_cache"][pair] = fk._clear(
                            fk.poses[0][pair[0]],
                            fk.poses[1][pair[1]],
                            margin_m=cache["clearance_m"] + 1e-6,
                        )
                    if not cache["configuration_cache"][pair]:
                        safe = False
                        break
                if safe:
                    safest = stop
                    break
            choices.extend([candidates[0], candidates[len(candidates) // 2]])
            if safest is not None:
                choices.insert(1, safest)
        choices = list(dict.fromkeys(choices))
        pools.append(choices)
        preferred.append(safest if safest is not None else choices[0])
    tuples = [tuple(preferred), *product(*pools)]
    return [
        dict(base, waiting_source_frames=list(t))
        for t in dict.fromkeys(tuples)
        if list(t) != auto
    ]


def process(record, manifest, work):
    from real_robot_data_retime.interaction.checkpoint_render import render_checkpoint
    from real_robot_data_retime.staged import load_joints
    from real_robot_data_retime.timeline.scheduler import NoSafeSchedule
    from real_robot_data_retime.timeline.workpiece_workspace import (
        DEFAULT_WORKSPACE,
        plan_workspace,
        workspace_planning_cache,
        workspace_waiting_stops,
    )

    if not completed_analysis(record):
        raise ValueError("analysis is not complete and accepted")
    ep = record["episode"]
    root = Path(record["work_dir"])
    analysis = Path(record["analysis"])
    urdf = Path(manifest["urdf"])
    meshes = Path(manifest["mesh_root"])
    timeline = json.loads((analysis / "interaction_timeline.json").read_text())
    if not all(
        e.get("release_evidence") is not None and e["release_evidence"]["verified"]
        for e in timeline["episodes"]
    ):
        raise ValueError("destination deposition is not verified")

    joints = load_joints(Path(record["joint_data"]), urdf, meshes, timeline)
    if "verified_render" in record:
        render = Path(record["verified_render"])
        from real_robot_data_retime.model_experiment import sha256

        report = json.loads((render / "report.json").read_text())
        if report["source_sha256"] != sha256(record["source_video"]):
            raise ValueError("verified render belongs to a different source")
        exported = np.asarray(
            pq.read_table(render / "trajectories.parquet")[
                "observation.state"
            ].to_pylist()
        )
        mapping = np.load(render / "source_mapping.npz")
        from real_robot_data_retime.timeline.smooth import sample_rows

        expected = np.concatenate(
            [
                sample_rows(joints[0][:, :7], mapping["left"]),
                sample_rows(joints[0][:, 7:], mapping["right"]),
            ],
            axis=1,
        )
        np.testing.assert_allclose(exported, expected)
        actual_action = np.asarray(
            pq.read_table(render / "trajectories.parquet")["action"].to_pylist()
        )
        expected_action = np.concatenate(
            [
                sample_rows(joints[1][:, :7], mapping["left"]),
                sample_rows(joints[1][:, 7:], mapping["right"]),
            ],
            axis=1,
        )
        np.testing.assert_allclose(actual_action, expected_action)
    else:
        attempts = []
        chosen = None
        from itertools import product

        cache = workspace_planning_cache(
            timeline, joints, DEFAULT_WORKSPACE["minimum_clearance_m"]
        )
        configs = [
            dict(DEFAULT_WORKSPACE, waiting_padding_m=padding)
            for padding in [
                0.0,
                0.04,
                0.09,
                0.12,
                0.02,
                0.06,
                0.01,
                0.03,
                0.05,
                0.07,
                0.08,
                0.10,
                0.11,
            ]
        ]
        configs.extend(
            dict(DEFAULT_WORKSPACE, waiting_padding_extra_m=list(extra))
            for extra in sorted(product([0.0, 0.04, 0.09, 0.12], repeat=3), key=sum)
            if len(set(extra)) > 1
        )

        def candidates():
            yield from configs
            valid = []
            for candidate in configs:
                try:
                    workspace_waiting_stops(cache, candidate)
                except ValueError as exc:
                    if str(exc) != "no outside approach for fixed-volume waiting pose":
                        raise
                    continue
                valid.append(candidate)
            if valid:
                for candidate in (valid[0], valid[-1]):
                    for enabled in product([False, True], repeat=3):
                        if all(enabled):
                            continue
                        yield dict(candidate, waiting_enabled=list(enabled))
            yield from source_waiting_configs(cache, configs)

        seen_waits = set()
        for config in candidates():
            try:
                stops = workspace_waiting_stops(cache, config)
                key = tuple(tuple(arm) for arm in stops)
                if key in seen_waits:
                    continue
                seen_waits.add(key)
                left, right, plan = plan_workspace(
                    timeline, joints, config, cache=cache
                )
            except NoSafeSchedule as exc:
                attempts.append({"workspace": config, "reason": str(exc)})
                continue
            except ValueError as exc:
                if str(exc) not in [
                    "no outside approach for fixed-volume waiting pose",
                    "no braking path after the protected execution",
                    "outside stopping ramp interrupts preceding placement",
                ]:
                    raise
                attempts.append({"workspace": config, "reason": str(exc)})
                continue
            chosen = config
            break
        write_json(root / "planning_attempts.json", attempts)
        if chosen is None:
            return {
                "status": "planning_rejected",
                "attempts": str(root / "planning_attempts.json"),
            }
        write_json(root / "workspace.json", chosen)
        render = root / "render"
        if render.exists():
            raise ValueError(
                "partial render exists; inspect and move it before retrying"
            )
        report = render_checkpoint(
            Path(record["source_video"]),
            analysis,
            render,
            joint_data=Path(record["joint_data"]),
            urdf=urdf,
            mesh_root=meshes,
            fixed_workspace=True,
            workspace=chosen,
        )
    plan = report["plan"]
    if (
        not report["source_origin_checks_passed"]
        or not plan["mesh_audit"]["observation.state"]["passed"]
    ):
        raise ValueError("render or continuous clearance audit failed")
    if (
        plan["clearance_requirement_m"] != 0.05
        or plan["stages"]["base_spacing_m"] != 0.59
    ):
        raise ValueError("wrong physical clearance or base spacing")
    if "preparation_onsets" not in plan["stages"]:
        raise ValueError("missing complete preparation audit")
    mapping = np.load(render / "source_mapping.npz")
    left, right = mapping["left"], mapping["right"]
    source = Path(manifest["source"])
    info = json.loads((source / "meta/info.json").read_text())
    row = next(row for row in source_episodes(source) if row["episode_index"] == ep)
    table = read_episode(source, info, row)
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
    mapped = mapped.append_column(
        "retime.synthetic_hold", pa.array(np.zeros(len(left), bool))
    )
    mapped = mapped.append_column(
        "retime.source_episode_index", pa.array(np.full(len(left), ep, dtype=np.int64))
    )
    for side, clock in [("left", left), ("right", right)]:
        mapped = mapped.append_column(
            f"retime.{side}_idle", pa.array(np.r_[np.diff(clock) == 0, True])
        )
    package = root / "package"
    package.mkdir(exist_ok=True)
    pq.write_table(mapped, package / "data.parquet")
    stats = {
        key: feature_statistics(
            np.asarray(mapped[key].to_pylist(), float).reshape(len(mapped), -1)
        )
        for key in mapped.column_names
    }
    shutil.copyfile(render / "parallel.mp4", package / "top.mp4")
    for side, clock in [("left", left), ("right", right)]:
        camera = f"observation.images.{side}_wrist"
        prefix = "videos/" + camera
        src = source / info["video_path"].format(
            video_key=camera,
            chunk_index=row[prefix + "/chunk_index"],
            file_index=row[prefix + "/file_index"],
        )
        first = round(row[prefix + "/from_timestamp"] * info["fps"])
        feature = info["features"][camera]
        remap_video(
            src,
            package / f"{side}_wrist.mp4",
            clock + first,
            info["fps"],
            feature["shape"][1],
            feature["shape"][0],
        )
    for camera in ["top", "left_wrist", "right_wrist"]:
        stats["observation.images." + camera] = image_statistics(
            package / f"{camera}.mp4", len(mapped)
        )
    write_json(
        package / "receipt.json",
        {
            "episode_index": ep,
            "length": len(mapped),
            "tasks": row["tasks"],
            "statistics": stats,
            "source_episode_index": ep,
            "plan": plan,
            "render_report": str(render / "report.json"),
            "analysis_directory": str(analysis),
        },
    )
    return {
        "status": "processed",
        "package": str(package),
        "render": str(render),
        "length": len(mapped),
    }


def finalize_packages(manifest, work, output):
    accepted = []
    failures = []
    for record in manifest["episodes"]:
        p = Path(record["work_dir"]) / "process_status.json"
        status = (
            json.loads(p.read_text())
            if p.exists()
            else json.loads(
                (Path(record["work_dir"]) / "analysis_status.json").read_text()
            )
        )
        phase = status["status"]
        if phase != "processed" and not phase.endswith(("_rejected", "_error")):
            raise ValueError(f"episode {record['episode']} is not terminal: {phase}")
        (accepted if status["status"] == "processed" else failures).append(
            (record, status)
        )
    if not accepted:
        raise ValueError("no accepted episodes to export")
    template = work / "metadata_template"
    source = Path(manifest["source"])
    shutil.copytree(source / "meta", template / "meta", dirs_exist_ok=True)
    info = json.loads((template / "meta/info.json").read_text())
    info["features"]["observation.images.top"] = info["features"].pop(
        "observation.images.right_environment_1"
    )
    for key, dtype in [
        ("retime.source_episode_index", "int64"),
        ("retime.left_idle", "bool"),
        ("retime.right_idle", "bool"),
    ]:
        info["features"][key] = {"dtype": dtype, "shape": [1], "names": None}
    write_json(template / "meta/info.json", info)
    for new_id, (record, status) in enumerate(accepted):
        package = Path(status["package"])
        table = pq.read_table(package / "data.parquet")
        table = table.set_column(
            table.schema.get_field_index("episode_index"),
            "episode_index",
            pa.array(
                np.full(len(table), new_id),
                type=table.schema.field("episode_index").type,
            ),
        )
        dest = output / f"data/chunk-000/file-{new_id:03d}.parquet"
        dest.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, dest)
        receipt = json.loads((package / "receipt.json").read_text())
        receipt["episode_index"] = new_id
        evidence = output / f"meta/evidence/episode_{new_id:03d}"
        evidence.mkdir(parents=True, exist_ok=True)
        for name in (
            "report.json",
            "interaction_timeline.json",
            "measurements.json",
            "robot_mask_audit.json",
        ):
            shutil.copyfile(Path(receipt["analysis_directory"]) / name, evidence / name)
        shutil.copyfile(receipt["render_report"], evidence / "render_report.json")
        receipt["statistics"]["episode_index"] = feature_statistics(
            np.full((len(table), 1), new_id)
        )
        write_json(output / f"meta/retime_receipts/episode_{new_id:03d}.json", receipt)
        for camera in ["top", "left_wrist", "right_wrist"]:
            dest = (
                output
                / f"videos/observation.images.{camera}/chunk-000/file-{new_id:03d}.mp4"
            )
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(package / f"{camera}.mp4", dest)
    summary = finalize(
        template,
        output,
        "workpiece-fullprep-5cm",
        episode_indices=list(range(len(accepted))),
    )
    summary.update(
        source_episodes=len(manifest["episodes"]),
        accepted_source_episodes=[r["episode"] for r, s in accepted],
        rejected=[dict(episode=r["episode"], **s) for r, s in failures],
        source_revision=manifest["source_revision"],
        source_dataset=manifest["source"],
        camera_semantics={
            "top": "main-view counterfactual composite",
            "left_wrist": "left source-clock replay",
            "right_wrist": "right source-clock replay",
        },
        minimum_clearance_m=0.05,
        base_spacing_m=0.59,
    )
    write_json(output / "processing_report.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=["all", "analyze", "process", "finalize"], default="all"
    )
    parser.add_argument("--episode", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--analysis-workers", type=int, default=2)
    parser.add_argument("--process-workers", type=int, default=1)
    args = parser.parse_args()
    work = args.manifest.parent
    manifest = json.loads(args.manifest.read_text())
    cv2.setNumThreads(2)
    check_output_location(
        Path(manifest["source"]), Path(manifest["source"]), args.output
    )
    if args.stage in ["analyze", "process"]:
        record = next(r for r in manifest["episodes"] if r["episode"] == args.episode)
        path = Path(record["work_dir"]) / (
            "analysis_status.json" if args.stage == "analyze" else "process_status.json"
        )
        if args.stage == "process" and path.exists():
            previous = json.loads(path.read_text())
            if previous.get("status") == "processed":
                package = Path(previous["package"])
                for name in [
                    "data.parquet",
                    "receipt.json",
                    "top.mp4",
                    "left_wrist.mp4",
                    "right_wrist.mp4",
                ]:
                    if not (package / name).is_file():
                        raise FileNotFoundError(package / name)
                print(json.dumps({"episode": args.episode, **previous}), flush=True)
                return
        write_json(
            path, {"status": "running", "stage": args.stage, "started": time.time()}
        )
        try:
            result = (
                analyze(record)
                if args.stage == "analyze"
                else process(record, manifest, work)
            )
        except ValueError as exc:
            traceback.print_exc()
            result = {"status": args.stage + "_rejected", "reason": str(exc)}
        write_json(path, result)
        print(json.dumps({"episode": args.episode, **result}), flush=True)
        return
    if args.stage == "finalize":
        print(json.dumps(finalize_packages(manifest, work, args.output)), flush=True)
        return

    def invoke(stage, record):
        log = Path(record["work_dir"]) / (stage + ".log")
        with log.open("w") as stream:
            result = subprocess.run(
                [
                    sys.executable,
                    __file__,
                    "--manifest",
                    str(args.manifest),
                    "--output",
                    str(args.output),
                    "--stage",
                    stage,
                    "--episode",
                    str(record["episode"]),
                ],
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        status_path = Path(record["work_dir"]) / (
            "analysis_status.json" if stage == "analyze" else "process_status.json"
        )
        if result.returncode:
            failure = {
                "status": stage + "_error",
                "returncode": result.returncode,
                "log": str(log),
            }
            write_json(status_path, failure)
        result = json.loads(status_path.read_text())
        print(json.dumps({"episode": record["episode"], **result}), flush=True)
        return result

    with (
        ThreadPoolExecutor(args.analysis_workers) as ap,
        ThreadPoolExecutor(args.process_workers) as pp,
    ):
        tasks = {ap.submit(invoke, "analyze", r): r for r in manifest["episodes"]}
        renders = []
        for task in as_completed(tasks):
            record = tasks[task]
            if task.result()["status"] == "analyzed":
                renders.append(pp.submit(invoke, "process", record))
        for task in as_completed(renders):
            task.result()
    print(json.dumps(finalize_packages(manifest, work, args.output)), flush=True)


if __name__ == "__main__":
    main()
