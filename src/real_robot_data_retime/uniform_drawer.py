"""Two uniformly spaced prerequisite timings per source drawer episode."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

from .automatic_dataset import check_output_location, write_retimed_episode, finalize
from .compositing.layers import composite
from .edit import registered_frames, native_render_inputs
from .interaction.measurements import inputs_fingerprint, producer_fingerprint
from .model_experiment import sha256
from .staged import load_joints
from .timeline.visual import plan_visual
from .timeline.drawer_wait import prepare_uniform_lift
from .timeline.scheduler import NoSafeSchedule
from .collision.drawer import DrawerGeometry
from .timeline.smooth import sample_rows
from .timeline.uniform import uniform_samples, validate_stage_schedule
from .trim import source_episodes, read_episode


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def source_map_digest(left, right):
    digest = hashlib.sha256()
    for name, values in (("left", left), ("right", right)):
        array = np.asarray(values, dtype="<f8")
        digest.update(f"{name}:{array.shape}".encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def source_inputs(config, index):
    source = Path(config["source"])
    info = read_json(source / "meta/info.json")
    row = source_episodes(source)[index]
    ep = row["episode_index"]
    video = source / info["video_path"].format(
        video_key="observation.images.top", chunk_index=0, file_index=ep
    )
    analysis = Path(config["analyses"][str(ep)])
    report = read_json(analysis / "report.json")
    if (
        read_json(analysis / "progress.json")["stage"] != "complete"
        or not report["success"]
        or not all(report["validation_gates"].values())
    ):
        raise ValueError(f"episode {ep}: analysis is not verified")
    manifest = read_json(analysis / "measurements.json")
    if manifest["inputs"]["video_sha256"] != sha256(video):
        raise ValueError(f"episode {ep}: analysis video differs from source")
    return source, info, row, video, analysis, report, manifest


def plan_episode(config, index):
    source, info, row, video, analysis, report, manifest = source_inputs(config, index)
    timeline = read_json(analysis / "interaction_timeline.json")
    table_path = source / info["data_path"].format(
        chunk_index=0, file_index=row["episode_index"]
    )
    joints = load_joints(table_path, config["urdf"], config["mesh_root"], timeline)
    output = Path(config["work"]) / f"episode_{index:03d}"
    output.mkdir(parents=True, exist_ok=True)
    (output / "planned.json").unlink(missing_ok=True)
    with (
        np.load(analysis / "tracks.npz") as tracks,
        np.load(analysis / "segmentation.npz") as segmentation,
    ):
        frames, fps = registered_frames(
            video, tracks["registration"], int(segmentation["frame_shape"][1])
        )
        if (
            inputs_fingerprint(
                video, frames, manifest["inputs"]["proposals"], tracks["registration"]
            )
            != manifest["inputs"]
        ):
            raise ValueError("analysis frame geometry or registration differs")
        positions = uniform_samples(index, info["total_episodes"])
        width = int(segmentation["frame_shape"][1])
        robots = np.unpackbits(segmentation["robots"], axis=-1, count=width).astype(
            bool
        )
        objects = np.unpackbits(segmentation["objects"], axis=-1, count=width).astype(
            bool
        )
        from .compositing.ownership import exclude_placed_objects

        exclude_placed_objects(robots, objects, timeline["episodes"])
        from .interaction.origin_identity import detached_origin

        event = timeline["episodes"][0]
        if not detached_origin(
            robots, objects, event["object_id"], event["pickup_frame"], fps
        )["verified"]:
            raise ValueError(
                "drawer target is robot-attached at origin; reselect interaction evidence"
            )
        prepared = prepare_uniform_lift(
            joints[0],
            joints[1],
            event,
            timeline["drawer_motion"],
            robots,
            objects,
            timeline["episodes"],
            joints[2],
            joints[3],
            fps,
            positions[0],
            scene_geometry=config.get("scene_geometry"),
        )
        attempts = []
        for peak in prepared.candidates:
            plans = []
            for variant, position in enumerate(positions):
                try:
                    plans.append(
                        plan_visual(
                            timeline,
                            frames,
                            tracks,
                            segmentation,
                            joints=joints,
                            uniform_position=position,
                            uniform_lift=prepared,
                            wait_source_frame=peak,
                        )
                    )
                except NoSafeSchedule as error:
                    attempts.append(
                        dict(peak_source_frame=peak, variant=variant, reason=str(error))
                    )
                    break
            if len(plans) == 2:
                break
        else:
            write_json(
                output / "feasibility.json", dict(passed=False, attempts=attempts)
            )
            raise NoSafeSchedule(
                f"no held peak satisfies both sampled variants; see {output / 'feasibility.json'}"
            )
        write_json(
            output / "feasibility.json",
            dict(passed=True, selected_peak=peak, attempts=attempts),
        )
        results = []
        for variant, (left, right, plan) in enumerate(plans):
            plan["wait_candidate_search"] = dict(
                selected_peak=peak, rejected_candidates=attempts
            )
            plan["stage_validation"] = validate_stage_schedule(
                left, right, plan["stages"]
            )
            # Preserve the source end; no invented terminal padding in this exporter.
            plan["synthetic_terminal_hold"] = dict(start_frame=len(left), frames=0)
            plan["source_episode_index"] = int(row["episode_index"])
            plan["variant"] = variant
            plan["source_sha256"] = sha256(video)
            plan["joint_data_sha256"] = sha256(table_path)
            plan["analysis_report_sha256"] = sha256(analysis / "report.json")
            plan["producer"] = producer_fingerprint()
            plan["source_map_digest"] = source_map_digest(left, right)
            np.savez_compressed(
                output / f"mapping_{variant}.npz", left=left, right=right
            )
            write_json(output / f"plan_{variant}.json", plan)
            results.append(plan["stages"]["uniform"])
    if not np.isclose(results[1]["position"] - results[0]["position"], 0.5):
        raise ValueError("variants do not differ by half the timing interval")
    write_json(
        output / "planned.json", dict(source=index, variants=results, passed=True)
    )
    return results


def render_episode(config, index):
    source, info, row, video, analysis, report, manifest = source_inputs(config, index)
    output, work = Path(config["output"]), Path(config["work"]) / f"episode_{index:03d}"
    check_output_location(source, Path(config["raw_source"]), output)
    if not read_json(work / "planned.json")["passed"]:
        raise ValueError("source has not passed both schedule checks")
    timeline = read_json(analysis / "interaction_timeline.json")
    table = read_episode(source, info, row)
    trim = read_json(source / "trim_manifest.json")["episodes"][index]
    interaction = dict(
        timeline=timeline,
        report=report,
        measurements=manifest,
        robot_mask_audit=read_json(analysis / "robot_mask_audit.json"),
    )
    with (
        np.load(analysis / "tracks.npz") as tracks,
        np.load(analysis / "segmentation.npz") as segmentation,
    ):
        native, masks, _ = native_render_inputs(
            video, tracks["registration"], segmentation
        )
    for variant in range(2):
        ep = index + info["total_episodes"] * variant
        plan = read_json(work / f"plan_{variant}.json")
        if plan["producer"] != producer_fingerprint() or plan[
            "source_sha256"
        ] != sha256(video):
            raise ValueError("source or implementation changed since planning")
        if (
            plan["stages"]["scene_geometry"]
            != DrawerGeometry.from_mapping(config.get("scene_geometry")).to_dict()
        ):
            raise ValueError("scene geometry changed since planning")
        table_path = source / info["data_path"].format(
            chunk_index=0, file_index=row["episode_index"]
        )
        if plan["joint_data_sha256"] != sha256(table_path):
            raise ValueError("joint data changed since planning")
        if plan["analysis_report_sha256"] != sha256(analysis / "report.json"):
            raise ValueError("analysis changed since planning")
        with np.load(work / f"mapping_{variant}.npz") as maps:
            left, right = maps["left"], maps["right"]
        if source_map_digest(left, right) != plan["source_map_digest"]:
            raise ValueError("source mapping changed after planning")
        validate_stage_schedule(left, right, plan["stages"])
        receipt_path = output / f"meta/retime_receipts/episode_{ep:03d}.json"
        receipt_path.unlink(missing_ok=True)
        debug = work / f"variant_{variant}"
        debug.mkdir(parents=True, exist_ok=True)
        main = output / f"videos/observation.images.top/chunk-000/file-{ep:03d}.mp4"
        rendered = composite(native, timeline, masks, left, right, main, debug)
        if not rendered["automatic_origin_audit"]["passed"]:
            raise ValueError(f"output {ep}: rendered source-origin check failed")
        rendered["output"] = str(main.relative_to(output))
        rendered["implementation_fingerprint"] = producer_fingerprint()
        receipt = write_retimed_episode(
            source,
            output,
            row,
            ep,
            table,
            left,
            right,
            trim,
            plan,
            rendered,
            plan["source_sha256"],
            interaction,
        )
        print(
            json.dumps(
                dict(
                    output_episode=ep,
                    source_episode=index,
                    variant=variant,
                    frames=receipt["length"],
                )
            ),
            flush=True,
        )


def finalize_dataset(config):
    source, output = Path(config["source"]), Path(config["output"])
    count = read_json(source / "meta/info.json")["total_episodes"]
    rows = source_episodes(source)
    info = read_json(source / "meta/info.json")
    producer = producer_fingerprint()
    for index in range(count):
        original = read_episode(source, info, rows[index])
        pair = []
        for variant in range(2):
            ep = index + count * variant
            receipt = read_json(output / f"meta/retime_receipts/episode_{ep:03d}.json")
            if receipt.get("visual_review", {}).get("passed") is False:
                raise ValueError(f"output {ep}: explicitly rejected by visual review")
            if receipt["plan"]["producer"] != producer:
                raise ValueError(
                    f"output {ep}: stale producer; replan and render with current code"
                )
            stages = receipt["plan"]["stages"]
            if (
                stages["scene_geometry"]
                != DrawerGeometry.from_mapping(config.get("scene_geometry")).to_dict()
            ):
                raise ValueError("receipt scene geometry differs from current config")
            if (
                not stages["origin_identity"]["verified"]
                or not stages["post_open_motion_validation"]["passed"]
            ):
                raise ValueError(f"output {ep}: failed target/motion evidence")
            if (
                abs(stages["tcp_height_m"] - stages["recorded_peak_height_m"])
                > 0.002 + 1e-9
            ):
                raise ValueError(
                    f"output {ep}: waiting pose is not in the recorded lift peak band"
                )
            if not receipt["compositing"]["automatic_origin_audit"]["passed"]:
                raise ValueError(f"output {ep}: failed rendered origin audit")
            if (
                receipt["source_episode_index"] != index
                or receipt["plan"]["variant"] != variant
            ):
                raise ValueError("output/source assignment mismatch")
            with np.load(
                output / f"meta/retime_source_indices/episode_{ep:03d}.npz"
            ) as maps:
                if (
                    source_map_digest(maps["left"], maps["right"])
                    != receipt["plan"]["source_map_digest"]
                ):
                    raise ValueError(
                        "exported source mapping differs from validated plan"
                    )
                from .timeline.planner import original_pair_edge

                for replay in stages["recorded_pair_replay"]["edges"]:
                    k = replay["output_edge"]
                    if not original_pair_edge(
                        maps["left"][k],
                        maps["right"][k],
                        maps["left"][k + 1],
                        maps["right"][k + 1],
                        {replay["source_edge"]},
                    ):
                        raise ValueError(
                            "recorded pair replay changed source pairing or speed"
                        )
                validate_stage_schedule(
                    maps["left"], maps["right"], receipt["plan"]["stages"]
                )
                table = pq.read_table(output / f"data/chunk-000/file-{ep:03d}.parquet")
                for key in ("action", "observation.state"):
                    values = np.asarray(original[key].to_pylist())
                    expected = np.c_[
                        sample_rows(values[:, :7], maps["left"]),
                        sample_rows(values[:, 7:], maps["right"]),
                    ]
                    if not np.allclose(
                        np.asarray(table[key].to_pylist()),
                        expected,
                        rtol=1e-6,
                        atol=1e-6,
                    ):
                        raise ValueError(
                            f"output {ep}: {key} differs from its source clocks"
                        )
                if len(table) != receipt["length"] or set(
                    table["episode_index"].to_pylist()
                ) != {ep}:
                    raise ValueError("output episode identity or length differs")
                if not np.allclose(
                    table["timestamp"].to_pylist(),
                    np.arange(len(table)) / info["fps"],
                    atol=1e-5,
                ):
                    raise ValueError("output timestamps differ from frame clock")
            if (
                receipt["plan"]["stages"]["uniform"]["position"]
                != uniform_samples(index, count)[variant]
            ):
                raise ValueError("sample position differs from global uniform grid")
            pair.append(receipt["plan"]["stages"]["uniform"])
        if (
            pair[0]["a_frames"] != pair[1]["a_frames"]
            or pair[0]["b_frames"] != pair[1]["b_frames"]
        ):
            raise ValueError("prerequisite durations differ between variants")
        if not np.isclose(pair[1]["position"] - pair[0]["position"], 0.5):
            raise ValueError("incorrect pair spacing")
    result = finalize(
        source, output, config["repo_id"], episode_indices=range(2 * count)
    )
    write_json(
        output / "uniform_manifest.json",
        dict(
            **result,
            source=config["source_provenance"],
            source_episodes=count,
            variants_per_source=2,
            interval="A entirely before B to B entirely before A",
            grid="u=(source_episode + variant*N)/(2*N), [0,1)",
            pair_spacing=0.5,
            frame_rounding="nearest frame; at most 0.5 frame per onset",
            dependency="C starts only after A and B complete",
            producer=producer_fingerprint(),
            collision_scope="projected new pairs, exact original paired replay, metric held peak/braking clearance",
            source_config=config,
        ),
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--phase", required=True, choices=["plan", "render", "finalize"]
    )
    parser.add_argument("--episode", type=int)
    args = parser.parse_args()
    config = read_json(args.config)
    if args.phase == "finalize":
        result = finalize_dataset(config)
    else:
        if args.episode is None:
            parser.error("--episode is required for plan/render")
        result = {"plan": plan_episode, "render": render_episode}[args.phase](
            config, args.episode
        )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
