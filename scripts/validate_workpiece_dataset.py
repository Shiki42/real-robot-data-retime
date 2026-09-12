"""Independently audit exported workpiece data, source mappings and mesh clearance."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from real_robot_data_retime.collision.continuous import ContinuousClearance
from real_robot_data_retime.timeline.smooth import sample_rows
from real_robot_data_retime.timeline.workpiece_workspace import preparation_onsets
from real_robot_data_retime.trim import read_episode, source_episodes


def validate_timing(original, table):
    clocks = [
        np.asarray(table[f"retime.{side}_source_frame"], float)
        for side in ["left", "right"]
    ]
    for clock in clocks:
        if (
            not np.isfinite(clock).all()
            or np.any(clock < 0)
            or np.any(clock > len(original) - 1)
            or np.any(np.diff(clock) < 0)
            or np.any(np.diff(clock) > 1 + 1e-8)
        ):
            raise ValueError("invalid or skipped source clock")
    for name in ["observation.state", "action"]:
        source = np.asarray(original[name].to_pylist())
        expected = (
            np.concatenate(
                [
                    sample_rows(source[:, :7], clocks[0]),
                    sample_rows(source[:, 7:], clocks[1]),
                ],
                axis=1,
            )
            .astype(np.float32)
            .astype(float)
        )
        actual = np.asarray(table[name].to_pylist())
        np.testing.assert_array_equal(actual, expected)
    return clocks


def validate(source, dataset, urdf, meshes):
    info = json.loads((dataset / "meta/info.json").read_text())
    raw_info = json.loads((source / "meta/info.json").read_text())
    rows = source_episodes(dataset)
    raw = {r["episode_index"]: r for r in source_episodes(source)}
    if [r["episode_index"] for r in rows] != list(range(info["total_episodes"])):
        raise ValueError("output episode IDs are not contiguous")
    total = 0
    audits = []
    for row in rows:
        ep = row["episode_index"]
        table = read_episode(dataset, info, row)
        n = len(table)
        source_ids = table["retime.source_episode_index"].to_pylist()
        if len(set(source_ids)) != 1:
            raise ValueError("source episode changes inside output")
        original = read_episode(source, raw_info, raw[source_ids[0]])
        clocks = validate_timing(original, table)
        np.testing.assert_array_equal(
            np.asarray(table["index"]), np.arange(total, total + n)
        )
        np.testing.assert_allclose(
            np.asarray(table["timestamp"]),
            np.arange(n) / info["fps"],
            atol=1e-5,
            rtol=0,
        )
        receipt = json.loads(
            (dataset / f"meta/retime_receipts/episode_{ep:03d}.json").read_text()
        )
        evidence = dataset / f"meta/evidence/episode_{ep:03d}"
        release_path = evidence / "measured_release_audit.json"
        if release_path.is_file():
            from real_robot_data_retime.tasks.workpiece import (
                refine_deposition_releases,
            )

            recorded = json.loads(release_path.read_text())["releases"]
            visual = json.loads((evidence / "interaction_timeline.json").read_text())
            for item in recorded:
                event = next(
                    e
                    for e in visual["episodes"]
                    if e["robot_id"] == item["arm"]
                    and e["object_id"] == item["object_id"]
                )
                if event["release_frame"] != item["release_frame"]:
                    raise ValueError(
                        "measured release disagrees with exported timeline"
                    )
                event["release_frame"] = item["visual_release_frame"]
            _, recomputed = refine_deposition_releases(
                visual, np.asarray(original["observation.state"].to_pylist())
            )
            if recomputed != recorded:
                raise ValueError("measured release evidence cannot be reproduced")
        plan = receipt["plan"]
        starts = preparation_onsets(
            np.asarray(original["observation.state"].to_pylist()),
            np.asarray(original["action"].to_pylist()),
            [
                [
                    {
                        "pickup_frame": next(
                            p["source_frame"]
                            for p in plan["pickup_order"]
                            if p["arm"] == side
                        )
                    }
                ]
                for side in ("left", "right")
            ],
        )
        if [c[0] for c in clocks] != starts:
            raise ValueError("preparation was truncated")
        if [p["arm"] for p in plan["pickup_order"]] != [
            "left",
            "right",
            "left",
            "right",
        ]:
            raise ValueError("pickup order changed")
        for camera in ["top", "left_wrist", "right_wrist"]:
            key = "observation.images." + camera
            prefix = "videos/" + key
            path = dataset / info["video_path"].format(
                video_key=key,
                chunk_index=row[prefix + "/chunk_index"],
                file_index=row[prefix + "/file_index"],
            )
            cap = cv2.VideoCapture(str(path))
            feature = info["features"][key]
            if (
                round(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != n
                or abs(cap.get(cv2.CAP_PROP_FPS) - info["fps"]) > 1e-4
            ):
                raise ValueError("video length/fps mismatch")
            for index in [0, n // 2, n - 1]:
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, image = cap.read()
                if not ok or list(image.shape) != feature["shape"]:
                    raise ValueError("video decode/shape mismatch")
            cap.release()
        state = np.asarray(table["observation.state"].to_pylist())
        checker = ContinuousClearance(
            state[:, :7], state[:, 7:], urdf, meshes, clearance_m=0.05
        )
        if not all(checker(i, i, i + 1, i + 1) for i in range(n - 1)):
            raise ValueError(f"exported float32 clearance failed: {ep}")
        audits.append(
            {
                "episode": ep,
                "source_episode": source_ids[0],
                "frames": n,
                "passed": True,
                "interior_samples": checker.checked_samples,
            }
        )
        checker.__call__.cache_clear()
        total += n
        print(json.dumps(audits[-1]), flush=True)
    if total != info["total_frames"]:
        raise ValueError("dataset total frame count mismatch")
    report = {
        "passed": True,
        "episodes": len(rows),
        "frames": total,
        "minimum_clearance_m": 0.05,
        "base_spacing_m": 0.59,
        "audits": audits,
    }
    (dataset / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["source", "dataset", "urdf", "mesh-root"]:
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    validate(a.source, a.dataset, a.urdf, a.mesh_root)


if __name__ == "__main__":
    main()
