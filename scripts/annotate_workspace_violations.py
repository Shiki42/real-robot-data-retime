"""Label nominal preview video with measured-state URDF distance violations."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

from real_robot_data_retime.collision.diagnostics import (
    minimum_mesh_distance,
    violation_intervals,
)
from real_robot_data_retime.collision.piperx import PiperXClearance
from real_robot_data_retime.interaction.video import write_video


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("render", "urdf", "mesh-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--threshold-mm", type=float, default=50.0)
    args = parser.parse_args()
    root = args.render
    if not np.isfinite(args.threshold_mm) or args.threshold_mm <= 0:
        raise ValueError("invalid threshold")
    receipt = json.loads((root / "report.json").read_text())
    diagnostic = receipt.get("diagnostic_only", False)
    certified = (
        receipt["plan"]
        .get("mesh_audit", {})
        .get("observation.state", {})
        .get("passed", False)
    )
    if not diagnostic and not certified:
        raise ValueError("requires an explicit diagnostic or mesh-certified render")
    if (
        certified
        and args.threshold_mm / 1000 > receipt["plan"]["clearance_requirement_m"]
    ):
        raise ValueError("annotation threshold exceeds certified clearance")
    values = np.asarray(
        pq.read_table(root / "trajectories.parquet")["observation.state"].to_pylist()
    )
    fk = PiperXClearance(
        values[:, :7],
        values[:, 7:],
        args.urdf,
        args.mesh_root,
        margin_m=args.threshold_mm / 1000,
    )
    if certified and receipt["plan"]["stages"]["base_spacing_m"] != fk.base_spacing_m:
        raise ValueError("annotation geometry differs from certified base spacing")
    n = len(values)
    fps = receipt["source_fps"]
    times = []
    distances = []
    pairs = []
    for i in range(n):
        for k in range(4) if i < n - 1 else [0]:
            left = fk._interpolated_pose(0, i, min(i + 1, n - 1), k, 4)
            right = fk._interpolated_pose(1, i, min(i + 1, n - 1), k, 4)
            d, pair = minimum_mesh_distance(fk, left, right)
            times.append((i + k / 4) / fps)
            distances.append(d)
            pairs.append(pair)
    times = np.asarray(times)
    distances = np.asarray(distances)
    pairs = np.asarray(pairs)
    intervals = violation_intervals(times, distances, args.threshold_mm / 1000)
    np.savez_compressed(
        root / "distance_samples.npz",
        time_seconds=times,
        distance_m=distances,
        mesh_pairs=pairs,
    )
    if certified and intervals:
        raise ValueError("sampled distances contradict continuous mesh audit")
    minimum_index = int(distances.argmin())
    report = {
        "diagnostic_only": diagnostic,
        "continuous_mesh_audit_passed": certified,
        "threshold_mm": args.threshold_mm,
        "base_spacing_m": fk.base_spacing_m,
        "sample_rate_hz": 4 * fps,
        "scope": "measured-state cross-arm URDF meshes; exact at sampled poses",
        "video_frame_fps": fps,
        "minimum_sampled_distance_mm": float(distances.min() * 1000),
        "minimum_sampled_time_seconds": float(times[minimum_index]),
        "below_threshold_intervals": intervals,
        "note": (
            "Planner certified continuous clearance; displayed minimum is sampled at 120 Hz."
            if certified
            else "Between samples is not certified. Zero denotes mesh intersection, not penetration depth."
        ),
    }
    (root / "violations.json").write_text(json.dumps(report, indent=2) + "\n")
    flags = np.array(
        [
            distances[i * 4 : min(i * 4 + 5, len(distances))].min()
            < args.threshold_mm / 1000
            for i in range(n)
        ]
    )
    cap = cv2.VideoCapture(str(root / "parallel.mp4"))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    mapping = np.load(root / "source_mapping.npz")
    names = [g.name for g in fk.model.visual_model.geometryObjects]

    def render():
        for i in range(n):
            ok, frame = cap.read()
            if not ok:
                raise ValueError("diagnostic video shorter than source mapping")
            current = i * 4
            stop = min(current + 5, len(distances))
            nearest = current + int(np.argmin(distances[current:stop]))
            mm = distances[nearest] * 1000
            color = (30, 40, 240) if flags[i] else (100, 190, 130)
            out = np.full((h + 174, w, 3), 22, np.uint8)
            out[68 : 68 + h] = frame
            cv2.putText(
                out,
                f"URDF CLEARANCE VERIFIED - {fk.base_spacing_m * 100:.0f} CM"
                if certified
                else "DIAGNOSTIC ONLY - NOT ACCEPTED",
                (12, 23),
                0,
                0.58,
                (40, 70, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                out,
                f"t={i / fps:6.3f}s  d(frame)={distances[current] * 1000:6.2f} mm",
                (12, 49),
                0,
                0.53,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )
            cv2.rectangle(out, (0, 68), (w - 1, h + 67), color, 3)
            y = h + 90
            label = "BELOW LIMIT" if flags[i] else "SAMPLED ABOVE LIMIT"
            if mm == 0:
                label = "MODEL MESH OVERLAP"
            cv2.putText(
                out,
                f"{label}: interval min {mm:.2f} / {args.threshold_mm:.2f} mm",
                (12, y),
                0,
                0.49,
                color,
                1,
                cv2.LINE_AA,
            )
            pair = pairs[nearest]
            cv2.putText(
                out,
                f"L {names[pair[0]]}  /  R {names[pair[1]]}",
                (12, y + 20),
                0,
                0.43,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                out,
                f"Source L={mapping['left'][i]:.2f} R={mapping['right'][i]:.2f} | mesh sampled at {4 * fps:.0f} Hz",
                (12, y + 40),
                0,
                0.4,
                (170, 170, 170),
                1,
                cv2.LINE_AA,
            )
            for x in range(12, w - 12):
                a = int((x - 12) / (w - 24) * n)
                b = max(a + 1, int((x - 11) / (w - 24) * n))
                c = (30, 40, 240) if flags[a : min(b, n)].any() else (70, 100, 70)
                cv2.line(out, (x, y + 56), (x, y + 65), c, 1)
            x = 12 + round((w - 25) * i / max(1, n - 1))
            cv2.line(out, (x, y + 52), (x, y + 69), (255, 255, 255), 2)
            yield out

    write_video(root / "diagnostic_annotated.mp4", render(), fps)
    cap.release()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
