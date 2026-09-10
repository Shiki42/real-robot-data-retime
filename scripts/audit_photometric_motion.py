"""Compare unchanged robot predictions against raw/corrected motion evidence."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from real_robot_data_retime.accuracy_experiment import load_scene
from real_robot_data_retime.interaction.measurements import producer_fingerprint
from real_robot_data_retime.interaction.photometric_motion import photometric_motion
from real_robot_data_retime.interaction.robot_discovery import robot_mask_audit
from real_robot_data_retime.interaction.video import write_video
from real_robot_data_retime.model_experiment import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "arms", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--candidate", type=Path)
    a = p.parse_args()
    cv2.setNumThreads(4)
    frames, fps, original = load_scene(a.input, a.arms)
    a.output.mkdir(parents=True, exist_ok=False)
    masks = original
    candidate_report = None
    if a.candidate:
        candidate_report = json.loads((a.candidate / "report.json").read_text())
        if candidate_report["input_sha256"] != sha256(a.input) or candidate_report[
            "arm_report_sha256"
        ] != sha256(a.arms / "report.json"):
            raise ValueError("candidate source identity differs")
        with np.load(a.candidate / "masks.npz") as data:
            if (
                not np.array_equal(data["source_indices"], np.arange(len(frames)))
                or int(data["width"]) != frames.shape[2]
            ):
                raise ValueError("candidate source mapping differs")
            masks = data["candidate_robots"]
        if masks.shape != original.shape:
            raise ValueError("candidate mask geometry differs")
    producer = producer_fingerprint()
    evidence = photometric_motion(frames)
    raw, corrected, coefficients = (
        evidence.raw,
        evidence.discovery,
        evidence.coefficients,
    )
    report = {
        "input_sha256": sha256(a.input),
        "arm_report_sha256": sha256(a.arms / "report.json"),
        "producer": producer,
        "correction": "photometric_opaque_support",
        "candidate_report_sha256": sha256(a.candidate / "report.json")
        if a.candidate
        else None,
        "raw_audit": robot_mask_audit(masks, raw, fps),
        "normalized_audit": robot_mask_audit(
            masks, corrected, fps, pixel_support=evidence.support
        ),
        "original_masks_normalized_audit": robot_mask_audit(
            original, corrected, fps, pixel_support=evidence.support
        ),
        "empty_masks_normalized_audit": robot_mask_audit(
            np.zeros_like(masks), corrected, fps
        ),
        "validated_for_compositing": False,
        "limitation": "Changed motion reference, unchanged segmentation. Coverage is not accuracy.",
    }
    np.savez_compressed(
        a.output / "motion.npz",
        raw=raw["masks"],
        corrected=corrected["masks"],
        coefficients=coefficients,
        positive_support=evidence.support,
        ambiguous=evidence.ambiguous,
        source_indices=np.arange(len(frames)),
    )

    def preview():
        for t, frame in enumerate(frames):
            panels = []
            for name, reference in [
                ("source", None),
                ("raw motion", raw["masks"][t]),
                ("opaque motion support", corrected["masks"][t] * evidence.support[t]),
            ]:
                view = frame.copy()
                if reference is not None:
                    for side, color in enumerate(([0, 200, 255], [255, 100, 0])):
                        region = reference == side + 1
                        view[region] = (
                            view[region] * 0.55 + np.array(color) * 0.45
                        ).astype(np.uint8)
                cv2.putText(
                    view, f"{name}, source {t}", (10, 22), 0, 0.55, (255, 255, 255), 2
                )
                panels.append(view)
            yield np.concatenate(panels, axis=1)

    write_video(a.output / "motion.mp4", preview(), fps)
    (a.output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
