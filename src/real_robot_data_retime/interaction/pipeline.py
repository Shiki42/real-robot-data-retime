import json
from dataclasses import asdict
from pathlib import Path
import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .discovery import (
    InteractionConfig,
    motion_and_grippers,
    object_proposals,
    track_candidates,
)
from .evidence import score_hypothesis, stable_runs
from .video import read_video, write_video
from ..tasks import PROFILES
from .registration import stabilize


def discover_task(frames):
    hsv = cv2.cvtColor(frames[0], cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    red = (
        ((hsv[:, :, 0] < 12) | (hsv[:, :, 0] > 170))
        & (hsv[:, :, 1] > 100)
        & (hsv[:, :, 2] > 80)
    )
    if np.mean(red[: h // 2]) > 0.015:
        return "drawer"
    table = hsv[h // 2 :]
    colored = (table[:, :, 1] > 100) & (table[:, :, 2] > 70)
    return "letters" if np.mean(colored) > 0.009 else "workpiece"


def run(input_path, output_dir, task=None, config=InteractionConfig()):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames, fps = read_video(input_path, width=424)
    frames, transforms, registration_confidence = stabilize(frames)
    task = task or discover_task(frames)
    profile = PROFILES[task]
    evidence = motion_and_grippers(frames)
    proposals = object_proposals(frames, profile["object_kind"], config)
    tracks = track_candidates(frames, proposals, evidence["centers"])
    candidates = []
    for side in range(2):
        aperture = evidence["apertures"][:, side]
        if not np.isfinite(aperture).any():
            continue
        threshold = max(1.0, float(np.nanmax(aperture) - np.nanmin(aperture)) * 0.08)
        closure = np.diff(aperture) < -threshold
        onset = [a + 1 for a, b in stable_runs(closure, 1)]
        for frame in onset:
            if frame < 3 or frame >= len(frames) - config.evidence_frames:
                continue
            for k, track in enumerate(tracks):
                result = score_hypothesis(
                    track["centers"],
                    evidence["centers"][:, side],
                    aperture,
                    frame,
                    frames.shape[2],
                    config,
                )
                candidates.append(
                    dict(
                        robot_id=["left", "right"][side],
                        object_id=k,
                        grasp_start=frame,
                        **result,
                    )
                )
    selected = []
    for c in sorted(candidates, key=lambda x: x["score"], reverse=True):
        if not c["accepted"]:
            continue
        if any(
            c["robot_id"] == x["robot_id"]
            and abs(c["pickup_frame"] - x["pickup_frame"]) < config.evidence_frames
            for x in selected
        ):
            continue
        selected.append(c)
    timeline = {
        "task": task,
        "fps": fps,
        "source_frames": len(frames),
        "episodes": sorted(selected, key=lambda x: x["pickup_frame"]),
    }
    report = dict(
        success=False,
        phase="interaction_understanding",
        task=task,
        num_robot_arms=int(np.sum(np.isfinite(evidence["centers"]).all(axis=(0, 2)))),
        num_object_proposals=len(proposals),
        num_manipulation_episodes=len(selected),
        status="requires_automatic_verification",
        limitations=[
            "geometry gripper estimator has not passed real-video validation",
            "occluded object tracks require segmentation-guided recovery",
        ],
        config=asdict(config),
    )
    # A plausible candidate score alone is deliberately insufficient for success.
    (output_dir / "interaction_timeline.json").write_text(
        json.dumps(timeline, indent=2, allow_nan=False)
    )
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False)
    )
    (output_dir / "grasp_candidates.json").write_text(
        json.dumps(candidates, indent=2, allow_nan=False)
    )
    np.savez_compressed(
        output_dir / "tracks.npz",
        grippers=evidence["centers"],
        apertures=evidence["apertures"],
        objects=np.array([x["centers"] for x in tracks]),
        registration=transforms,
        registration_confidence=registration_confidence,
    )
    fig, axs = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    for side, ax in enumerate(axs):
        ax.plot(np.arange(len(frames)) / fps, evidence["apertures"][:, side])
        ax.set_ylabel(["Left", "Right"][side] + " gap (px)")
    axs[-1].set_xlabel("Source time (s)")
    fig.tight_layout()
    fig.savefig(output_dir / "gripper_aperture.png")
    plt.close(fig)

    def annotated(mode):
        ids = {x["object_id"] for x in selected}
        for t, source in enumerate(frames):
            f = source.copy()
            for side in range(2):
                center = evidence["centers"][t, side]
                if np.isfinite(center).all():
                    xy = tuple(center.astype(int))
                    cv2.circle(f, xy, 12, [(255, 80, 0), (0, 220, 0)][side], 2)
                    cv2.putText(
                        f, ["left", "right"][side], xy, 0, 0.5, (0, 255, 255), 1
                    )
            if mode != "grippers":
                for k, track in enumerate(tracks):
                    if mode == "selected" and k not in ids:
                        continue
                    c = track["centers"][t]
                    if np.isfinite(c).all():
                        xy = tuple(c.astype(int))
                        cv2.circle(f, xy, 6, (0, 0, 255), 2)
                        cv2.putText(f, str(k), xy, 0, 0.5, (0, 0, 255), 1)
            cv2.putText(
                f, f"{task} source {t} | UNVERIFIED", (8, 22), 0, 0.5, (0, 0, 255), 1
            )
            yield f

    for filename, mode in [
        ("gripper_tracks.mp4", "grippers"),
        ("interaction_candidates.mp4", "all"),
        ("selected_object_candidates.mp4", "selected"),
    ]:
        write_video(output_dir / filename, annotated(mode), fps)
    return report
