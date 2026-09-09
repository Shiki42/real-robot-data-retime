import json
from dataclasses import asdict, replace
from pathlib import Path
import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .discovery import (
    InteractionConfig,
    motion_and_grippers,
    task_object_proposals,
    object_proposals,
    track_candidates,
)
from .evidence import score_hypothesis, stable_runs
from scipy.ndimage import median_filter
from .video import read_video, write_video
from ..tasks import PROFILES
from .registration import stabilize
from .verification import (
    validate_origin_departure,
    pickup_interval,
    attachment_visibility,
)


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
    from .video import components

    wood = (
        (hsv[:, :, 0] > 5)
        & (hsv[:, :, 0] < 35)
        & (hsv[:, :, 1] > 50)
        & (hsv[:, :, 2] > 65)
    )
    wood[: int(h * 0.35)] = False
    bins = [
        stat
        for mask, stat, center in components(wood, int(h * w * 0.01))
        if stat[2] > w * 0.15 and (center[0] < w * 0.25 or center[0] > w * 0.75)
    ]
    return "workpiece" if len(bins) >= 2 else "letters"


def run(input_path, output_dir, task=None, config=InteractionConfig(), backend="sam2"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames, fps = read_video(input_path, width=424)
    frames, transforms, registration_confidence = stabilize(frames)
    task = task or discover_task(frames)
    profile = PROFILES[task]
    evidence = motion_and_grippers(frames)
    proposals = task_object_proposals(frames, task, config)
    retries = []
    contact_distances = None
    if backend == "sam2":
        from .neural_tracks import (
            segment_candidates,
            recover_candidates,
            object_gripper_distances,
            terminal_letter_recovery,
        )
        from ..segmentation.sam_backend import SamVideo

        config = replace(
            config, evidence_frames=max(config.evidence_frames, round(fps * 2))
        )
        sam = SamVideo("facebook/sam2.1-hiera-large")

        from .cache import gripper_cache

        grippers = gripper_cache(
            frames, evidence, sam, Path.home() / ".cache" / "real-robot-data-retime"
        )
        evidence["centers"] = grippers["centers"]
        evidence["apertures"] = grippers["apertures"]
        tracks = segment_candidates(frames, proposals, evidence, sam)
        tracks, retries = recover_candidates(frames, proposals, tracks, evidence, sam)
        if task == "letters":
            tracks, terminal_retries = terminal_letter_recovery(
                frames, proposals, tracks, sam
            )
            retries.extend(terminal_retries)
        contact_distances = object_gripper_distances(grippers["masks"], tracks)
        np.savez_compressed(
            output_dir / "segmentation.npz",
            frame_shape=frames.shape[1:3],
            grippers=np.packbits(grippers["masks"], axis=-1),
            robots=grippers["robot_masks"],
            objects=np.array([t["packed_masks"] for t in tracks]),
        )
    elif backend == "geometry":
        tracks = track_candidates(frames, proposals, evidence["centers"])
    else:
        raise ValueError(f"unknown tracking backend: {backend}")
    drawer_motion = None
    if task == "drawer" and backend == "sam2":
        from ..tracking.points import track_points
        from ..tasks.drawer_constraints import (
            discover_drawer_motion,
            drawer_area_motion,
        )

        handle_proposals = object_proposals(frames, "saturated", config)
        xy, visible, sampled_indices = track_points(frames, handle_proposals)
        np.savez_compressed(
            output_dir / "drawer_point_tracks.npz",
            xy=xy,
            visible=visible,
            source_indices=sampled_indices,
        )
        motion = discover_drawer_motion(
            xy.transpose(1, 0, 2),
            evidence["centers"][sampled_indices, 1],
            fps / 3,
            frames.shape[2],
        )
        if motion is None:
            drawer_motion, drawer_area = drawer_area_motion(
                frames, fps, evidence["centers"][:, 1]
            )
            np.save(output_dir / "drawer_area.npy", drawer_area)
        else:
            drawer_motion = dict(
                open_frame=int(sampled_indices[motion.open_frame]),
                pull_start=int(sampled_indices[motion.pull_start]),
                close_start=int(sampled_indices[motion.close_start]),
                confidence=motion.confidence,
                handle_candidate=motion.handle_candidate,
                reference_candidate=motion.reference_candidate,
            )
    candidates = []
    for side in range(2):
        aperture = evidence["apertures"][:, side]
        if not np.isfinite(aperture).any():
            continue
        valid = np.isfinite(aperture)
        aperture = median_filter(
            np.interp(np.arange(len(frames)), np.flatnonzero(valid), aperture[valid]),
            size=5,
        )
        threshold = max(1.0, float(np.ptp(aperture)) * 0.06)
        closure = aperture[:-8] - aperture[8:] > threshold
        onset = set(a + 4 for a, b in stable_runs(closure, 1))
        # Also propose nearby motion onsets when aperture is weakly observable.
        for track in tracks:
            speed = np.linalg.norm(np.diff(track["centers"], axis=0), axis=1)
            near = (
                np.linalg.norm(
                    track["centers"][:-1] - evidence["centers"][:-1, side], axis=1
                )
                < frames.shape[2] * 0.2
            )
            moving = (speed > frames.shape[2] * 0.0015) & near
            onset.update(int(t) for t in np.flatnonzero(moving)[::3])
        onset = sorted(onset)
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
                    contact_distance=None
                    if contact_distances is None
                    else contact_distances[k, :, side],
                )
                result["evidence_window_frames"] = config.evidence_frames
                if (
                    not result["accepted"]
                    and "insufficient_future_visibility" in result["rejection_reasons"]
                ):
                    expanded = replace(
                        config,
                        evidence_frames=max(config.evidence_frames, round(fps * 4)),
                    )
                    alternative = score_hypothesis(
                        track["centers"],
                        evidence["centers"][:, side],
                        aperture,
                        frame,
                        frames.shape[2],
                        expanded,
                        contact_distance=None
                        if contact_distances is None
                        else contact_distances[k, :, side],
                    )
                    alternative["evidence_window_frames"] = expanded.evidence_frames
                    if (
                        alternative["accepted"]
                        or alternative["score"] > result["score"]
                    ):
                        result = alternative
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
        side = ["left", "right"].index(c["robot_id"])
        verification = validate_origin_departure(
            frames,
            proposals[c["object_id"]],
            evidence["centers"][:, side],
            c["pickup_frame"],
            c["release_frame"],
        )
        c["origin_verification"] = verification
        if not verification["verified"]:
            c["accepted"] = False
            c["rejection_reasons"].append(verification["reason"])
            continue
        object_path = tracks[c["object_id"]]["centers"]
        interval = pickup_interval(
            object_path,
            evidence["centers"][:, side],
            proposals[c["object_id"]],
            c["pickup_frame"],
            fps,
        )
        c["pickup_evidence"] = interval
        c["pickup_frame"] = interval["pickup_frame"]
        c["grasp_start"] = min(c["grasp_start"], c["pickup_frame"])

        velocity = np.linalg.norm(np.diff(evidence["centers"][:, side], axis=0), axis=1)
        active = np.flatnonzero(velocity > frames.shape[2] * 0.002)
        preceding = active[active < c["grasp_start"]]
        following = active[active >= c["release_frame"]]
        c["approach_start"] = int(preceding[0]) if len(preceding) else c["grasp_start"]
        c["retract_end"] = (
            int(following[-1] + 1) if len(following) else c["release_frame"]
        )
        c["object_origin"] = proposals[c["object_id"]]["origin"].tolist()
        c["transport_start"] = c["pickup_frame"]
        c["grasp_frame"] = c["pickup_evidence"]["first_confirmed_attachment"]
        c["track_confidence"] = float(
            np.isfinite(object_path[c["pickup_frame"] : c["release_frame"]])
            .all(axis=1)
            .mean()
        )
        c["track_visibility_fraction"] = c["track_confidence"]
        if backend == "sam2":
            robot_visibility = np.unpackbits(
                grippers["robot_masks"][:, side], axis=-1, count=frames.shape[2]
            ).astype(bool)
            visibility = attachment_visibility(
                object_path,
                evidence["centers"][:, side],
                robot_visibility,
                c["pickup_frame"],
                c["release_frame"],
            )
            c["visibility_evidence"] = visibility
            c["track_confidence"] = visibility["explained_fraction"]
        selected.append(c)
    from ..timeline.hypotheses import choose_episodes

    selected = choose_episodes(selected, len(proposals), task, fps)
    from ..timeline.episode import assign_boundaries

    selected = assign_boundaries(selected, evidence["centers"], fps, frames.shape[2])
    timeline = {
        "task": task,
        "fps": fps,
        "source_frames": len(frames),
        "episodes": sorted(selected, key=lambda x: x["pickup_frame"]),
        "drawer_motion": drawer_motion,
    }
    arm_count = int(np.sum(np.isfinite(evidence["centers"]).all(axis=2).any(axis=0)))
    gates = dict(
        two_arms=arm_count == 2,
        all_objects_identified=len({x["object_id"] for x in selected})
        == profile["expected_objects"],
        persistent_tracks=bool(selected)
        and all(x["track_confidence"] >= 0.35 for x in selected),
        origin_departure=bool(selected)
        and all(x["origin_verification"]["verified"] for x in selected),
    )
    if task == "drawer":
        gates["drawer_open_close"] = (
            drawer_motion is not None and drawer_motion["confidence"] >= 0.5
        )
        gates["correct_roles"] = (
            len(selected) == 1 and selected[0]["robot_id"] == "left"
        )
    else:
        gates["balanced_arm_assignments"] = all(
            sum(x["robot_id"] == side for x in selected) == 2
            for side in ["left", "right"]
        )
    complete = all(gates.values())
    report = dict(
        success=complete,
        phase="interaction_understanding",
        backend=backend,
        task=task,
        num_robot_arms=arm_count,
        num_object_proposals=len(proposals),
        expected_objects=profile["expected_objects"],
        num_manipulation_episodes=len(selected),
        validation_gates=gates,
        retries=retries,
        config=asdict(config),
        mean_interaction_confidence=float(np.mean([x["score"] for x in selected]))
        if selected
        else 0.0,
        status="verified_interaction_evidence"
        if complete
        else "requires_automatic_recovery",
        weak_aperture_events=sum(
            not x["closure_observed"] or not x["release_opening_observed"]
            for x in selected
        ),
        limitations=[
            "confidence scores are heuristic, not calibrated probabilities",
            "pickup uncertainty is explicitly reported for occluded transitions",
        ],
    )
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
        ax.set_ylabel(["Left", "Right"][side] + " apparent spread (px)")
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
                f,
                f"{task} source {t} | {report['status']}",
                (8, 22),
                0,
                0.5,
                (0, 0, 255),
                1,
            )
            yield f

    for filename, mode in [
        ("gripper_tracks.mp4", "grippers"),
        ("interaction_candidates.mp4", "all"),
        ("selected_object_candidates.mp4", "selected"),
    ]:
        write_video(output_dir / filename, annotated(mode), fps)
    return report
