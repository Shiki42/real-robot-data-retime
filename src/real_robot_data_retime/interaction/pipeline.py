import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter

from ..tasks import PROFILES
from .discovery import (
    InteractionConfig,
    object_proposals,
    task_object_proposals,
    track_candidates,
)
from .evidence import score_hypothesis, stable_runs
from .photometric_motion import photometric_motion
from .registration import stabilize
from .verification import (
    attachment_visibility,
    pickup_interval,
    validate_origin_departure,
)
from .video import read_video, write_video


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


def run(
    input_path,
    output_dir,
    task=None,
    config=InteractionConfig(),
    backend="sam2",
    analysis_width=640,
    reuse_measurements=None,
    retry_objects=False,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (
        backend == "sam2"
        and reuse_measurements is None
        and (output_dir / "measurements.json").exists()
    ):
        reuse_measurements = output_dir
    started = time.monotonic()

    def progress(stage):
        state = dict(
            stage=stage,
            elapsed_seconds=time.monotonic() - started,
            input=str(input_path),
        )
        temporary = output_dir / "progress.tmp.json"
        temporary.write_text(json.dumps(state, indent=2))
        temporary.replace(output_dir / "progress.json")
        print(json.dumps(state), flush=True)

    progress("decode")
    frames, fps = read_video(input_path, width=analysis_width)
    scale = (analysis_width / 424) ** 2
    config = replace(
        config,
        minimum_object_area=round(config.minimum_object_area * scale),
        maximum_object_area=round(config.maximum_object_area * scale),
    )
    progress("background_registration")
    frames, transforms, registration_confidence = stabilize(frames)
    task = task or discover_task(frames)
    profile = PROFILES[task]
    progress("robot_discovery")
    photometry = photometric_motion(frames)
    evidence = photometry.discovery
    np.savez_compressed(
        output_dir / "motion_photometry.npz",
        coefficients=photometry.coefficients,
        raw_motion_pixels=(photometry.raw["masks"] > 0).sum(axis=(1, 2)),
        discovery_pixels=(evidence["masks"] > 0).sum(axis=(1, 2)),
        audit_support_pixels=photometry.support.sum(axis=(1, 2)),
        ambiguous_pixels=photometry.ambiguous.sum(axis=(1, 2)),
    )

    def audit_robots(robots):
        from .robot_discovery import robot_mask_audit

        support = photometry.support
        scene = []
        audited_geometry = evidence
        if task == "workpiece":
            from ..tasks.workpiece import bin_aware_audit_support

            motion, support, scene = bin_aware_audit_support(
                frames, robots, evidence["masks"], support
            )
            audited_geometry = dict(evidence, masks=motion)
        audit = robot_mask_audit(robots, audited_geometry, fps, pixel_support=support)
        if scene:
            audit["stationary_bin_references"] = scene
        return audit

    proposals = task_object_proposals(frames, task, config)
    retries = []
    contact_distances = None
    cached_points = None
    sam = None
    from .measurements import inputs_fingerprint, load_hypotheses, producer_fingerprint

    measurement_inputs = inputs_fingerprint(input_path, frames, proposals, transforms)
    measurement_producer = producer_fingerprint()
    if reuse_measurements is not None:
        stored = json.loads(
            (Path(reuse_measurements) / "measurements.json").read_text()
        )
        if stored["inputs"] != measurement_inputs:
            retries.append(
                dict(
                    kind="automatic_hypothesis_cache_invalidated",
                    reason="input_registration_or_proposals_changed",
                )
            )
            reuse_measurements = None
    if backend == "sam2":
        from ..segmentation.sam_backend import SamVideo
        from .neural_tracks import (
            object_gripper_distances,
            recover_candidates,
            segment_candidates,
            terminal_letter_recovery,
        )

        config = replace(
            config, evidence_frames=max(config.evidence_frames, round(fps * 2))
        )
        if reuse_measurements is not None:
            progress("reinterpret_automatic_hypotheses")
            grippers, tracks, cached_points, measurement_producer = load_hypotheses(
                reuse_measurements, measurement_inputs, proposals
            )
            from .neural_tracks import grippers_from_robots, segment_robots

            mask_audit = audit_robots(grippers["robot_masks"])
            failed_sides = [
                i for i, arm in enumerate(mask_audit["arms"]) if not arm["passed"]
            ]
            if failed_sides:
                progress("repair_incomplete_robot_masks")
                sam = SamVideo("facebook/sam2.1-hiera-large")
                repaired, seeds = segment_robots(
                    frames, evidence, sam, sides=failed_sides
                )
                robots = grippers["robot_masks"].copy()
                robots[:, failed_sides] = repaired[:, failed_sides]
                grippers = grippers_from_robots(robots, frames.shape[1:3])
                measurement_producer = dict(
                    parent=measurement_producer,
                    robot_repair=producer_fingerprint(),
                    repaired_sides=failed_sides,
                )
                retries.append(
                    dict(
                        kind="whole_arm_mask_repair",
                        prior_audit=mask_audit,
                        seeds=seeds,
                    )
                )
            evidence["centers"] = grippers["centers"]
            evidence["apertures"] = grippers["apertures"]
            retries.append(
                dict(
                    kind="reverified_automatic_measurement_hypotheses",
                    producer=measurement_producer,
                )
            )
        else:
            sam = SamVideo("facebook/sam2.1-hiera-large")

            from .cache import gripper_cache

            progress("robot_and_gripper_tracking")
            grippers = gripper_cache(
                frames, evidence, sam, Path.home() / ".cache" / "real-robot-data-retime"
            )
            evidence["centers"] = grippers["centers"]
            evidence["apertures"] = grippers["apertures"]
            progress("object_segmentation")
            if task == "workpiece":
                from .event_seeded_tracking import event_seeded_tracks

                robot_union = (
                    np.unpackbits(
                        grippers["robot_masks"], axis=-1, count=frames.shape[2]
                    )
                    .astype(bool)
                    .any(axis=1)
                )
                tracks, origin_seeds = event_seeded_tracks(
                    frames, proposals, robot_union, sam, fps
                )
                (output_dir / "origin_events.json").write_text(
                    json.dumps(origin_seeds, indent=2)
                )
            else:
                tracks = segment_candidates(frames, proposals, evidence, sam)
            progress("object_identity_recovery")
            tracks, retries = recover_candidates(
                frames, proposals, tracks, evidence, sam
            )
            if task == "letters":
                tracks, terminal_retries = terminal_letter_recovery(
                    frames, proposals, tracks, sam
                )
                retries.extend(terminal_retries)
        if retry_objects and reuse_measurements is not None:
            progress("retry_object_identity_from_cached_tracks")
            if sam is None:
                sam = SamVideo("facebook/sam2.1-hiera-large")
            tracks, object_retries = recover_candidates(
                frames, proposals, tracks, evidence, sam
            )
            retries.extend(object_retries)
            measurement_producer = dict(
                parent=measurement_producer, object_repair=producer_fingerprint()
            )
        if task == "letters":
            from .neural_tracks import grippers_from_robots, object_free_robot_masks

            # Whole-arm SAM can include a released letter and pull the inferred
            # fingertip back onto it. Independent letter tracks disambiguate the
            # two semantic classes; carried letters are restored by the compositor.
            robots = object_free_robot_masks(grippers["robot_masks"], tracks)
            if not np.array_equal(robots, grippers["robot_masks"]):
                measurement_producer = dict(
                    parent=measurement_producer,
                    object_exclusion=producer_fingerprint(),
                )
                grippers = grippers_from_robots(robots, frames.shape[1:3])
                evidence["centers"] = grippers["centers"]
                evidence["apertures"] = grippers["apertures"]
                retries.append(dict(kind="letter_pixels_excluded_from_robot_masks"))
        mask_audit = audit_robots(grippers["robot_masks"])
        contact_distances = object_gripper_distances(grippers["masks"], tracks)
        # A completion marker must never certify a partly replaced checkpoint.
        (output_dir / "measurements.json").unlink(missing_ok=True)
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
    from .gripper_state import aperture_states

    gripper_states = np.stack(
        [aperture_states(evidence["apertures"][:, side], fps) for side in [0, 1]],
        axis=1,
    )
    gripper_velocity = np.concatenate(
        [np.full((1, 2, 2), np.nan), np.diff(evidence["centers"], axis=0) * fps]
    )
    np.savez_compressed(
        output_dir / "tracks.npz",
        grippers=evidence["centers"],
        gripper_states=gripper_states,
        gripper_velocity_px_s=gripper_velocity,
        apertures=evidence["apertures"],
        objects=np.array([x["centers"] for x in tracks]),
        registration=transforms,
        registration_confidence=registration_confidence,
    )
    if backend == "sam2":
        points_path = output_dir / "drawer_point_tracks.npz"
        if cached_points is not None:
            np.savez_compressed(points_path, **cached_points)
        else:
            points_path.unlink(missing_ok=True)
        marker = output_dir / "measurements.json"
        temporary = marker.with_suffix(".tmp.json")
        temporary.write_text(
            json.dumps(
                dict(inputs=measurement_inputs, producer=measurement_producer), indent=2
            )
        )
        temporary.replace(marker)
        progress("automatic_measurements_checkpoint")
    if sam is not None:
        import gc

        import torch

        del sam
        gc.collect()
        torch.cuda.empty_cache()
    drawer_motion = None
    if task == "drawer" and backend == "sam2":
        from ..tasks.drawer_constraints import (
            discover_drawer_motion,
            drawer_area_motion,
        )
        from ..tracking.points import track_points

        handle_proposals = object_proposals(frames, "saturated", config)
        if cached_points is not None:
            xy, visible, sampled_indices = (
                cached_points[k] for k in ["xy", "visible", "source_indices"]
            )
        else:
            xy, visible, sampled_indices = track_points(frames, handle_proposals)
        temporary_points = output_dir / "drawer_point_tracks.tmp.npz"
        np.savez_compressed(
            temporary_points, xy=xy, visible=visible, source_indices=sampled_indices
        )
        temporary_points.replace(output_dir / "drawer_point_tracks.npz")
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
    progress("interaction_hypotheses")
    bins = []
    deposit_cache = {}
    if task == "workpiece" and backend == "sam2":
        from ..tasks.workpiece import bin_visit, discover_bins, verify_deposit

        bins = discover_bins(frames[0])
        all_robots = (
            np.unpackbits(grippers["robot_masks"], axis=-1, count=frames.shape[2])
            .astype(bool)
            .any(axis=1)
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
                if (
                    task == "workpiece"
                    and len(bins) == 2
                    and result["pickup_frame"] is not None
                ):
                    pickup = result["pickup_frame"]
                    visit = bin_visit(
                        evidence["centers"][:, side],
                        pickup,
                        min(len(frames), pickup + round(fps * 8)),
                        bins[side]["bbox"],
                        fps,
                    )
                    if visit is not None:
                        key = (
                            side,
                            pickup,
                            visit["entry_frame"],
                            visit["clearance_frame"],
                        )
                        if key not in deposit_cache:
                            deposit_cache[key] = verify_deposit(
                                frames,
                                all_robots,
                                pickup,
                                visit,
                                bins[side]["bbox"],
                                fps,
                            )
                        deposit = {
                            **deposit_cache[key],
                            **visit,
                            "method": "occluded_bin_deposition",
                        }
                        updated = score_hypothesis(
                            track["centers"],
                            evidence["centers"][:, side],
                            aperture,
                            frame,
                            frames.shape[2],
                            replace(
                                config, evidence_frames=result["evidence_window_frames"]
                            ),
                            contact_distance=contact_distances[k, :, side],
                            release_evidence=deposit,
                        )
                        updated["evidence_window_frames"] = result[
                            "evidence_window_frames"
                        ]
                        result = updated
                    if (
                        result.get("release_evidence") is None
                        or not result["release_evidence"]["verified"]
                    ):
                        result["accepted"] = False
                        result["rejection_reasons"].append(
                            "destination_deposition_not_verified"
                        )
                history = track["centers"][
                    max(0, frame - config.evidence_frames) : frame
                ]
                observed = history[np.isfinite(history).all(axis=1)]
                origin_before = bool(
                    len(observed) >= 3
                    and np.linalg.norm(
                        np.median(observed, axis=0) - proposals[k]["origin"]
                    )
                    <= max(proposals[k]["bbox"][2:]) * 1.25
                )
                result["origin_observed_before_grasp"] = origin_before
                if not origin_before:
                    result["accepted"] = False
                    result["rejection_reasons"].append(
                        "candidate_not_at_origin_before_grasp"
                    )
                candidates.append(
                    dict(
                        robot_id=["left", "right"][side],
                        object_id=k,
                        grasp_start=frame,
                        **result,
                    )
                )
    if task == "drawer" and backend == "sam2":
        from .origin_identity import detached_origin

        origin_robots = np.unpackbits(
            grippers["robot_masks"][: max(3, round(fps * 0.5))],
            axis=-1,
            count=frames.shape[2],
        ).astype(bool)
        origin_objects = np.unpackbits(
            np.array(
                [track["packed_masks"][: max(3, round(fps * 0.5))] for track in tracks]
            ),
            axis=-1,
            count=frames.shape[2],
        ).astype(bool)
    selected = []
    visibility_masks = {}
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
            if side not in visibility_masks:
                visibility_masks[side] = np.unpackbits(
                    grippers["robot_masks"][:, side], axis=-1, count=frames.shape[2]
                ).astype(bool)
            robot_visibility = visibility_masks[side]
            visibility = attachment_visibility(
                object_path,
                evidence["centers"][:, side],
                robot_visibility,
                c["pickup_frame"],
                c["release_frame"],
                object_radius_px=float(
                    np.linalg.norm(proposals[c["object_id"]]["bbox"][2:4]) / 2
                ),
            )
            c["visibility_evidence"] = visibility
            c["track_confidence"] = visibility["evidence_confidence"]
        if task == "drawer" and backend == "sam2":
            identity = detached_origin(
                origin_robots, origin_objects, c["object_id"], c["pickup_frame"], fps
            )
            c["origin_identity"] = identity
            if not identity["verified"]:
                c["accepted"] = False
                c["rejection_reasons"].append("object_origin_is_robot_attached")
                continue
        selected.append(c)
    from ..timeline.hypotheses import choose_episodes

    selected = choose_episodes(selected, len(proposals), task, fps)
    from ..timeline.episode import assign_boundaries

    selected = assign_boundaries(selected, evidence["centers"], fps, frames.shape[2])
    if task == "drawer" and selected and backend == "sam2":
        release = max(c["release_frame"] for c in selected)

        def consistent(motion):
            return (
                motion is not None
                and motion["open_frame"] < release < motion["close_start"]
            )

        if not consistent(drawer_motion):
            alternative = discover_drawer_motion(
                xy.transpose(1, 0, 2),
                evidence["centers"][sampled_indices, 1],
                fps / 3,
                frames.shape[2],
                release_frame=int(np.searchsorted(sampled_indices, release)),
            )
            if alternative is not None:
                drawer_motion = dict(
                    open_frame=int(sampled_indices[alternative.open_frame]),
                    close_start=int(sampled_indices[alternative.close_start]),
                    pull_start=int(sampled_indices[alternative.pull_start]),
                    confidence=alternative.confidence,
                    handle_candidate=alternative.handle_candidate,
                    reference_candidate=alternative.reference_candidate,
                )
            else:
                alternative, drawer_area = drawer_area_motion(
                    frames, fps, evidence["centers"][:, 1]
                )
                if consistent(alternative):
                    drawer_motion = alternative
                    np.save(output_dir / "drawer_area.npy", drawer_area)
            retries.append(
                dict(kind="drawer_event_chronology", accepted=consistent(drawer_motion))
            )
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
    if backend == "sam2":
        gates["whole_robot_masks"] = mask_audit["passed"]
        (output_dir / "robot_mask_audit.json").write_text(
            json.dumps(mask_audit, indent=2)
        )
    if task == "drawer":
        gates["detached_object_origin"] = (
            bool(selected) and all(c["origin_identity"]["verified"] for c in selected)
            if backend == "sam2"
            else False
        )
        gates["drawer_open_close"] = (
            drawer_motion is not None and drawer_motion["confidence"] >= 0.5
        )
        gates["drawer_event_order"] = (
            bool(selected)
            and drawer_motion is not None
            and (
                drawer_motion["open_frame"]
                < selected[0]["release_frame"]
                < drawer_motion["close_start"]
            )
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
        motion_reference="photometric_opaque_support",
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

    progress("debug_rendering")
    for filename, mode in [
        ("gripper_tracks.mp4", "grippers"),
        ("interaction_candidates.mp4", "all"),
        ("selected_object_candidates.mp4", "selected"),
    ]:
        write_video(output_dir / filename, annotated(mode), fps)
    progress("complete")
    return report
