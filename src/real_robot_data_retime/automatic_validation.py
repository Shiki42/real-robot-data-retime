"""Audit retimed numeric source identity, wrist pixels, and metadata consistency."""

import json
from pathlib import Path
import cv2
import numpy as np
from .trim import source_episodes, read_episode
from .video import _SequentialVideoReader
from .timeline.smooth import sample_rows
import pyarrow as pa


def wrist_pixel_error(source, output, mapping, samples=None):
    original, edited = (
        _SequentialVideoReader(Path(source)),
        _SequentialVideoReader(Path(output)),
    )
    errors = []
    try:
        targets = (
            np.arange(len(mapping))
            if samples is None
            else np.unique(
                np.linspace(0, len(mapping) - 1, min(samples, len(mapping))).astype(int)
            )
        )
        for target in targets:
            expected = original.sample(float(mapping[target]))
            actual = edited.read_to(int(target))
            if expected.shape != actual.shape:
                raise ValueError("wrist video dimensions changed")
            errors.append(float(cv2.absdiff(expected, actual).mean()))
    finally:
        original.close()
        edited.close()
    if max(errors) > 6:
        raise ValueError(
            f"wrist pixels do not match mapped source: error {max(errors):.3f}"
        )
    return dict(samples=len(errors), maximum_mean_absolute_pixel_error=max(errors))


def validate(source, output):
    source, output = Path(source), Path(output)
    si = json.loads((source / "meta/info.json").read_text())
    oi = json.loads((output / "meta/info.json").read_text())
    original_rows, rows = source_episodes(source), source_episodes(output)
    if len(rows) != len(original_rows):
        raise ValueError("output does not contain every source episode")
    results = []
    total = 0
    for original_row, row in zip(original_rows, rows):
        result = validate_episode(
            source, output, si, oi, original_row, row, global_start=total
        )
        results.append(result)
        total += result["frames"]
    if total != oi["total_frames"]:
        raise ValueError("total frame metadata differs from episode sum")
    report = dict(passed=True, episodes=len(rows), frames=total, results=results)
    (output / "validation.json").write_text(json.dumps(report, indent=2))
    return report


def validate_episode(source, output, si, oi, original_row, row, *, global_start=0):
    episode = row["episode_index"]
    if episode != original_row["episode_index"]:
        raise ValueError("episode identities changed")
    original = read_episode(source, si, original_row)
    table = read_episode(output, oi, row)
    with np.load(
        output / f"meta/retime_source_indices/episode_{episode:03d}.npz"
    ) as archive:
        maps = {key: archive[key] for key in archive.files}
    left, right = maps["left"], maps["right"]
    n = len(left)
    if n != len(table) or right.shape != left.shape:
        raise ValueError("source map and output lengths differ")
    for mapping in [left, right]:
        if (
            mapping[0] != 0
            or mapping[-1] != len(original) - 1
            or mapping.min() < 0
            or mapping.max() >= len(original)
            or np.any(np.diff(mapping) < 0)
        ):
            raise ValueError("invalid or non-monotone source mapping")
    numeric_sources = {
        key: np.asarray(original[key].to_pylist())
        for key in ["action", "observation.state"]
    }
    for key in ["action", "observation.state"]:
        values = numeric_sources[key]
        actual = np.asarray(table[key].to_pylist())
        expected = np.c_[
            sample_rows(values[:, :7], left), sample_rows(values[:, 7:], right)
        ]
        expected = np.asarray(
            pa.array(
                expected.tolist(), type=original.schema.field(key).type
            ).to_pylist()
        )
        if not np.array_equal(actual, expected):
            raise ValueError(f"{episode}: {key} violates per-arm source mapping")
    if table["index"].to_pylist() != list(range(global_start, global_start + n)):
        raise ValueError("global data indices are not contiguous")
    if not np.allclose(
        table["timestamp"].to_numpy(), np.arange(n) / oi["fps"], atol=1e-5
    ):
        raise ValueError("timestamps do not follow output frame rate")
    if (
        table["retime.left_source_frame"].to_pylist() != left.tolist()
        or table["retime.right_source_frame"].to_pylist() != right.tolist()
    ):
        raise ValueError("embedded source frame columns differ from receipts")
    if "complementary_info.rgb_device_timestamp_ns.top" in table.column_names:
        raise ValueError("composite main view cannot have a single capture timestamp")
    for key in original.column_names:
        if key.startswith("complementary_info.left_") or key.endswith(".left_wrist"):
            mapping = left
        elif key.startswith("complementary_info.right_") or key.endswith(
            ".right_wrist"
        ):
            mapping = right
        elif key == "task_index":
            mapping = left
        else:
            continue
        actual = table[key].to_pylist()
        values = original[key].to_pylist()
        if actual != [values[int(t)] for t in mapping]:
            raise ValueError(
                f"{episode}: telemetry field {key} violates source identity"
            )
    synthetic = maps["synthetic_hold"]
    if table["retime.synthetic_hold"].to_pylist() != synthetic.tolist():
        raise ValueError("synthetic hold labels differ from receipts")
    count = int(synthetic.sum())
    if count != round(2 * oi["fps"]) or not synthetic[-count:].all():
        raise ValueError("synthetic terminal hold is not exactly two seconds")
    if len(np.unique(left[-count:])) != 1 or len(np.unique(right[-count:])) != 1:
        raise ValueError("terminal hold moves a source arm")
    camera_reports = {}
    for camera, feature in oi["features"].items():
        if feature["dtype"] != "video":
            continue
        file = output / oi["video_path"].format(
            video_key=camera, chunk_index=0, file_index=episode
        )
        cap = cv2.VideoCapture(str(file))
        observed = (
            int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            cap.get(cv2.CAP_PROP_FPS),
        )
        cap.release()
        if observed != (n, feature["shape"][0], feature["shape"][1], oi["fps"]):
            raise ValueError(f"{episode}: video metadata mismatch for {camera}")
        if camera != "observation.images.top":
            mapping = {
                "observation.images.left_wrist": left,
                "observation.images.right_wrist": right,
            }[camera]
            src = source / si["video_path"].format(
                video_key=camera, chunk_index=0, file_index=episode
            )
            camera_reports[camera] = wrist_pixel_error(src, file, mapping)
    receipt = json.loads(
        (output / f"meta/retime_receipts/episode_{episode:03d}.json").read_text()
    )
    for side, mapping in [("left", left), ("right", right)]:
        if not np.array_equal(maps[f"raw_{side}"], mapping + receipt["trim"]["start"]):
            raise ValueError("raw source indices disagree with the trim offset")
    dependencies = receipt["plan"]["dependencies"]
    if dependencies:
        if not np.all(
            (left <= dependencies["safe_wait_frame"])
            | (right >= dependencies["open_frame"])
        ):
            raise ValueError("cube insertion precedes drawer opening")
        if not np.all(
            (right < dependencies["close_start"])
            | (left >= dependencies["withdrawal_frame"])
        ):
            raise ValueError("drawer closing precedes left-arm withdrawal")
    tolerance = np.array([0.3] * 6 + [0.5])
    for side, mapping in enumerate([left, right]):
        for start, stop in zip(mapping[:-1], mapping[1:]):
            if stop - start <= 1:
                continue
            for key in ["action", "observation.state"]:
                values = numeric_sources[key][
                    int(np.floor(start)) : int(np.ceil(stop)) + 1,
                    side * 7 : side * 7 + 7,
                ]
                if np.any(np.ptp(values, axis=0) > tolerance + 1e-9):
                    raise ValueError(
                        "retiming skipped a meaningful pose or command change"
                    )
    interaction = receipt["interaction"]
    if dependencies and interaction["timeline"]["task"] == "drawer":
        event = interaction["timeline"]["episodes"][0]
        confirmation = event.get("release_confirmation_frame", event["release_frame"])
        if np.any((right >= dependencies["close_start"]) & (left <= confirmation)):
            raise ValueError("drawer closing precedes visual release confirmation")
        if (
            dependencies["wait_method"] == "held_pose_clear_of_future_drawer_sweep"
            and "held_wait_maximum_aperture_mm" in dependencies
        ):
            wait = dependencies["safe_wait_frame"]
            aperture = max(
                numeric_sources["action"][wait, 6],
                numeric_sources["observation.state"][wait, 6],
            )
            if aperture > dependencies["held_wait_maximum_aperture_mm"] + 1e-9:
                raise ValueError("held waiting pose commands an opening gripper")
    if not (
        interaction["report"]["success"]
        and all(interaction["report"]["validation_gates"].values())
        and interaction["robot_mask_audit"]["passed"]
    ):
        raise ValueError("missing verified interaction and whole-arm masks")
    if not receipt["plan"]["new_edges_collision_free"]:
        raise ValueError("missing new-edge collision audit")
    replay = receipt["plan"]["preserved_original_pair_edges"]
    if bool(replay) == receipt["plan"]["swept_edges_verified"]:
        raise ValueError("strict clearance flag disagrees with source-contact ledger")
    for edge in replay:
        k, t = edge["output_edge"], edge["source_edge"]
        if not (
            0 <= k < len(left) - 1
            and left[k] == right[k] == t
            and left[k + 1] == right[k + 1] == t + 1
        ):
            raise ValueError(
                "source-contact exception is not an exact original paired edge"
            )
    if (
        not receipt["compositing"]
        .get("automatic_origin_audit", {})
        .get("passed", False)
    ):
        raise ValueError("missing rendered object-origin verification")
    return dict(episode=episode, frames=n, wrists=camera_reports)


def validate_pending_episode(source, output, episode):
    """Audit a completed per-episode artifact before global metadata assembly."""
    source, output = Path(source), Path(output)
    info = json.loads((source / "meta/info.json").read_text())
    original = next(
        row for row in source_episodes(source) if row["episode_index"] == episode
    )
    receipt = json.loads(
        (output / f"meta/retime_receipts/episode_{episode:03d}.json").read_text()
    )
    row = dict(episode_index=episode, length=receipt["length"])
    row.update({"data/chunk_index": 0, "data/file_index": episode})
    return validate_episode(source, output, info, info, original, row)
