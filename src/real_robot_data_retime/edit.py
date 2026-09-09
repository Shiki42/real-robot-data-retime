"""Video-only entry point and shared dataset rendering orchestration."""

import json
from pathlib import Path
import cv2
import numpy as np
from .interaction.pipeline import run
from .interaction.video import read_video
from .compositing.layers import composite
from .timeline.visual import plan_visual


def registered_frames(input_path, transforms, width):
    frames, fps = read_video(input_path, width=width)
    h, w = frames.shape[1:3]
    if len(frames) != len(transforms):
        raise ValueError("analysis and input frame counts differ")
    return np.array(
        [
            cv2.warpAffine(f, m[:2], (w, h), borderMode=cv2.BORDER_REFLECT)
            for f, m in zip(frames, transforms)
        ]
    ), fps


def edit_video(
    input_path, output_path, debug_dir, task=None, *, backend="sam2", analysis_width=640
):
    debug = Path(debug_dir)
    report = run(
        input_path, debug, task, backend=backend, analysis_width=analysis_width
    )
    if not report["success"]:
        return report
    if backend != "sam2":
        raise ValueError("compositing requires tracked robot and object segmentation")
    timeline = json.loads((debug / "interaction_timeline.json").read_text())
    with (
        np.load(debug / "tracks.npz") as tracks,
        np.load(debug / "segmentation.npz") as segmentation,
    ):
        frames, fps = registered_frames(
            input_path, tracks["registration"], analysis_width
        )
        left, right, plan = plan_visual(timeline, frames, tracks, segmentation)
        np.savez_compressed(debug / "source_mapping.npz", left=left, right=right)
        native, masks, _ = native_render_inputs(
            input_path, tracks["registration"], segmentation
        )
        del frames
        render = composite(native, timeline, masks, left, right, output_path, debug)
    report.update(phase="parallel_compositing", schedule=plan, compositing=render)
    report["status"] = "rendered_pending_visual_validation"
    (debug / "report.json").write_text(json.dumps(report, indent=2))
    return report


def native_render_inputs(input_path, transforms, segmentation):
    cap = cv2.VideoCapture(str(input_path))
    native_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()
    if native_width < 2 or native_width % 2:
        raise ValueError("expected an even native video width")
    old_h, old_w = map(int, segmentation["frame_shape"])
    scale = native_width / old_w
    matrix = np.diag([scale, scale, 1.0])
    native_transforms = np.array(
        [matrix @ m @ np.linalg.inv(matrix) for m in transforms]
    )
    frames, fps = registered_frames(input_path, native_transforms, native_width)
    h, w = frames.shape[1:3]
    masks = dict(frame_shape=np.array([h, w]))
    for key in ["robots", "objects"]:
        values = segmentation[key]
        shape = values.shape[:-2]
        flat = values.reshape((-1, old_h, (old_w + 7) // 8))
        output = np.empty((len(flat), h, (w + 7) // 8), np.uint8)
        for i, packed in enumerate(flat):
            mask = np.unpackbits(packed, axis=-1, count=old_w)
            output[i] = np.packbits(
                cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST), axis=-1
            )
        masks[key] = output.reshape((*shape, h, (w + 7) // 8))
    return frames, masks, native_transforms
