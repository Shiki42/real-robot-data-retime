"""Lossless AIST HDF5 reader with explicit nominal time and depth semantics.

No motion thresholds, unit conversions, arm-clock retiming or inferred success.
"""

from pathlib import Path

import cv2
import h5py
import numpy as np


class AistEpisode:
    def __init__(self, path, *, fps=None):
        self.path = Path(path)
        self._fps_profile = fps
        self.file = h5py.File(self.path, "r")
        try:
            self._validate()
        except Exception:
            self.file.close()
            raise

    def _validate(self):
        f = self.file
        if "frame_rate" in f.attrs:
            self.fps = float(f.attrs["frame_rate"])
            self.fps_origin = "hdf5_frame_rate_attribute"
            if self._fps_profile is not None and not np.isclose(
                float(self._fps_profile), self.fps
            ):
                raise ValueError("Explicit FPS profile conflicts with file metadata")
        elif self._fps_profile is not None:
            self.fps = float(self._fps_profile)
            self.fps_origin = "explicit_dataset_profile"
        else:
            raise ValueError(
                "AIST frame_rate attribute required; do not guess 30/50 Hz"
            )
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("frame_rate must be finite and positive")
        q, a = f["observations/qpos"], f["action"]
        if q.ndim != 2 or q.shape[1] != 14 or a.shape != q.shape or len(q) < 1:
            raise ValueError("Expected matching nonempty N x 14 state and action")
        self.frames = len(q)
        self.cameras = tuple(f["observations/images"].keys())
        for camera in self.cameras:
            if len(f["observations/images"][camera]) != self.frames:
                raise ValueError("Camera/control sample counts differ: " + camera)
        if "observations/qvel" in f and f["observations/qvel"].shape != q.shape:
            raise ValueError("Velocity/control shapes differ")

    def close(self):
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @property
    def state(self):
        return self.file["observations/qpos"][:]

    @property
    def action(self):
        return self.file["action"][:]

    @property
    def nominal_timestamps(self):
        """Control sample grid; never presented as camera exposure timestamps."""
        return np.arange(self.frames, dtype=np.float64) / self.fps

    def rgb(self, camera, frame):
        if camera not in self.cameras:
            raise KeyError(camera)
        if not 0 <= frame < self.frames:
            raise IndexError(frame)
        raw = np.asarray(self.file["observations/images"][camera][frame])
        if raw.ndim == 3 and raw.shape[2] == 3:
            # AIST compression convention uses OpenCV/BGR source storage.
            bgr = raw
        else:
            bgr = cv2.imdecode(raw.astype(np.uint8).reshape(-1), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Unable to decode " + camera + " frame " + str(frame))
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def raw_depth(self, camera, frame):
        if not 0 <= frame < self.frames:
            raise IndexError(frame)
        return self.file["observations/depth"][camera][frame]

    def metric_depth(self, camera, frame, *, meters_per_unit=None):
        if meters_per_unit is None:
            raise ValueError(
                "Metric depth scale is unverified; provide calibrated meters_per_unit"
            )
        if not np.isfinite(meters_per_unit) or meters_per_unit <= 0:
            raise ValueError("Invalid depth scale")
        return self.raw_depth(camera, frame).astype(np.float32) * meters_per_unit

    def receipt(self):
        depth = self.file.get("observations/depth")
        return {
            "source": str(self.path),
            "frames": self.frames,
            "nominal_fps": self.fps,
            "nominal_fps_origin": self.fps_origin,
            "timestamp_origin": "generated_from_nominal_fps",
            "camera_exposure_timestamps_verified": False,
            "state_action_units": "raw_source_units_no_conversion",
            "cameras": list(self.cameras),
            "depth": {}
            if depth is None
            else {
                k: {
                    "dtype": str(v.dtype),
                    "shape": list(v.shape),
                    "metric_scale_verified": False,
                }
                for k, v in depth.items()
            },
            "taxonomy_raw": str(self.file.attrs.get("taxonomy", "")),
            "task_name": str(self.file.attrs.get("Task name", "")),
        }
