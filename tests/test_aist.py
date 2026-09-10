import cv2
import h5py
import numpy as np
import pytest

from real_robot_data_retime.aist import AistEpisode


def sample(path):
    with h5py.File(path, "w") as f:
        f.attrs["frame_rate"] = 50
        f.create_dataset("action", data=np.arange(42, dtype=np.float32).reshape(3, 14))
        f.create_dataset("observations/qpos", data=np.zeros((3, 14), np.float32))
        f.create_dataset("observations/qvel", data=np.zeros((3, 14), np.float32))
        bgr = np.zeros((16, 20, 3), np.uint8)
        bgr[:, :, 2] = 255
        ok, jpg = cv2.imencode(".jpg", bgr)
        assert ok
        f.create_dataset(
            "observations/images/cam_high", data=np.tile(jpg.reshape(1, -1), (3, 1))
        )
        f.create_dataset(
            "observations/depth/dcam_high", data=np.ones((3, 16, 20), np.uint8)
        )


def test_decodes_rgb_without_changing_robot_signals(tmp_path):
    p = tmp_path / "ep.hdf5"
    sample(p)
    with AistEpisode(p) as e:
        np.testing.assert_array_equal(
            e.action, np.arange(42, dtype=np.float32).reshape(3, 14)
        )
        rgb = e.rgb("cam_high", 1)
        assert (
            rgb.shape == (16, 20, 3)
            and rgb[:, :, 0].mean() > 250
            and rgb[:, :, 2].mean() < 3
        )
        np.testing.assert_allclose(e.nominal_timestamps, [0, 0.02, 0.04])
        assert not e.receipt()["camera_exposure_timestamps_verified"]
        with pytest.raises(ValueError, match="unverified"):
            e.metric_depth("dcam_high", 0)
        np.testing.assert_allclose(
            e.metric_depth("dcam_high", 0, meters_per_unit=0.01), 0.01
        )


def test_rejects_missing_time_contract(tmp_path):
    p = tmp_path / "ep.hdf5"
    sample(p)
    with h5py.File(p, "a") as f:
        del f.attrs["frame_rate"]
    with pytest.raises(ValueError, match="frame_rate"):
        AistEpisode(p)


def test_rejects_camera_control_mismatch(tmp_path):
    p = tmp_path / "ep.hdf5"
    sample(p)
    with h5py.File(p, "a") as f:
        del f["observations/images/cam_high"]
        f.create_dataset(
            "observations/images/cam_high", data=np.zeros((2, 50), np.uint8)
        )
    with pytest.raises(ValueError, match="sample counts"):
        AistEpisode(p)


def test_explicit_fps_profile_is_auditable_and_cannot_override_metadata(tmp_path):
    p = tmp_path / "ep.hdf5"
    sample(p)
    with pytest.raises(ValueError, match="conflicts"):
        AistEpisode(p, fps=30)
    with h5py.File(p, "a") as f:
        del f.attrs["frame_rate"]
    with AistEpisode(p, fps=50) as e:
        assert e.receipt()["nominal_fps_origin"] == "explicit_dataset_profile"


def test_receipt_declares_nominal_timestamp_origin(tmp_path):

    path = tmp_path / "timestamps.hdf5"
    sample(path)
    with AistEpisode(path) as episode:
        assert episode.receipt()["timestamp_origin"] == "generated_from_nominal_fps"
