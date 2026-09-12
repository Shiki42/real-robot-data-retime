import numpy as np
from real_robot_data_retime.interaction.cache import photometry_cache
from real_robot_data_retime.interaction import photometric_motion as module


def test_photometry_cache_preserves_arrays_and_invalidates_changed_pixels(
    tmp_path, monkeypatch
):
    calls = []

    def compute(frames):
        calls.append(1)
        support = frames[..., 0] > 0
        return module.PhotometricMotion(
            {"masks": frames[..., 0]},
            {"masks": frames[..., 0], "prompt_support": support},
            support,
            ~support,
            np.ones((len(frames), 3)),
        )

    monkeypatch.setattr(module, "photometric_motion", compute)
    frames = np.zeros((3, 4, 4, 3), np.uint8)
    first = photometry_cache(frames, tmp_path)
    second = photometry_cache(frames, tmp_path)
    assert len(calls) == 1
    np.testing.assert_array_equal(first.raw["masks"], second.raw["masks"])
    np.testing.assert_array_equal(first.support, second.support)
    assert second.discovery["prompt_support"] is second.support
    frames[0, 0, 0] = 1
    assert photometry_cache(frames, tmp_path).support[0, 0, 0]
    assert len(calls) == 2
