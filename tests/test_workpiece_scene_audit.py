import numpy as np
from real_robot_data_retime.tasks import workpiece


def test_static_deposition_is_removed_before_motion_components(monkeypatch):
    n, h, w = 8, 100, 200
    monkeypatch.setattr(
        workpiece,
        "discover_bins",
        lambda frame: [dict(bbox=[0, 50, 40, 35]), dict(bbox=[160, 50, 40, 35])],
    )
    yy, xx = np.mgrid[:h, :w]
    reference = np.repeat(
        (100 + xx / 5 + yy / 8).astype("uint8")[:, :, None], 3, axis=2
    )
    reference[65:78, 15:22] = 20
    frames = np.repeat(reference[None], n, axis=0)
    frames[:2, 52:80, :12] = 0  # A real arm, missing from the first source mask.
    robots = np.zeros((n, 2, h, w), bool)
    robots[1, 0, 52:80, :12] = True
    motion = np.zeros((n, h, w), np.uint8)
    motion[:, 50:85, :40] = 1
    motion[:, 70, 100] = 1  # Non-bin evidence remains unchanged.
    support = motion > 0
    clean_motion, clean_support, records = workpiece.bin_aware_audit_support(
        frames, np.packbits(robots, axis=-1), motion, support
    )
    assert not clean_motion[2:, 50:85, :40].any()
    assert not clean_support[:, 65:78, 15:22].any()
    assert clean_motion[:2, 52:80, :12].all()
    assert clean_support[:2, 52:80, :12].all()
    assert clean_motion[:, 70, 100].all() and clean_support[:, 70, 100].all()
    assert motion[:, 50:85, :40].all() and support[:, 50:85, :40].all()
    assert records[0]["reference_frames"] == [3, 4, 5, 6, 7]


def test_bin_without_clear_references_retains_all_evidence(monkeypatch):
    monkeypatch.setattr(
        workpiece,
        "discover_bins",
        lambda frame: [dict(bbox=[0, 50, 40, 35]), dict(bbox=[160, 50, 40, 35])],
    )
    frames = np.full((4, 100, 200, 3), 100, np.uint8)
    robots = np.zeros((4, 2, 100, 25), np.uint8)
    motion = np.ones((4, 100, 200), np.uint8)
    support = motion.astype(bool)
    clean_motion, clean_support, records = workpiece.bin_aware_audit_support(
        frames, robots, motion, support
    )
    assert np.array_equal(clean_motion, motion) and np.array_equal(
        clean_support, support
    )
    assert all(not r["reference_frames"] for r in records)
