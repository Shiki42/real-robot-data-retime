import json

import pytest

from real_robot_data_retime.model_experiment import sha256
from scripts.render_interaction_checkpoint import render_checkpoint


def test_unfinished_analysis_cannot_render_an_old_success_report(tmp_path):
    source = tmp_path / "input.mp4"
    source.write_bytes(b"source identity")
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    (analysis / "progress.json").write_text(
        json.dumps({"stage": "automatic_measurements_checkpoint"})
    )
    (analysis / "report.json").write_text(json.dumps({"success": True}))
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="unfinished"):
        render_checkpoint(source, analysis, output)
    assert not output.exists()


def test_other_source_cannot_reuse_successful_analysis(tmp_path):
    source = tmp_path / "input.mp4"
    source.write_bytes(b"source identity")
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    (analysis / "progress.json").write_text(json.dumps({"stage": "complete"}))
    (analysis / "report.json").write_text(json.dumps({"success": True}))
    (analysis / "measurements.json").write_text(
        json.dumps({"inputs": {"video_sha256": "wrong source"}})
    )
    with pytest.raises(ValueError, match="another source"):
        render_checkpoint(source, analysis, tmp_path / "output")


def test_failed_analysis_cannot_be_promoted_to_compositing(tmp_path):
    source = tmp_path / "input.mp4"
    source.write_bytes(b"source identity")
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    (analysis / "progress.json").write_text(json.dumps({"stage": "complete"}))
    (analysis / "report.json").write_text(
        json.dumps({"success": False, "validation_gates": {"whole_robot_masks": False}})
    )
    (analysis / "measurements.json").write_text(
        json.dumps({"inputs": {"video_sha256": sha256(source)}})
    )
    with pytest.raises(ValueError, match="failed verification"):
        render_checkpoint(source, analysis, tmp_path / "output")


def test_render_cli_exits_nonzero_when_automatic_checks_fail(monkeypatch):
    import sys

    from scripts import render_interaction_checkpoint as renderer

    monkeypatch.setattr(
        sys, "argv", ["render", "--input", "i.mp4", "--analysis", "a", "--output", "o"]
    )
    monkeypatch.setattr(
        renderer,
        "render_checkpoint",
        lambda *a, **kw: {"automatic_checks_passed": False},
    )
    with pytest.raises(SystemExit) as error:
        renderer.main()
    assert error.value.code == 1


def test_changed_registration_cannot_render_under_old_manifest(monkeypatch, tmp_path):
    import numpy as np

    from real_robot_data_retime.interaction.measurements import inputs_fingerprint
    from scripts import render_interaction_checkpoint as renderer

    source = tmp_path / "input.mp4"
    source.write_bytes(b"source identity")
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    frames = np.zeros((2, 4, 6, 3), np.uint8)
    transforms = np.repeat(np.eye(3)[None], 2, axis=0)
    original = inputs_fingerprint(source, frames, [], transforms)
    (analysis / "progress.json").write_text(json.dumps({"stage": "complete"}))
    (analysis / "report.json").write_text(
        json.dumps({"success": True, "validation_gates": {"test": True}})
    )
    (analysis / "measurements.json").write_text(json.dumps({"inputs": original}))
    (analysis / "interaction_timeline.json").write_text("{}")
    transforms[1, 0, 2] = 2
    np.savez(analysis / "tracks.npz", registration=transforms)
    np.savez(analysis / "segmentation.npz", frame_shape=[4, 6])
    monkeypatch.setattr(renderer, "registered_frames", lambda *a: (frames, 30))
    with pytest.raises(ValueError, match="registration differs"):
        renderer.render_checkpoint(source, analysis, tmp_path / "output")
