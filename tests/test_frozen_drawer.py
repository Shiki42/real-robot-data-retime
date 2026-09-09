"""The accepted task is isolated from changes to the main experiment route."""

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "modules/drawer_task"


def module_at(root):
    spec = importlib.util.spec_from_file_location(
        "frozen_drawer_entry", ROOT / "run.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = root
    return module


def test_drawer_snapshot_hashes_are_intact():
    module_at(ROOT).verify()


def test_drawer_refuses_modified_snapshot(tmp_path):
    root = tmp_path / "drawer_task"
    shutil.copytree(ROOT, root, ignore=shutil.ignore_patterns("__pycache__"))
    source = root / "render/src/real_robot_data_retime/compositing/layers.py"
    source.write_text(source.read_text() + "\n# changed\n")
    with pytest.raises(ValueError, match="frozen drawer file changed"):
        module_at(root).verify()


def test_drawer_refuses_untracked_python_in_snapshot(tmp_path):
    root = tmp_path / "drawer_task"
    shutil.copytree(ROOT, root, ignore=shutil.ignore_patterns("__pycache__"))
    (root / "analysis/src/extra.py").write_text("pass\n")
    with pytest.raises(ValueError, match="source inventory changed"):
        module_at(root).verify()


def test_drawer_rejects_other_task_checkpoint(tmp_path):
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    (analysis / "interaction_timeline.json").write_text('{"task":"letters"}')
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "run.py"),
            "render",
            "--input",
            "unused.mp4",
            "--analysis",
            str(analysis),
            "--output",
            str(tmp_path / "render"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "only accepts drawer checkpoints" in result.stderr
    assert not (tmp_path / "render").exists()
