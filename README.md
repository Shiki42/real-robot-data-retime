# real-robot-data-retime

`main` 的抽屉、字母、工件视频前景分割与双臂合成，统一采用
`shuyuan/robotseg-tapir-experiment` 中已验收的实现（`826b5a7`）。
默认入口为 `main.py`，批处理使用 `batch.py`；两者复用同一套代码。
旧候选路线保存在 [`from-scratch`](https://github.com/Shiki42/real-robot-data-retime/tree/from-scratch)，不再作为主分支的实现。
迁移范围与验证见[主路线迁移记录](docs/main-pipeline-migration.md)。

[拉抽屉冻结快照](modules/drawer_task/README.md)仅用于精确复现此前认可版本，
其文件保持不变，日常开发使用上面的统一入口。

Automatically identify dual-arm interactions in a video and edit sequential
manipulations into overlapping actions. Normal operation requires a video file;
robot, object, grasp and release prompts are generated from image evidence.

Supported scene profiles are cube-into-drawer, letter sorting and workpiece
storage. The drawer profile enforces opening before insertion and withdrawal
before closing. Letter and workpiece manipulations are independent, with
left-arm priority when collision constraints require waiting.

## Install

Python 3.11 and `ffmpeg` are required. The tested GPU runtime uses PyTorch
2.8.0, torchvision 0.23.0 and transformers 5.16.1. Model revisions are pinned
in `segmentation/model_versions.py`.

```bash
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[interaction,segmentation,collision,test]'
```

RoboVisualize is additionally required for joint-based dataset scheduling.
Its asset path and the robot URDF are explicit inputs; this repository includes
the provided PiperX URDF snapshot in `assets/`.

## Edit a video

```bash
python main.py --input input.mp4 --output parallel_actions.mp4
```

To inspect interaction understanding without rendering:

```bash
python main.py --input input.mp4 --analysis-only --debug-dir debug/input
```

To process a directory without interactive annotation:

```bash
python batch.py --input-dir videos --output-dir outputs
```

Diagnostics include gripper tracks, aperture plots, object candidates,
`interaction_timeline.json`, segmentation/track checkpoints and `report.json`.
Low-confidence episodes fail with a report rather than requesting clicks or
inventing missing observations. Restarting with the same debug directory reuses
compatible automatic measurements and verifies their source identity.

The video-only path checks projected silhouettes. The dataset path below uses
recorded joints and RoboVisualize mesh clearance. These verification scopes are
reported separately.

## Trim and retime a LeRobot v3 dataset

The dataset path supports the PiperX 14-value action/state layout: six joint
angles in degrees and one gripper aperture in millimetres per arm. Main RGB is
composited; each wrist view and its telemetry follow that arm's source clock.

Trim initial stillness and retain available terminal stillness:

```bash
real-robot-trim --dataset /path/to/raw --output /path/to/trimmed \
  --repo-id owner/trimmed --tail-seconds 2
```

Generate one synchronized output per trimmed episode:

```bash
PYTHONPATH=/path/to/robo-visualize/src python -m real_robot_data_retime.automatic_dataset \
  --source /path/to/trimmed --raw-source /path/to/raw \
  --output /path/to/retimed --work-dir /path/to/analysis \
  --urdf assets/piper_x_description.urdf \
  --mesh-root /path/to/robo-visualize/src/robo_visualize/arms/piperx/assets \
  --repo-id owner/retimed
```

`--episodes 0 1` processes selected episode indices without assembling global
metadata. Failed episodes do not receive completion receipts. Finalization
requires every source episode; publication additionally requires data and
visual validation.

Exact per-arm source indices, raw-episode offsets, synthetic-hold flags,
interaction evidence and collision ledgers are stored in output metadata.
Retiming adds 60 repeated boundary frames at 30 FPS as an explicitly labelled
synthetic terminal hold. Trimming itself does not fabricate unavailable frames.

See [pipeline and verification details](docs/automatic-interaction.md),
[timing-grid augmentation](docs/uniform-retiming.md), and the tests in `tests/`.

`timeline.scheduler.schedule_sources` supports left-priority waits and task
precedence gates. `collision.piperx.PiperXClearance` reuses RoboVisualize's FK
and meshes, using the recorded 0.49 m base spacing and this repository's URDF
snapshot. The collision integration test needs the RoboVisualize package on
`PYTHONPATH` and its assets directory in `ROBOVISUALIZE_ASSETS`.

Experimental RobotSeg and BootsTAPIR backends, measured pilot results and
reproduction commands are documented in [model experiments](docs/model-experiments.md).

Accuracy-first full-frame-rate probes and remaining failures are recorded in
[accuracy experiments](docs/accuracy-experiments.md).

Photometric-reference corrections and the source-verified parallel preview are
documented in [photometric verification](docs/photometric-verification.md).

Workpiece and letter scene-ownership refinements are documented in
[episode refinement](docs/object-episode-iterations.md).
