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
before closing. Letter manipulations use left-arm priority. Workpiece storage
requires two pickups per arm, alternating left, right, left, right. The first
left execution cannot be stopped or slowed by the right arm. The second left
execution may wait before admission for the first right pickup, then runs
uninterrupted; the second right execution yields until it can enter safely.

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

All task planners apply smooth transitions at scheduled waits (0.5 s braking,
0.3 s restart), with post-retiming clearance checks. See
[general smooth scheduling](docs/general-smoothing.md) and the
[joint-guided drawer lift/wait/insert mode](docs/staged-drawer.md).

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

Workpiece video scheduling starts both arms at their original detected approach
poses on output frame zero. The right preparation is part of the same paired
schedule as left 1: there is no right-only lead-in and no omitted approach.
Only an arm approaching a wait brakes and restarts. The first left execution
retains source speed; admitted executions cannot be preempted by the other arm.
A very short initial approach uses a shorter brake instead of being stretched
into a slow approach that would force the left arm to yield. Explicit onset-delay
arguments remain opt-in; the default inserts no onset delay.

All pickup/transport/place frames are retained in source order. The final paired
path is checked for original starting poses, simultaneous approach onset, pickup
precedence, uninterrupted execution and swept projected foreground clearance.
Reports include `plan.pickup_order` and `plan.stages` with starting poses,
right preparation boundaries, waiting poses, protected intervals and staging
search evidence. See [workpiece priority and verification](docs/workpiece-alternating.md).

## Five-round screw insertion pilot

The [screw pilot](docs/screw-pilot/README.md) lets the later arm flow directly
into insertion while only the earlier arm waits or adjusts its approach speed.
The right arm must complete its original insertion and downward retreat as one
continuous task before its next pickup can be retimed. Left storage plus the
next pickup remains continuous. Videos and action/state share the existing
continuous source-clock interpolation. Reviewed boundaries and segmentation
prompts are explicit in the pilot config.
