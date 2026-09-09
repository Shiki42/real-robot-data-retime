# Timing-grid augmentation

A small standalone toolkit for counterfactually retiming sequential dual-arm
real-robot demonstrations.

It contains only the functional code extracted from the validated PiperX
sort-letters retiming work:

- detect left-arm and right-arm work segments from robot state, with action
  motion as an independent sanity check;
- schedule the two original trajectories at uniformly sampled relative timings;
- guarantee that no new both-arms-idle gap is inserted;
- hold an arm's boundary pose and wrist frame while that arm is waiting;
- write per-frame left-idle, right-idle, and overlap labels;
- record exact per-arm execution, idle, and overlap intervals in frames and
  seconds;
- preserve each wrist camera's own source time;
- build a counterfactual main view using the left half from the left-arm source
  time and the right half from the right-arm source time;
- validate numeric mappings, timing labels, video frame counts, sampled pixels,
  and a complete SHA-256 inventory.

## Supported dataset layout

The end-to-end CLI currently targets a dual-arm LeRobot v3 shared-video layout:

- 14-D `action`: seven left-arm values followed by seven right-arm values;
- 14-D `observation.state` in the same order;
- `observation.images.left_wrist`;
- `observation.images.right_wrist`;
- `observation.images.top`;
- shared Parquet and MP4 files with episode ranges in
  `meta/episodes/chunk-000/file-000.parquet`.

The scheduling primitives in `real_robot_data_retime.retime` can also be used
directly.

## Installation

Python dependencies:

```bash
python -m pip install -e .
```

Video generation additionally requires `ffmpeg` on `PATH`.

## Uniform schedule

For (N) source episodes, the pipeline constructs four phase-shifted samples
per source:

```text
grid_size = 4 * N
grid_index = source_episode + N * sample_index
u = grid_index / grid_size
```

At `u=0`, the left arm finishes immediately before the right arm starts.
Increasing `u` advances the right arm relative to the left. The omitted
`u=1` endpoint would put the left arm immediately after the right arm.

Use `--grid-parity even` and `--grid-parity odd` to generate the two
alternating, complementary halves of the complete grid.

## Generate

The source directory must be a Git checkout at the exact revision passed on the
command line. The output directory must not exist.

```bash
real-robot-retime \
  --dataset /path/to/source-checkout \
  --source-repo owner/source-dataset \
  --source-revision 0123456789abcdef0123456789abcdef01234567 \
  --output /path/to/new-output \
  --repo-id owner/output-dataset \
  --grid-parity even
```

The output includes:

- `retime.left_idle`, `retime.right_idle`, and `retime.overlap` in the
  data Parquet;
- per-episode frame/second interval metadata;
- exact source-index and active-mask arrays under
  `meta/retime_source_indices/`;
- `retime_manifest.json` with source provenance, schedule grid, heuristic
  settings, and file inventory.

## Validate

```bash
real-robot-retime-validate \
  --source /path/to/source-checkout \
  --dataset /path/to/retimed-output \
  --report /path/to/retimed-output/VALIDATION_RECEIPT.json
```

Validation fails if any output frame has both arms idle, if a numeric or timing
mapping differs from its receipt, if video counts differ, or if sampled video
pixels exceed the configured re-encoding tolerance.

## Important semantic boundary

The split main view is a video-edit counterfactual, not a physically captured
simultaneous world observation. Retiming does not establish task success,
collision safety, or real simultaneous dual-arm behavior.

## Trim static episode boundaries (PiperX)

This migrates the static-edge analysis and RGB trimming pipeline into this package.
Supports LeRobot v3 per-episode or shared data/video files, with arbitrary tasks,
episode counts, and dataset FPS.

The fixed 14-D action layout is left six joints (degrees), left gripper (mm),
right six joints (degrees), right gripper (mm). Default strict thresholds are
0.1 degrees/second for joints and 0.1 mm/second for grippers. Speeds use adjacent
action differences times FPS. Static frames must pass both adjacent edges
(one edge at episode boundaries). No smoothing or internal static deletion.

- Head: remove all consecutive frames where both arms are static.
- Tail: retain 2 seconds after the later-stopping arm, keeping both arms and
  all cameras synchronized. Earlier-stopping arms may retain longer holds.
- Short tails: keep the original end without inventing/repeating frames; report shortfalls.
- Entirely static episodes: retain the final two seconds, or all available frames.
  With zero tail duration retain one frame to avoid empty episodes.
- Preserve action, state, other numeric fields, task IDs and task text.
  All RGB cameras use the same contiguous interval; omit depth features/storage.

Analyze only:

```bash
real-robot-trim --dataset /path/to/source --report /path/to/edges.json
```

Analyze and write a new RGB dataset:

```bash
real-robot-trim \
  --dataset /path/to/source \
  --output /path/to/new-rgb-dataset \
  --repo-id owner/new-rgb-dataset \
  --tail-seconds 2 \
  --joint-threshold 0.1 \
  --gripper-threshold 0.1
```

Output must not exist and must be outside the source. Nothing is uploaded.
Interrupted outputs are left for inspection; retry with a new path.
Outputs contain per-episode Parquet/videos with reset timestamps, updated
metadata, recomputed statistics, and trim_manifest.json documenting per-arm
static counts, source intervals [start, stop), removals, and hold shortfalls.
RGB statistics sample up to 100 uniform frames per episode resized to 64x64;
every generated video is decoded to verify its frame count.

The earlier one-off script defaulted to both action and measured state.
This tool explicitly uses action, with separate gripper units, and is not
intended to reproduce the old 1119-frame removal total. Trimming runs
independently of retiming; retiming's existing no-both-idle policy is unchanged.


## Original + right-then-left (bi-sequential)

Keep each original episode and append one generated episode at the exact `u=1`
endpoint. The original core numeric columns and RGB video bytes are preserved;
only the generated half uses motion-segment boundary holds. Original recorded
stillness is allowed. Thus episode count doubles but frame count need not double.
The training feature set matches Random Retime; acquisition diagnostics are
omitted and the single task text is normalized to `sort letters`.

```bash
python -m real_robot_data_retime.bi_sequential \
  --source /path/to/source-checkout --source-repo owner/source-dataset \
  --source-revision 0123456789abcdef0123456789abcdef01234567 \
  --output /path/to/bi-sequential --repo-id owner/bi-sequential
```

This command validates before producing a passing `VALIDATION_RECEIPT.json`.
It checks the right-first endpoint independently of the scheduler, all numeric
mappings, merged episode/data/video indices, original video hashes, recomputed
numeric statistics and sampled reverse video pixels including the arm handoff.
Generation also decodes every video to verify frame counts. The input must use
one shared LeRobot v3 data file and one shared video file per camera.

To repeat validation, pass `--source`, `--output`, and `--validate-only`.
No upload occurs automatically.
