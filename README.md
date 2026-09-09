# real-robot-data-retime

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
