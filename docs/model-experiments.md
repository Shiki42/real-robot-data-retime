# RobotSeg / BootsTAPIR experiment

Branch: `shuyuan/robotseg-tapir-experiment`, based on `c33dca7`.
The original Coder A checkout had separate uncommitted work at experiment start; this experiment uses
`/home/coder/share/real-robot-data-retime-robotseg` so that work is preserved.

The actual baseline is SAM2.1, not SAM3. CoTracker3 is already used for drawer
points; small-object identities currently depend primarily on SAM masks.
These experiments do not replace the production interaction/compositing path.
The CLI always records `validated_for_compositing: false`.

## Upstream implementations

- [RobotSeg](https://github.com/showlab/RobotSeg), source
  `dafb8c0d507276e2f96d2b07ac3661a7b3a41a5f`: SAM2-derived tiny backbone,
  automatic robot/arm/gripper semantic prompts. Automatic semantic output does
  not establish left/right instance identity. Components connected to exactly
  one entry edge are assigned to that side; components touching both edges or
  neither edge remain explicitly unassigned. Missing wrist components are not
  silently painted into either arm. This conservative assignment is an
  experiment, not a complete instance-tracking solution.
- [TAPIR / RoboTAP](https://github.com/google-deepmind/tapnet), source
  `c2cbab81cc06092b5f05bfe2da7bfec54e2079c9`: official PyTorch BootsTAPIR,
  `bootstapir_checkpoint_v2.pt`, HF `google/tapnet` revision
  `5d3fb48e76c5422841e38501514121e251beabb7`. RoboTAP uses TAPIR trajectories;
  it is not another drop-in foreground-mask model. The adapter preserves
  `(t,y,x)` query versus `(x,y)` output ordering, aspect-ratio coordinate mapping,
  BGR-to-RGB conversion, and the official occlusion/expected-distance visibility
  rule. Unobserved points become NaN. Both trackers receive identical automatic
  interior points with persistent object IDs.
- [EgoLoc](https://github.com/IRMVLab/EgoLoc), inspected source
  `94e7941c427c284d4d6099a8b2e4adeaa00bc9ad`: its 2D demo uses human-hand
  GroundingDINO/SAM and GPT-4o API calls. The 3D path adds hand reconstruction and
  depth processing. It is not a self-contained lightweight robot release model.
  No API requests or video uploads were made. Adapting contact/separation
  reasoning would require robot-specific prompts, an explicit VLM runtime and
  evaluation against independent event evidence. Do not interpret a brief pause
  or an occluded predicted point as confirmed separation.

## Runtime and setup

Experiments run on Coder A's shared RTX 5090, Python 3.11.15,
Torch 2.8.0+cu128. A separate venv reads the existing runtime's site-packages
and adds dependencies without changing the active pipeline environment.
All upstream source clones, weights, logs and previews remain outside git at
`/home/coder/share/retime-model-experiments-20260910`.

With the project's interaction/segmentation dependencies already installed,
clone the upstream repositories at the revisions above and put their roots on
`PYTHONPATH`. The PyTorch TAPIR path needs `dm-tree==0.1.10` and `einshape==1.0`;
it does not require installing the upstream JAX training stack.
RobotSeg additionally uses `hydra-core==1.3.6`, `omegaconf==2.3.1`,
`antlr4-python3-runtime==4.9.3`, `iopath==0.1.10`, `portalocker==4.3.0`,
`kornia==0.8.3`, and `kornia-rs==0.1.14`.

RobotSeg uses the public JPEG-directory loader at quality 100, CPU-offloaded
frames/state, and upstream BF16 autocast. Optional CUDA hole filling is explicitly
disabled (`fill_hole_area=0`); other upstream decoder settings are retained.
JPEG conversion and CPU transfers count toward reported model-stage time.
This differs from the upstream all-GPU demo and its advertised FPS.
Original decoded source pixels are used for the overlays.

The user supplied `robotseg.pt`; expected SHA-256:
`1384c38c4f313487a32503025d4566ac1d815f6be983199362fc4af19aaa06a2`.
It is loaded by upstream with `weights_only=True` and strict state-dict matching.

## Reproduce

From the experiment worktree, with upstream roots on `PYTHONPATH`:

```bash
python -m real_robot_data_retime.model_experiment \
  --input /path/to/workpiece_0.mp4 --output /path/to/new-tapir-run \
  --backend tapir --task workpiece --stop 180 --stride 3 \
  --points-per-object 16 --checkpoint /path/to/bootstapir_checkpoint_v2.pt \
  --upstream-repo /path/to/tapnet

python -m real_robot_data_retime.model_experiment \
  --input /path/to/workpiece_0.mp4 --output /path/to/new-cotracker-run \
  --backend cotracker --task workpiece --stop 180 --stride 3 \
  --points-per-object 16 --upstream-repo /path/to/co-tracker

python -m real_robot_data_retime.model_experiment \
  --input /path/to/workpiece_0.mp4 --output /path/to/new-robotseg-run \
  --backend robotseg --task workpiece --stop 1092 --stride 6 \
  --checkpoint /path/to/robotseg.pt --upstream-repo /path/to/RobotSeg

python -m real_robot_data_retime.model_experiment \
  --input /path/to/workpiece_0.mp4 --output /path/to/new-sam2-run \
  --backend sam2 --task workpiece --stop 1092 --stride 6
```

Output directories must be new. `report.json` is written only after inference
and preview encoding finish. Partial directories and full stderr identify
failures; no success report is fabricated. Outputs include bit-packed masks or
point tracks, original frame indices, an overlay MP4, model provenance, wall
time including model loading, GPU starting load and peak allocated tensor
memory. Hashes identify the input and checkpoint; newer runs also hash all
package sources to distinguish dirty experiment revisions.

## Point-tracking pilot evidence

Same workpiece episode 0, source frames [0,180), 640x480 decoding, every third
frame, 60 processed frames, 4 automatically discovered objects, 16 points/object.
Both point sets and frame mappings were checked for exact equality.
BootsTAPIR internally resizes to 256x256; CoTracker3 uses its own native model
resolution. This is a practical backend comparison, not a FLOP-matched test.

| Cached-weight run | Load + inference | Peak allocated tensor memory | Visible fraction |
|---|---:|---:|---:|
| CoTracker3 | 4.99 s | 5.39 GB | 81.17% |
| BootsTAPIR | 5.60 s | 1.50 GB | 82.53% |

TAPIR used about 72% less peak tensor memory but was about 12% slower in this
single shared-GPU run. The initial TAPIR run took 17.96 s including first-use
checkpoint resolution/setup; it is not the cached-weight comparison.
Visibility is model confidence, not tracking accuracy. Visual checks at source
frames 0, 90 and 177 show similar stationary-object tracks, with differing
occlusion decisions. They do not establish lower identity-switch rates or
accurate release frames.

An initial 1092-frame / stride-3 CoTracker run failed with CUDA OOM. The subsequent
TAPIR full-video attempt failed during initial CUDA synchronization, before
model inference, as other GPU jobs changed resource usage. Neither failure
provides a valid full-video speed/quality comparison. Their tracebacks are
retained as `cotracker-full.log` and `tapir-full.log` in the artifact directory.

Acceptance requires whole-arm and gripper coverage, stable left/right identity,
small-object identity through transport/occlusion/deposition, and correct
contact/separation intervals across held-out episodes of all tasks. Complete
foreground masks and existing compositor/event verification gates remain
necessary before replacing the default pipeline.

## RobotSeg whole-arm pilot

Same complete workpiece episode 0, source frames [0,1092), decoded at 640x480,
every sixth frame (182 model frames). The controlled runs use four Torch/OpenCV
threads and `OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4`.
The first unconstrained-thread short RobotSeg attempt was interrupted after
several minutes without producing a result; it is excluded from comparisons.

| Backend | Load + inference | Peak allocated tensor memory | Motion coverage |
|---|---:|---:|---|
| SAM2.1 large, existing automatic prompts | 41.66 s | 1.67 GB | Pass |
| RobotSeg, automatic semantic prompt at frame 0 | 10.95 s | 0.78 GB | Fail |
| RobotSeg, existing automatic instance prompts | 107.61 s | 0.78 GB | Pass |

The unprompted model locked onto the background chair when the opening frame
contained no robot and did not recover the entering arms. Its apparent speed is
not useful output. The `robotseg-prompted` backend instead reuses
`segment_robots` and its automatic entry-anchored prompts, forward/backward
propagation, identity checks and visibility-reentry retries. This is a deliberate
separate experimental choice; failures never switch backend silently.

Run it by replacing `--backend robotseg` with `--backend robotseg-prompted`
in the command above. Prompted RobotSeg and SAM2 both passed the existing
motion-coverage gate, but that gate is necessary rather than sufficient:
visual inspection at source frame 540 (arms absent) showed SAM2's left-arm mask
covering the left bin interior. Prompted RobotSeg removed that large false
positive, while leaving a small stray foreground component at the bottom edge.
At frames 150 and 870, both found the visible arm, with differing fine boundaries,
cable coverage, and whether held-object pixels were included. These masks are
not human-annotated ground truth and have not passed the full compositor audit.

RobotSeg's first prompted run used about 53% less peak tensor memory but took
about 2.6 times as long in this shared-GPU setting. It is not a demonstrated
speed upgrade. Additional setup profiling is recorded separately to distinguish
JPEG/state initialization from the complete measured stage. No advertised
paper FPS is substituted for these measured end-to-end stage timings.

The dedicated semantic probe selects the frame with the largest automatically
observed robot-motion region, then compares `robot`, `arm`, and `gripper` on up
to 32 subsequent sampled frames without clicks or boxes:

```bash
python scripts/robotseg_categories.py \
  --input /path/to/workpiece_0.mp4 --checkpoint /path/to/robotseg.pt \
  --output /path/to/new-category-probe
```

Its output is semantic foreground, not verified left/right gripper instances.
The three-column video makes wrist/gripper confusion directly reviewable.

The visible-pose semantic probe selected source frame 102 automatically. In its
32 sampled frames, the unprompted `robot` and `arm` modes still included the
background chair. `gripper` broadly localized the distal assembly and excluded
most of the arm, but leaked onto an unrelated tabletop workpiece at source frame
198 and had visibly incomplete boundaries. It does not by itself resolve
precise jaw aperture, left/right association, or separation timing.

A repeated prompted run took 24.73 s with the same approximately 0.78 GB peak
allocation and passed the motion-coverage gate again. Instrumentation recorded
0.25 s of JPEG encoding and 5.01 s initializing eight tracking states (including
reentry recovery); most elapsed time was elsewhere. The wide 24.73–107.61 s
range prevents a robust throughput claim on this shared GPU. Inspection of the
upstream Robot Prompt Generator shows nested per-memory/per-region clustering
with `.item()` and tensor-valued Python conditions, which introduces GPU/CPU
synchronization. That is a profiling lead, not a measured attribution of all
remaining time to that component.

The initial pilot runs did not fix upstream randomness. RobotSeg's farthest-point
sampling uses `torch.randint`. The final CLI now defaults to `--seed 0`, records
it, and seeds Python, NumPy and Torch before model loading. This fixes RNG inputs;
it does not claim bitwise deterministic CUDA kernels. Historical reports without
`random_seed` remain labelled as unseeded observations.

While this experiment was running, the main development checkout advanced to
`b8c98b6` (background-person audits and placed-object ownership). These isolated
comparisons remain against `c33dca7`, not a claim about the latest main's behavior.

## Final fixed-seed runs and decision

Final sequential arm runs used seed 0, four CPU threads and identical source
frames/prompt generation. Both passed motion coverage. SAM2 large took 130.89 s
and 1.672 GB peak allocated tensor memory; prompted RobotSeg took 289.84 s and
0.783 GB. RobotSeg preparation accounted for only 0.27 s JPEG writing and 5.42 s
initializing eight states. GPU utilization at start was 98% and 99%, respectively.
Together with the earlier runs, this supports a memory reduction, not a stable
latency improvement for this adapter on the shared machine.

The complete point-tracking comparison used all 1092 source frames at stride 3
(364 model frames), the same 64 points, four discovered objects, and seed 0.
Queries, memberships and original frame mappings match exactly.

| Full-video point backend | Load + inference | Preprocessing | Peak allocated tensor memory | Visible fraction |
|---|---:|---:|---:|---:|
| BootsTAPIR, 256x256 internal input | 3.11 s | 0.51 s | 12.92 GB | 51.27% |
| CoTracker3, native internal input, expandable allocator | 6.33 s | 1.38 s | 18.26 GB | 41.68% |

CoTracker failed first with a 3.81 GiB allocation request and 5.57 GiB reserved
but unallocated. A separate run with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` completed. The final CLI
records this and the selected thread environment variables. This is an explicit
runtime experiment, not an error-swallowing model fallback. It is a lower-cost
option to try before changing a tracker merely because of this OOM.

The successful runs started at different shared GPU loads (93% for TAPIR, 99%
for CoTracker) and use different internal resolutions, so these measurements
are practical observations rather than an isolated algorithm ranking. TAPIR's
full-video memory is much larger than its short-video memory. Do not extrapolate
short-clip peaks to arbitrary episode lengths. Visual inspection of TAPIR at
source frames 360, 720 and 1089 shows some transported-object observations but
also stray visible tracks on background/table pixels. Higher visible fraction
does not prove fewer identity switches. Existing geometric, appearance and
causal verification remains necessary.

Decision: keep the established default pipeline. Retain prompted RobotSeg as a
candidate for more mask-quality evaluation and BootsTAPIR as a candidate where
point-tracking memory/throughput matters. Neither has established correct events
and source-pixel compositing across all held-out tasks. EgoLoc was inspected but
not executed or integrated; its human-hand/VLM dependencies are not a justified
speed replacement here. No new retimed dataset or production deployment is
claimed by these experiments.

Validation: 85 tests passed, 4 skipped; changed Python files pass Ruff and the
patch passes `git diff --check`. The skips are retained, not counted as passes.
Machine-readable observations are in [the evidence receipt](model-experiments-20260910.json).
Full raw arrays, videos and logs remain under the Coder A artifact root above.
