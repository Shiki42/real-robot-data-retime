# Accuracy-first foreground experiments

Accuracy and source-pixel correctness are the priority; latency is not a model
selection gate. The user confirmed that no human masks or contact/separation
annotations are available. All percentages below are explicitly named diagnostic
measures, not segmentation accuracy, precision/recall, or event-frame error.

The experiment branch incorporates committed baseline `d4d6519` in addition to
its earlier `9708716` experiment. Other ongoing uncommitted work in the main
Coder A checkout is separate. Code changes and inference run on Coder A; local
changes are Git synchronization only.

## Evidence and remaining failures

Three videos were processed at every source frame (30 FPS), not stride 3/6:

| Input | Frames | Analyzed size | Initial RobotSeg motion-coverage gate |
|---|---:|---|---|
| workpiece_0 | 1092 | 640x480 | Pass |
| workpiece_1 | 976 | 640x480 | Pass |
| drawer_0 | 634 | 640x362 | Fail |

These are the existing sample videos. Workpiece samples are at their source
camera dimensions; the drawer sample is downscaled from the 848x480 dataset
camera. Cropping increases model detail within those available pixels; it does
not restore detail lost in the source sample encoding/downsampling.

The strongest improvement observed is targeted arm resegmentation in drawer_0.
The original video tracker loses the partially visible left gripper after it
returns to the image edge. At source frames 500 and 550, a new current-frame
proposal recovers visible gripper foreground that was absent in the tracked mask.
The final version requires agreement between a full-image prediction and a
magnified crop prediction before adding pixels, reducing the speckled additions
seen in the initial crop-only trial.

| Drawer diagnostic | Original | Crop/full-image agreement proposal |
|---|---:|---:|
| Left insufficient-motion-coverage fraction | 48.73% | 5.41% |
| Left longest insufficient run | 147 frames | 4 frames |
| Right insufficient-motion-coverage fraction | 11.76% | 9.58% |
| Right longest insufficient run | 18 frames | 17 frames |

223 weak frame/arm cases were inspected; 149 resegmentation proposals were
selected. **The right arm still fails the existing longest-run gate, and the
complete video remains unvalidated.** The coverage thresholds were not relaxed.
The proxy favors covering observed motion and does not establish correct
boundaries or freedom from false positives. Visual inspection remains necessary.

## Local gripper segmentation and bounded gap proposals

`accuracy_experiment` reuses the whole-arm identity and geodesic end-effector
locator to crop a neighborhood around each arm's distal region. It runs the
RobotSeg `gripper` semantic decoder independently in each crop, avoiding
propagation from an empty first frame and excluding distant background chairs.
The outputs retain both raw semantic masks and support-filtered masks.

Support filtering permits only a small neighborhood around the owning arm
(approximately six pixels at width 640), rejects unsupported components and
leaves overlap with the other arm unassigned. It does not invent invisible
coordinates. Too-small arm support or unresolved end-effectors remain unknown.

| Input | Eligible frame/arm observations | Direct masks available | Additional agreeing gap proposals |
|---|---:|---:|---:|
| workpiece_0 | 967 | 774 | 26 |
| workpiece_1 | 825 | 682 | 11 |
| drawer_0 | 1055 | 974 | 5 |

Counts are per frame **and arm**, not distinct video frames or correct detections.
Local crops remove the previous tabletop-workpiece gripper false positive at
workpiece_0 frame 198, but direct segmentation also misses the real blurred
gripper there. This is not counted as an accuracy improvement.

For missing intervals of at most one second, an optional second stage uses
nearby direct masks of at least 32 pixels as seeds, up to half a second before
and after the interval. Positive points come from those automatic masks;
negative points come from the remaining arm region. Only the intersection of
forward/backward masks with IoU >= 0.6 and at least eight pixels becomes a new
proposal. Direct masks, their availability flags and centers are preserved.
Proposals cannot be recursively reused as direct seed observations.

In the final workpiece_0 frame-198 repair, the two directions have IoU about
0.193 and disagree too much; no repair is accepted. This known failure is kept.

`candidate_robot_masks` is the union of the original arm masks and supported
local/agreed gripper proposals. It never deletes existing arm pixels, but this
**does not guarantee improvement**: inspection of maximum-change examples
(workpiece_0 frame 377, workpiece_1 frame 809, drawer_0 frame 75) shows changes
around carried objects, reflective boundaries and possible background growth.
These candidates are for review, not automatic promotion to dataset rendering.

## Point identity under partial occlusion

Two workpiece videos supplied eight automatically detected origin-departure
windows. Every frame in each window is used. The original contact/occlusion seed
policy was extracted into `tracking_seed_frame` and reused by the SAM object
tracker and point experiment without changing the SAM policy.

The point experiment registers persistent background features, seeds before
image-space contact/occlusion evidence, verifies origin appearance and excludes
robot-covered pixels from the query mask. The final run uses up to 24 visible
interior points per object, comparing BootsTAPIR at 256 and 512 internal
resolution. Query points use registered `(t,y,x)` coordinates; measured tracks
are restored to each original source frame. Raw observations remain available.

A spatial agreement diagnostic requires at least three points and 60% of the
**currently visible** points in a bounded cluster. Requiring half of the initial
24 queries wrongly discards legitimate partial views when most object points
are occluded. That rejected formulation and earlier seed choices remain in the
remote experiment history. The final diagnostic preserves a consistent visible
subset without filling hidden points or extrapolating through missing frames.

This checks point-cloud agreement, not identity truth: a coherent group can
still follow the wrong object or a gripper. At workpiece_0 frame 198, a few
512-resolution points follow the visible screw shaft while other parts are
occluded. The 256-resolution model reports additional points near the gripper.
At deposition and motion-blurred frames, both resolutions can lose evidence.
512 resolution has lower visible/cluster coverage in these trials; that alone
proves neither better nor worse accuracy. No contact or separation frame is
certified from these point measurements, and the windows do not constitute a
complete interaction timeline.

## Reproduce on the experiment branch

Use the runtime and pinned upstream repositories in
[the initial experiment setup](model-experiments.md). In particular, put this
branch's `src`, RobotSeg and TAPNet roots on `PYTHONPATH`. Set Torch/OpenCV CPU
threads to four (the CLIs do this) and optionally set
`OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4`.
All output directories must be new. Inference inputs contain no human clicks,
boxes, object identities, event labels or robot telemetry.

```bash
# Whole-arm measurements at full frame rate.
python -m real_robot_data_retime.model_experiment \
  --input /path/to/workpiece_0.mp4 --output /path/to/workpiece_0-arms \
  --backend robotseg-prompted --task workpiece --stop 1092 --stride 1 \
  --checkpoint /path/to/robotseg.pt --upstream-repo /path/to/RobotSeg

# Direct local semantic observations.
python -m real_robot_data_retime.accuracy_experiment \
  --input /path/to/workpiece_0.mp4 --arms /path/to/workpiece_0-arms \
  --checkpoint /path/to/robotseg.pt --output /path/to/direct-grippers

# Reuse direct evidence with checked video/arm/checkpoint identities.
python -m real_robot_data_retime.accuracy_experiment \
  --input /path/to/workpiece_0.mp4 --arms /path/to/workpiece_0-arms \
  --checkpoint /path/to/robotseg.pt --single-frame /path/to/direct-grippers \
  --output /path/to/agreement-grippers

# Source-aligned, contact-candidate-adjacent point tracking.
python scripts/point_accuracy.py \
  --input /path/to/workpiece_0.mp4 --arms /path/to/workpiece_0-arms \
  --checkpoint /path/to/bootstapir_checkpoint_v2.pt --task workpiece \
  --output /path/to/point-review

# Target weak whole-arm masks with corroborated current-frame resegmentation.
python scripts/reseed_arm_accuracy.py \
  --input /path/to/drawer_0.mp4 --arms /path/to/drawer_0-arms \
  --checkpoint /path/to/robotseg.pt --output /path/to/reseed-review
```

The gripper CLI requires full-rate source-aligned arm measurements and checks
video identity, dimensions and frame mapping. A report is finalized only after
preview encoding. The first direct-gripper run exposed NumPy integer crop
coordinates in JSON; this was fixed, regression-tested, and rerun into `-v2`
outputs. Its incomplete receipt is not treated as a successful run.

Raw arrays, videos and full reports are at
`/home/coder/share/retime-accuracy-20260910` on Coder A:

- `*-arms`: full-rate masks and context overlays;
- `*-grippers-v2`: completed direct observations;
- `*-grippers-agreement-v2`: reliable-seed bidirectional proposals, preserving direct data;
- `*-points-visible-seed`: final visible-point initialization and occlusion-aware diagnostics;
- `drawer_0-reseed-agreement`: final crop/full-image resegmentation comparison.

No result is labelled validated for compositing. The experiment still needs
boundary/identity validation, and nothing here establishes that a release occurs
at a brief pause or inside an unobservable interval. The lack of human truth is
not treated as a reason to stop inspecting failures, nor as permission to report
diagnostic agreement as accuracy.

Machine-readable observations and report hashes: [evidence receipt](accuracy-experiments-20260910.json).

Validation: 100 tests passed, 5 skipped. New/experimental Python files pass Ruff;
the two existing seed-policy modules also pass with their pre-existing C408 style
rule excluded. Coordinate serialization, source identity, partial occlusion,
bidirectional disagreement, preservation of direct evidence and complementary
arm-mask additions have regression coverage.
