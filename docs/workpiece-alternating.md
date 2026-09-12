# Alternating workpiece pickups with synchronous starts

Both arms start their recorded approaches on the first output interval. The
right preparation runs concurrently with left 1; it is not cropped and is not
prepended while the left arm is held still. Default onset delays are zero.

The priority contract is:

1. Left 1 cannot stop or slow because of either right action.
2. Left 2 can wait before admission for right 1.
3. Once admitted, left 2 continues through completion; right 2 yields as needed.
   Admission order is left 1, right 1, left 2, right 2.

The verified source interactions define pickup, release and approach boundaries.
First-action protection extends through release; the following recorded approach
belongs to the next action. Independent clocks brake only when approaching a
staging hold and restart before admission. After restarting, an admitted action
advances exactly one source frame per output frame through its protected end.
The paired-clock smoother is not used, so waiting by one arm cannot slow the other.

The planner includes the entire first right approach when searching the paired
path. It searches nearer staging poses first, backs off when required, and checks
every swept edge. Only staging points and completed trajectories permit holds;
a nonzero initial right approach cannot be frozen at its starting pose. There is
no initial left hold, preparation prefix, or prepositioned right starting frame.
Explicit user-requested onset delays are reported separately.

Braking normally uses 0.5 seconds and restart uses 0.3 seconds. If the first
approach has less source distance than a full brake requires, its brake duration
is shortened to twice that distance in frame intervals. This preserves an initial
source speed of 1x and a smooth stop without stretching the entire short approach.
For Episode 2, the three-frame approach uses a six-interval (0.20 s) brake rather
than the previous fifteen-interval brake. Left 1 remains unchanged.

`plan.stages.start_policy` is `synchronous_original_approaches`.
`initial_source_frames` identifies the original poses. `right_preparation` records
the source interval and its actual output start/end, which overlap left 1.
`onset_delay_frames` contains only explicitly requested delays. Waiting poses,
admission gates, protected intervals and search attempts are also recorded.

The final output must pass original-start, immediate-approach-onset, pickup-order,
uninterrupted-execution and swept projected-clearance checks. These are main-view
image-space checks, not a calibrated robot-control collision guarantee. Missing
interactions or an infeasible safe schedule fail explicitly.

## Validation

- 162 tests passed, 4 optional tests skipped on Coder A.
- A slow-right scene verifies both original starts, left 1 at unchanged source
  speed, left 2 yielding before admission, and uninterrupted admitted motions.
- Tests reject either arm being frozen at startup and a cropped right approach.
- A short-approach test checks the six-interval brake, source continuity, maximum
  clock rate and preserved subsequent source speed.
- Object-ownership regressions cover neighboring-object drift before pickup,
  after release and in disconnected carried-mask components.

All three pilot plans start at source pairs (63, 547), (31, 532), and (50, 674).
Both source clocks advance on the first output interval. Left 1 matches its
original source clock exactly, including while right 1 approaches and waits.

Current artifacts are on Coder A under
`/home/coder/share/retime-workpiece-synchronous-20260911`.

All three regenerated clips passed the startup and uninterrupted-clock checks.
They have zero detected origin duplicates and zero moving-foreground overlap
pixels. Startup, waiting-pose, pickup and final-scene keyframes were visually
reviewed. The left source clock is identical to the earlier non-preemptive
reference, and the right clock is identical after its first waiting-pose arrival.
Only its original initial approach has been restored alongside left 1.

| Episode | Left / right source starts | Right wait arrival | Video duration |
| --- | --- | ---: | ---: |
| 0 | 63 / 547 | 1.40 s | 21.13 s |
| 1 | 31 / 532 | 1.23 s | 21.00 s |
| 2 | 50 / 674 | 0.20 s | 22.33 s |

[Synchronous-start verification receipt](workpiece-synchronous-verification.json).

## Superseded preview records

The [prepositioned preview receipt](workpiece-alternating-verification.json) and
[serial preparation receipt](workpiece-preparation-verification.json) are retained
as historical evidence. Neither describes the current startup policy. The current
implementation has removed the serial preparation prefix instead of keeping it
as an alternate mode.

## Shadow-connected object proposals

Episode 2 initially yielded only two proposals: a dark sleeve and an adjacent
bolt were joined by a brighter shadow into an oversized component, which the
existing object-size filter rejected. Oversized central components are now
examined across their observed brightness levels. Only a split with at least
two independently large dark cores is accepted. Nearest-core ownership retains
all original support pixels in disjoint object prompts; normal-sized components
remain unchanged. This does not lower the four-interaction verification gate.

The first-eight-frame proposal masks for Episodes 0 and 1 are pixel-identical
to the prior implementation. Episode 2 now produces four proposals. Regressions
cover shadow-connected objects, unchanged separated objects and a uniform large
component that must not be split without evidence.

## Stationary destination audit

Episode 0's original masks depict the withdrawing left arm, but the opaque-motion
reference also assigned the changed left bin to that arm. After the arm left,
newly deposited bolts and rim changes still formed an entry-connected component.
The full-arm audit therefore failed despite all four interactions being valid.

The audit now identifies stable destination appearance using the last five
source observations with less than 1% robot coverage of each bin. It reuses the
compositor's three-pixel bin rim and the existing robust local exposure fit.
Only pixels consistent with stable clear-bin observations are excluded, before
constructing motion components; true moving pixels and unsupported observations
retain their original evidence. Source robot masks are not changed by this fix.
Reference frame indices and excluded evidence counts are recorded in the audit.

## Foreground ownership

Before pickup, an untouched workpiece is limited to its independently observed
initial footprint. A later tracking hypothesis cannot paint a neighboring
workpiece over its correctly emptied origin. After release, the destination
bin scene owns the deposited object; its stale track is not painted again as
independent scene foreground. Moving, carried objects still use tracked masks. A disconnected carried-track
component that overlaps another independently observed stationary origin is
excluded; a component attached to the carrying arm is retained. The same
ownership helper is used by both scheduling and compositing.

## Reproduce

Use the pinned environment and source video corresponding to a successful
analysis checkpoint:

```bash
python main.py --input /path/to/workpiece.mp4 --task workpiece \
  --analysis-only --debug-dir /path/to/analysis
python scripts/render_interaction_checkpoint.py \
  --input /path/to/workpiece.mp4 --analysis /path/to/analysis \
  --output /path/to/new-render
```

`report.json` includes source hashes, verification results, smoothing transitions
and `plan.pickup_order`. `source_mapping.npz` contains both final source clocks.

## Wider waiting poses

Waiting candidates reserve an additional 2% of image width around the preceding
execution's swept foreground (13 pixels at 640-pixel analysis width), beyond the
existing two-pixel foreground dilation. This changes staging clearance only;
all moving transitions still pass the original swept-clearance checks. The
margin is recorded as `waiting_clearance_width_fraction`. Source paths are not
translated: an earlier recorded pose is selected when needed. A pose already
outside the expanded sweep may remain unchanged. Synchronous startup and
non-preemptive execution are preserved.

The 0.5 s braking and 0.3 s restart surround the waiting pose; the actual hold
has zero source-clock speed. A short first approach uses a shorter brake.
