# Alternating workpiece pickups

Workpiece storage has two pickups per arm. The video pipeline enforces this
priority contract:

1. The first left execution cannot stop or slow because of either right action.
2. The second left execution can wait before admission for the first right pickup.
3. Once admitted, the second left execution continues through completion; the
   second right action waits as needed. Admission order is left 1, right 1,
   left 2, right 2.

The four verified source interactions define pickup, release and approach
boundaries. First-action protection extends through release; the following
recorded approach belongs to the next action. A held staging pose separates
each pair of actions. Independent source clocks use the existing 0.5 s braking
and 0.3 s restart curves only around these approach holds. The paired-clock
smoother is not used for workpiece video editing, so another arm's wait does not
slow the admitted arm. After its restart finishes, each admitted execution must
advance exactly one source frame per output frame through its protected end.

Staging candidates are checked against the preceding manipulation's carried
foreground sweep. The planner searches nearer poses first and backs off only
when no complete uninterrupted paired path exists. It checks every swept edge,
not just the waiting poses. The right arm starts at a recorded staging pose;
its omitted pre-grasp preparation frames are reported explicitly. Four complete
pickup/transport/place motions remain in source order. Waiting locations, entry
gates, protected intervals and search attempts are recorded under `plan.stages`.

The final clocks must pass pickup precedence, uninterrupted-execution and swept
projected-clearance checks. This is image-space video validation, not a
calibrated robot-control collision guarantee. Missing interactions fail explicitly.

## Validation

- 158 tests passed, 4 optional tests skipped on Coder A.
- A slow-right shared-space regression checks that left 1 retains its original
  clock, left 2 can wait before entry, and admitted left/right motions cannot stop.
- Independent ramp tests verify source joins and maximum clock rate.
- Object-ownership regressions reject both pre-pickup neighboring-object drift
  and post-deposit tracks that would restore a taken workpiece at its old origin.

Private analysis, source clocks, renders and verification artifacts are under
`/home/coder/share/retime-workpiece-alternating-20260911` on Coder A.
`final_v2_episode_0`, `final_v2_episode_1` and `final_v2_episode_2` are the final exports;
previous candidate outputs are retained separately as diagnostics.

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
