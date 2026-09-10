# Main video pipeline authority

The user accepted the drawer, workpiece and letter previews and requested that
`main` use the implementation from `shuyuan/robotseg-tapir-experiment` (`826b5a7`).
The previous candidate was archived as `from-scratch` (`f10a5b4`).

## Replacement scope

The root interaction, segmentation, foreground compositing, background recovery,
task profiles, tracking, video editor and visual planner now match `826b5a7`
exactly. Conflicting candidate implementations were replaced, not combined with
new heuristic branches. The candidate-only `interaction/robot_recovery.py` and
its tests were deleted, along with the candidate drawer-release tests and the
superseded release implementation they exercised.

`main.py`, `batch.py` and `scripts/render_interaction_checkpoint.py` all use the
same root package. The automatic LeRobot exporter imports that same interaction
pipeline and compositor. There is no candidate backend selector or compatibility
wrapper. RobotSeg/TAPIR research scripts remain explicitly experimental; the
accepted default video route uses SAM2 and the verified photometric/ownership
refinements described in the episode records.

Independent trimming, uniform timing-grid augmentation, joint-space planning,
numeric dataset validation and held-grasp safeguards were preserved. These have
different responsibilities from foreground extraction. The dedicated numeric
retiming command is not an alternate semantic segmentation implementation.

`modules/drawer_task` is the separately requested immutable snapshot of the
accepted drawer, not the unsuccessful candidate. It and its manifest/tests were
kept unchanged for exact historical reproduction. It is not imported by the
normal video editor. Its README describes its original freeze context.

## Verification

- 124 tests passed, 5 skipped using the integration worktree's `src` on PYTHONPATH.
- The frozen drawer launcher's `verify` command passed.
- The default video CLI lists drawer, letters and workpiece and imports the
  integration worktree's implementation.
- Git comparison confirms byte-identical video implementation and entry points
  to accepted experiment commit `826b5a7`; frozen drawer files are identical to
  pre-migration main `5230648`.
- The replaced candidate commit `fded85d` is an ancestor of `from-scratch`, so its
  previous implementation remains available there.

This migration does not claim a new model accuracy result or completion of the
archived 87-episode export attempt. The accepted video evidence is documented in
[episode refinement](object-episode-iterations.md) and
[photometric verification](photometric-verification.md).
