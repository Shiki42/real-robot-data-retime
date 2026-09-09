# Automatic interaction and constrained retiming

## Input and scene contracts

The normal interface accepts video without clicks, bounding boxes, chosen
objects or event-frame labels. Robot state/action data are used only by trimming,
mesh scheduling and output validation, not by interaction discovery.

| Profile | Scene contract | Temporal constraint |
| --- | --- | --- |
| Drawer | Right arm operates the drawer; left arm places one cube | Pickup/lifting may overlap opening; insertion waits for opening; closing waits for withdrawal |
| Letters | Two objects per arm | Independent manipulations with collision-constrained left priority |
| Workpieces | Four parts, two per destination bin | Independent manipulations; release requires persistent deposition evidence |

These are supported task priors, not a claim of universal recognition across
arbitrary robots, viewpoints, lighting or object materials.

## Automatic evidence

Image motion and entry-side support generate whole-arm SAM2 prompts. Gripper
geometry is derived from the arm silhouette after suppressing thin cables and
bridging reflective collar gaps for geometry only. Original segmentation pixels
remain available for rendering.

Gripper closure proposes hypotheses. Acceptance also requires nearby object
motion, sustained attachment and a verified release. Object identity is checked
against source-location departure and future observations. Occlusion-consistent
positions are distinguished from visible observations; heuristic confidence
scores are not calibrated probabilities.

Colored proposals retain their hue model even when their median saturation is
low. Recovery can initialize a new SAM track at an automatically observed
reappearance. Workpiece releases require changed bin contents rather than a
brief transport pause. Drawer state combines cabinet-relative point motion,
interior appearance and observed right-arm motion. Closing onset does not imply
that the source recording eventually closes the drawer completely.

## Checkpoints and recovery

Measurement reuse checks video identity, frame dimensions, registration and the
proposal set. Current causal tests are rerun on reused hypotheses. Incomplete
whole-arm tracks are repaired per arm, keeping compatible object and drawer
measurements. Object recovery likewise reuses the existing robot measurements.

Segmentation, tracks and producer provenance are checkpointed before the
higher-memory point-tracking stage. A checkpoint is not a success report.
Completion markers are invalidated before replacing checkpoint arrays; point
archives are replaced atomically. Batch failures retain a traceback and a
nonzero outcome rather than being silently accepted.

## Scheduling and geometry

The joint scheduler uses the supplied RoboVisualize PiperX meshes, 0.49 m base
spacing and parallel +X orientation. The provided URDF includes the gripper
mount rotation. Total measured jaw aperture is split equally between the two
finger joints; out-of-stroke excursions increase the clearance allowance.

A* searches monotone per-arm source clocks. Retained poses keep their original
order and speed; waiting repeats a recorded pose. Idle compaction is allowed
only when both state and action ranges remain within 0.3 degrees / 0.5 mm.
Beginning/end poses and motion guards are retained. The drawer planner checks
its moving body, held waiting poses outside the future drawer sweep, and
empty-gripper withdrawal. The held-cube proxy radius is 35 mm.

New source-pose combinations undergo swept mesh checks. Strict clearance can
already be violated by the original cooperative drawer recording. After strict
planning is found infeasible, only exactly recorded adjacent paired edges may
be preserved in the open-drawer phase. No mismatched clocks, skipped paired
frames or extended contact holds are admitted by this rule.

| Receipt field | Meaning |
| --- | --- |
| `swept_edges_verified` | Every scheduled edge passed the configured mesh clearance |
| `new_edges_collision_free` | Every newly combined source-pose edge passed that check |
| `preserved_original_pair_edges` | Explicit output/source indices for recorded contact edges retained verbatim |

An output with a nonempty contact ledger must not be described as absolutely
collision-free. The checks concern the declared robot meshes and estimated
scene geometry; they are not a physical execution certificate.

## Pixel and object ownership

Only the main view is spatially composited. Each arm uses actual pixels at its
own source time. Source-clock origin restoration removes a picked object from
its original location. Unselected scene objects are retained. Once a cube is
released, the shared drawer scene owns its pixels and closing occlusion.

Real, registered temporal observations supply clean plates and scene patches.
Exposure fitting uses shared background evidence. Drawer-dataset overlap uses
its aligned metric depth; unavailable depth is reported separately. The video
interface without depth reports its fixed overlap ordering.

Rendering reports include source-origin duplication checks and unresolved scene
patch counts. They do not replace visual inspection of arm completeness,
placement, scene boundaries and temporal transitions. Main-only rerenders stage
and measure their video before publication; interruption cannot leave an old
receipt certifying a replaced video.

## Dataset verification

LeRobot v3 output includes per-arm source indices, raw-episode offsets and an
explicit synthetic terminal-hold flag. Action, state and arm-specific telemetry
are remapped independently. Wrist videos retain their own source view and time.
The composite main image has no single captured sensor timestamp.

`automatic_validation.validate_pending_episode` checks an individual completed
episode before global metadata is assembled. `automatic_validation.validate`
checks the complete dataset after finalization. Checks cover every numeric row,
source endpoints, bounded idle jumps, task dependencies, contact ledgers,
synthetic holds, video dimensions/counts/FPS, and every decoded wrist frame.

Final release requires all requested episodes, complete metadata/statistics,
source/provenance records, numeric/video checks, visual review and verification
of the uploaded files. Automatic processing success and preview availability
alone do not establish a completed release.
