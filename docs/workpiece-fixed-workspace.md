# Backdated withdrawal with strict URDF clearance

The fixed cuboid selects an outside EE staging pose. Following the user's latest
rule, an observed lateral retreat of 3 cm confirms an event, then the nominal
release is backdated to the onset of that retreat. The owner need not have left
the cuboid. Simultaneous EE presence inside it is therefore allowed.

Left-outward motion is +Y; right-outward is -Y in the existing URDF world frame.
From each verified pickup the detector maintains the inward-most observed lateral
position and its latest frame (the end of an equal-position dwell). The first
sample 3 cm outward confirms the event; that saved turning frame is the nominal
release. Both `withdrawal_frame` and `withdrawal_confirmed_frame` are reported.
This is offline retrospective editing, not a causal online release detector.

## Safety precedence

The prior EE-only acceptance policy is superseded. Every cross-arm URDF mesh
must stay strictly farther than 2 cm, for both measured state and commanded
action trajectories, including all waits, brakes and restarts. The system first
constructs the nominal backdated timeline and audits it. If it fails, the existing
monotone scheduler delays admission and, when necessary, backs the waiting pose
farther out along the same recorded outside approach. It never skips poses or
pauses an admitted execution. Left 1 cannot be slowed or delayed. Right/left
approaches still start together. A failure to find such a path is explicit.

The collision repair is intentionally separate from the nominal fixed-volume
release rule. Actual admission and additional wait relative to the backdated
onset are recorded. It is not possible to promise exact-onset admission and also
override a detected collision; the strict distance requirement takes precedence.

## Between-frame guarantee within the geometric model

`collision.continuous.ContinuousClearance` reuses `PiperXClearance` FK, meshes,
and its conservative reach/angle/aperture motion bound. It checks separation at
2.1001 cm while limiting relative mesh travel between checked samples to 0.2 cm.
Every intermediate point is within 0.1 cm of a checked sample, leaving a strict
lower bound above 2 cm. Full-motion AABB separation can certify an entire edge
without subdivision. Edges are piecewise-linear in the exported joint rows.
The final trajectory is independently audited after scheduling.

This certifies the supplied URDF/base-spacing/interpolation model, not unknown
physical calibration or unmodeled objects. Image overlap is allowed in rendering
because physically separated geometry can overlap in projection; the established
stable foreground ordering remains a visual approximation without depth.

## Artifacts and entry points

Use the existing `scripts/render_interaction_checkpoint.py --fixed-workspace`
with verified analysis, matching `--joint-data`, `--urdf` and `--mesh-root`.
`plan_workpiece_workspace.py` produces timing and mesh-audit reports without video.
`nominal_collision_failures`, `mesh_audit`, `stages.admissions` and
`stages.collision_search_attempts` document what was changed for clearance.
Earlier EE-only results remain historical records, not current acceptance gates.

Current experiment artifacts:
`/home/coder/share/retime-workpiece-retreat-20260912` on Coder A.

## Current feasibility result

The three original synchronized recorded trajectories have minimum measured-state
mesh distances of 13.60, 11.94 and 10.95 cm. Episode 1 at left/right source frame
330 has 15.56 cm separation. Thus the newly retimed 1.55 cm case is not an original
close-approach exception. See [original clearances](workpiece-original-clearance.json).

Episode 1 proves a conflict among the retained constraints: when left source 330
must execute, right source must be at least 592 (left-2's release gate) and at most
690 (right-2's outside staging boundary, gated until left source 331). Exact
recorded-pose evaluation gives a maximum available measured clearance of 1.553 cm
across that interval. Subdivision with a 1 mm motion-bound allowance bounds all
interpolated right poses below 1.653 cm. Therefore even arbitrary waiting cannot
meet >2 cm with this region, source path and gate combination. The independent
commanded-action bound is also below 2 cm.

The [interval proof](workpiece-retreat-infeasibility.json) preserves the bounds.
The safe-video objective is not completed: no unsafe output is relabeled as safe.
Remaining choice is to revise the fixed region/waiting constraints while retaining
recorded motion, or authorize new geometric avoidance trajectories. `main` is
unchanged. The nominal three-centimetre onset detector and fail-closed continuous
clearance implementation are retained on this experimental branch.

Normal braking remains 0.5 s. Candidate safety repair may shorten a braking window
only after the previous placement is complete; it must not retime that completed
execution. Every such ramp is recorded, and this does not waive mesh clearance.
