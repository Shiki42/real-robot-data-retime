# Earliest safe right-arm admission at 59 cm base spacing

The current planner uses the fixed EE volume only to choose outside waiting
poses. It does not gate admission on withdrawal or on leaving that volume.
It preserves simultaneous approach starts, alternating pickups, and uninterrupted
admitted motion. Existing 0.5 s braking / 0.3 s restart clocks remain in use.

A lexicographic A* objective minimizes right-1 admission time, then right-2
admission time, then completion time, for the selected waiting poses and 30 Hz
source clocks. Cached exact URDF pose checks screen candidate paths. Adaptive
midpoint motion bounds certify every complete interpolated transition; rejected
transitions are excluded before repeating the search. This is offline planning
with known recorded trajectories, not a guarantee for unobserved live motion.
The minimum accepted measured-state mesh gap is 0.05 m plus a 1e-6 m guard.

`waiting_padding_m` enlarges the waiting envelope without changing robot meshes
or the physical EE radius. The three revised examples use padding 0 / 0.04 / 0.09 m
for Episodes 0 / 1 / 2. An earliest-time claim is conditional on those waiting
poses; it is not a global optimum over arbitrary new arm trajectories.
`render_interaction_checkpoint.py --fixed-workspace --workspace CONFIG.json`
uses the same continuously validated planner for rendering. Episode 0 uses the defaults; Episodes 1 and 2 use
`configs/workpiece-wait-padding-4cm.json` and
`configs/workpiece-wait-padding-9cm.json`, respectively.

## Full preparation onset correction

Visual `approach_start` annotations locate approach near the work area; they are
not source motion onsets. The planner now reuses measured-joint motion energy
and the original sequential phase split to find each preparation onset, with
boundary padding, before constructing both source clocks. No cumulative-energy
fraction is used to trim the beginning. Both clocks advance from output frame 0.
The first right waiting pose remains outside the configured volume.

For Episodes 0 / 1 / 2, the right source starts are 507 / 497 / 643, replacing
547 / 532 / 674. This restores 40 / 35 / 31 source frames before the prior visual
approach annotations. Left source starts are likewise 0 / 0 / 28 rather than
63 / 31 / 50. Stage receipts report both the motion onsets and the old annotation
anchors; regression coverage checks that a late visual annotation cannot cut
off the measured preparation motion.

## Historical experiments (superseded geometry and release rules)

# Backdated withdrawal with 1.55 cm measured-state URDF clearance

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

The current minimum accepted distance is 1.55 cm, as requested by the user.
The acceptance gate uses the actual recorded joint state. Commanded action is
reported independently and is not interpreted as the simultaneously observed
physical pose. Every link mesh is included in the measured-state check.

The nominal timeline is audited first. Safety repair can delay admission or
choose an earlier recorded outside waiting pose, but cannot pause an admitted
execution or delay left 1. Fixed-volume timing remains the nominal rule;
collision repair is a separately reported correction. No failed candidate is
rendered as satisfying the distance requirement.

## Adaptive between-frame verification

`ContinuousClearance` certifies each piecewise-linear exported joint interval
using midpoint FK and a conservative relative mesh motion bound. A midpoint
separation greater than 1.55 cm plus half the interval's motion bound proves the
whole interval clear. Otherwise it subdivides, rejects any violating midpoint,
and fails closed at the numerical depth limit. The only fixed threshold guard
is 1 micrometre; there is no extra fixed 1 mm rejection margin. Endpoint FK and
configuration checks are reused across candidate clocks only within the same
source trajectories and clearance setting.

This certifies the supplied URDF/base-spacing/interpolation model, not unknown
physical calibration or unmodeled objects. Image overlap can occur without a
3-D collision; rendering uses the established visual ordering approximation.

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

## Recheck at 1.55 cm

The original 3-D box and EE envelope remain the checked-in configuration; trial
reductions in box height and the envelope did not produce an accepted schedule
and were not adopted. No new passing video has been produced for this request.

For Episode 1 at left source 331, right source is constrained to [592,690] by
the backdated gates and original waiting region. An adaptive distance upper-bound
proof covers every interpolation in that entire right-source interval and keeps
its best achievable clearance below 1.55 cm. The maximum sampled gap was 1.408 cm.
Thus changing the threshold from 2 cm to 1.55 cm alone does not remove this
bottleneck. Modifying a return trajectory is beyond pure retiming and remains a
user decision. See [the 1.55 cm proof](workpiece-clearance155-infeasibility.json).

Earlier 2 cm audit results above are retained as historical evidence. Current
artifacts: `/home/coder/share/retime-workpiece-clearance155-20260912` on Coder A.

## Explicit rejected-candidate visual diagnostics

`render_interaction_checkpoint.py --diagnostic-nominal-workspace` renders the
nominal three-centimetre backdated timing without clearance repair. This option
is mutually exclusive with `--fixed-workspace`, preserves video/checkpoint source
verification, and always sets `diagnostic_only: true`,
`automatic_checks_passed: false`, and a diagnostic-not-validated status. Normal
validated planning is unchanged. Successful CLI completion in diagnostic mode
means the diagnostic artifact was generated, not that clearance passed.

`annotate_workspace_violations.py --render /path/to/diagnostic --urdf ...
--mesh-root ...` computes measured-state cross-arm URDF distance at 120 Hz for a
30 FPS clip. The annotated video displays output time, current-frame distance,
minimum sampled distance to the next frame, the closest link pair and a red
violation timeline. It labels the video DIAGNOSTIC ONLY / NOT ACCEPTED throughout.
Zero means model mesh intersection, not a measured penetration depth. Intervals
are sampled observations, not continuous safe certificates between samples.

Artifacts are under `/home/coder/share/retime-workpiece-violations-20260912`.

## Measured base spacing correction — 2026-09-12

The user remeasured the physical base spacing as **0.59 m**, replacing the
previous 0.49 m assumption. Bases are at Y = +0.295 and -0.295 m.
Earlier clearance numbers and infeasibility results at 0.49 m do not apply
to this corrected geometry. Recompute before accepting any schedule.
The first 0.59 m diagnostic replay retains the existing source clocks and
changes only geometry and distance annotations; it is not a replanned schedule.
