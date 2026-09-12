# Fixed-workspace experiment

Branch: `shuyuan/workpiece-fixed-workspace`. The accepted video route and main
branch are unchanged. This is a rejected initial candidate, not a safe exporter.

The requested policy uses a constant cuboid and observed withdrawal rather than
the future opposing-arm sweep. `workpiece_workspace.py` uses recorded state FK
through the existing `PiperXClearance` model (0.49 m base spacing). The current
box is X [0.27,0.46], Y [-0.12,0.12], Z [-0.04,0.18] metres. X is forward, Y is
left, Z is up. Leftward retreat toward the left base is +Y; rightward is -Y.
These approximate task-space bounds are not a calibrated camera projection.

The initial candidate uses a 25 mm EE envelope, waits immediately outside the
expanded fixed box, and admits the next arm after the previous EE has moved
40 mm laterally outward from its running inward extremum and cleared the box.
The running extremum is causal: later source poses cannot change a past release.
The original two approaches start together. Three fixed admission gates impose
left 1, right 1, left 2, right 2. Only designated outside staging poses may hold.
Existing independent braking/restart source clocks are reused.

## Findings

All three original unmodified episodes pass swept URDF mesh checks for both
state and action at a 5 mm margin. The initial concurrent fixed-EE-volume
candidate fails: state/action failed output edges are 32/67, 61/68 and 138/143.
Projected overlap frames are 0, 0 and 124. A projected-only check is therefore
insufficient. Full-arm meshes must be accounted for; EE exclusion alone cannot
serve as physical shared-space occupancy.

A whole-arm/box probe finds that Episode 1 right 1 does not fully exit this box
before its inferred retract end, even though its EE leaves. Requiring full-arm
clearance with this box would deadlock. Do not silently bypass this with a future
sweep planner or render the rejected candidates as safe. The next geometry design
must distinguish persistent proximal-link occupancy from the actual shared
manipulation volume, while retaining full-mesh validation independently.

`plan_workpiece_workspace.py` writes clearly named candidate maps and a full
state/action swept-mesh audit. Mesh failure returns nonzero and the receipt stays
`validated_for_rendering: false`; even a mesh pass requires a projected/render
check. There is no automatic collision-dependent rescheduling.

183 tests passed, 4 skipped. New tests verify fixed entry/withdrawal boundaries,
future independence, and explicit invalid-geometry/missing-withdrawal failures.

Artifacts, synchronized per-episode joint tables, FK tracks, cuboid illustration,
original and candidate audits live at
`/home/coder/share/retime-workpiece-fixed-workspace-20260912` on Coder A.
The [audit receipt](workpiece-fixed-workspace-audit.json) records candidate failures.

```bash
PYTHONPATH=src:/path/to/robo-visualize/src python scripts/plan_workpiece_workspace.py \
  --analysis /path/to/verified-analysis --joint-data /path/to/episode.parquet \
  --urdf assets/piper_x_description.urdf --mesh-root /path/to/piperx/assets \
  --output /path/to/new-candidate
```
