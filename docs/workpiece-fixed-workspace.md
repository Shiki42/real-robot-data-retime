# EE-only fixed-workspace scheduling

Branch: `shuyuan/workpiece-fixed-workspace`. This experimental mode is selected
explicitly; the accepted default video pipeline remains unchanged.

Workspace occupancy and admission consider only the measured end effector (EE).
Link meshes neither block admission nor determine a waiting pose. Existing URDF
forward kinematics supplies EE positions with the 0.49 m base spacing.

The fixed box is X [0.27,0.46], Y [-0.12,0.12], Z [-0.04,0.18] metres; X is
forward, Y left, Z up. A 25 mm EE envelope expands the bounds. A waiting pose is
the last outside source frame before that EE's next entry. The next arm leaves
its waiting pose only after the previous EE has moved 40 mm outward from its
running inward extremum and left the expanded box. Left withdrawal is +Y;
right withdrawal is -Y. This gate uses already-observed movement, not a future
opposing-arm sweep. Both approaches start together and admitted actions are
uninterrupted. Braking and restart reuse the independent clocks.

Interpolated joint FK is evaluated at quarter-output-frame intervals to detect
simultaneous EE occupancy. The audit explicitly excludes link collision checks;
it is a sampled EE-volume audit, not a continuous full-robot collision certificate.
State supplies the observed EE trajectory; action remains synchronized in exported
trajectories and is not used to predict the observed withdrawal.

For this mode only, projected link overlap is permitted in compositing and is
reported. Without depth the compositor uses its existing stable right-foreground
ordering. This is an explicit visual layering approximation, not geometric
collision avoidance. Object-origin and source-identity checks remain active.

## Commands

Plan and inspect EE-workspace timing:

```bash
PYTHONPATH=src:/path/to/robo-visualize/src python scripts/plan_workpiece_workspace.py \
  --analysis /path/to/verified-analysis --joint-data /path/to/episode.parquet \
  --urdf assets/piper_x_description.urdf --mesh-root /path/to/piperx/assets \
  --output /path/to/new-plan
```

Render using the same planner and verified source checkpoint:

```bash
PYTHONPATH=src:/path/to/robo-visualize/src python scripts/render_interaction_checkpoint.py \
  --fixed-workspace --input /path/to/source.mp4 --analysis /path/to/verified-analysis \
  --joint-data /path/to/episode.parquet --urdf assets/piper_x_description.urdf \
  --mesh-root /path/to/piperx/assets --output /path/to/new-render
```

The original full-link audit is retained in
[the earlier experiment receipt](workpiece-fixed-workspace-audit.json), but its
link failures are not acceptance gates for the user-requested EE-only mode.
Current artifacts are under `/home/coder/share/retime-workpiece-ee-only-20260912`.
