# Uniform drawer timing

`real_robot_data_retime.uniform_drawer` samples **two** prerequisite timings
for every source episode. A is pickup through the recorded held lift peak;
B ends when the drawer is fully open. Insertion C begins
only after both finish. Closing remains after left-arm withdrawal.

For source episode `i` among `N` sources:

```text
u(i, variant) = (i + variant * N) / (2 * N), variant in {0, 1}
relative_onset = A_duration - u * (A_duration + B_duration)
```

The two positions differ by exactly 0.5. Across the dataset, positions cover
`[0, 1)` uniformly; `u=0` is A immediately before B, and the excluded `u=1`
boundary is B immediately before A. Durations are frame intervals, not counts
of endpoint poses. Onsets round to the nearest video frame (at most 0.5 frame
per onset, and at most one frame error in the pair difference).

Each variant uses the same 0.5 s braking / 0.3 s restart clock at A's peak,
including when no peak dwell is needed, so A's duration cannot depend on the
sampled partner delay. The right clock uses the main staged drawer implementation: opening to the
verified open frame, an open-pose wait, then the original closing suffix.
Post-opening gripper adjustments are not added to B to enlarge its interval.

The measured jaw closure/reopening interval constrains peak selection. A late
visual attachment confirmation does not exclude an earlier held height peak,
and a post-release empty-jaw closure cannot become a candidate holding pose.

The sampled clocks are fixed. A geometry failure never changes the sample's
phase or substitutes a different source episode. The sampler retains the main staged planner's projected-silhouette check
and its metric held-peak/braking clearance checks. Its model retains the
repository's assumed 0.49 m base spacing and inferred drawer box. These are
model checks, not calibrated physical collision certification.

## Execution

A JSON config specifies `source` (trimmed RGB dataset), `raw_source`, `output`,
`work`, `urdf`, `mesh_root`, `repo_id`, `source_provenance`, and `analyses`
(mapping source episode IDs as strings to verified analysis directories).
Source video hashes, registered frame geometry and analysis gates must match.

```bash
python -m real_robot_data_retime.uniform_drawer --config config.json --phase plan --episode 0
python -m real_robot_data_retime.uniform_drawer --config config.json --phase render --episode 0
python -m real_robot_data_retime.uniform_drawer --config config.json --phase finalize
```

Planning stores both source clocks, phase values, rounded delays, actual stage
output frames and producer fingerprints. Rendering reuses the main foreground
compositor and shared LeRobot telemetry/wrist-video exporter. Output episode
IDs are `i` and `i+N`; every receipt records its original source episode.
Finalization requires all `2N` receipts. A folder with only pilot episodes is
not a complete dataset and must not be used as one.

## Initial 87-episode run

Source: `Travor278/piperx-put-cube-in-drawer-20260908-87ep`, revision
`58bbbd720f6e78d162b8f4bc7077759d34c5162f`.
Coder A run directory: `/home/coder/share/drawer-uniform-174-20260911`.
The original contains 87 episodes and 54,463 frames; the existing RGB trim
contains the same 87 episodes and 53,796 frames at 30 FPS.

The run is a preflight/pilot, **not a completed 174-episode dataset**. Existing
analysis checkpoints passed interaction checks for 82/87 episodes; five
additional source-matching verified checkpoints were located, and episode 0
was reanalyzed with the current pipeline. Interaction verification alone does
not establish that both sampled timelines pass geometry checks. Per-episode
logs retain failed source edges. Pilot source-clock, telemetry and RGB frame
checks are separate from completing and visually reviewing the full dataset.
