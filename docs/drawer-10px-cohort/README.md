# 10 px P90 cohort: 156 episodes

Generated on Coder A at `/home/coder/share/drawer-10px-cohort-20260912/dataset`.

User accepted a background-relative fixed-knob displacement P90 of at most 10 pixels
in the 640-pixel-wide analysis view. This is not a maximum-displacement criterion.
78 of the original 87 sources are retained; 19 sources / 38 outputs are restored.
Sources 0, 1, 2, 21, 28, 35, 36, 45 and 67 remain excluded.
See `selection.json` for all original source IDs and measured values.

All 78 sources were replanned and rendered with production code 04a1387.
The uniform grid depends on cohort size, so all 156 outputs were regenerated:
`u=(retained_source_rank + variant*78)/156`. Paired positions differ by 0.5.
Original trajectories, static-wait compression and post-release left retreat closing
policy are retained. No new manual source annotations were introduced.

Finalization and numerical/structural validation passed: 156 episodes, 83,187 frames,
30 FPS, 468 RGB videos and 156 Action/State Parquet files. All camera videos were
decoded to verify frame counts; source-clock Action/State interpolation was checked
at stored precision. Global indices, source cohort, pair spacing and timestamps
were verified. This does not certify flawless compositing; human review remains pending.
Known source exposure/color and segmentation-edge notes are retained in the review UI.

Existing Trim results are reused, without re-trimming: 48,372 raw frames -> 47,757
trimmed input frames; 615 initial frames (20.5 seconds) removed, no tail frames removed.
Old 174- and 118-output datasets and local reviews are preserved.

Main-camera review copies contain every output frame at 640x362 / 30 FPS, H264 CRF23,
86,013,340 bytes total. The new review has independent localStorage and a next-restored
button for the 38 added outputs. The full three-camera dataset remains on Coder A.
Local review target: `/Users/shuyuan/Downloads/drawer-10px-review-156-20260912`.
Run scripts and reports are preserved here; use a new output root for reproduction.
