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

## Task-aware idle mask review

All 78 source trajectories now have explicit close-preparation boundaries in
`review/close-preparation-boundaries.json`, applied to both variants (156 outputs).
The exporter requires complete source coverage and no longer has a per-frame
quiet-mask or missing-annotation branch. Source clocks are verified against
stored Parquet columns before exporting masks.

Left-arm required lift/open waiting remains supervised. Right-arm waiting before
the left withdrawal gate remains supervised. Between drawer opening and closing,
right-arm idle is exactly the continuous interval from the withdrawal gate to the
source's close-preparation boundary, or empty when preparation is already underway.
This overrides duplicate-clock labels in the entire pre-close phase so required
preparation cannot retain fragmented idle labels. Other phases remain unchanged.

Boundary proposals use existing measured/action quiet evidence to locate a phase
boundary, not classify individual loss frames: the last stable run before closing,
with a three-source-frame guard, and an earlier gripper-change onset when present.
Gripper changes use a 1mm excursion followed back through contiguous >0.02mm steps.
All 78 sources were inspected at four video stages and against numeric evidence.
These are sampled boundary reviews, not every-frame physical-safety certification.
Source4 keeps its prior reviewed boundary454. Source32 has substantial repositioning
already underway (over20 degrees), so its preparation remains supervised from the
withdrawal point. Source66 has no substantive extra wait. Evidence pages and the
proposal records are retained; the committed boundaries are authoritative.

`continuous-wait-validation.json` records all156 checks: 39 output masks changed,
26 outputs contain continuous excess waits (958 frames total). Before the final-rest
cap, left idle totaled8,885 frames and right idle totaled10,234. Output001 keeps [429,527), and output079
keeps [322,420). Both are half-open zero-based output-frame ranges.

The local review preserves its original localStorage key and user notes. Orange
means exclude that arm's seven action dimensions; blue means retain supervision.
These remain exported review labels: the training loader has not been connected.
The original video/Action dataset and compressed archive have not been rewritten.

## Final-rest supervision cap

Each arm now retains at most1.5 seconds of non-idle labels after it has completed
its task and settled at its final rest pose. The full suffix must fit the same
0.3-degree joint /0.5mm gripper pose ranges used by timeline.holds, for BOTH state
and action. Backward range accumulation prevents slow drift and earlier pauses
from being mistaken for final rest. Detection is bounded after left withdrawal
or right close-start; any remaining return movement delays the detected rest.

At30 FPS, frames from rest_start+45 onward are idle. Previously idle frames
remain idle, so fewer than45 supervised frames may remain. No video, Action,
source clock, or task-wait boundary is changed. Export records contain per-arm
terminal-rest boundaries; hover the timeline label to inspect them.
See review/terminal-rest-validation.json for full-cohort counts and checks.

Final-rest cap adds7,712 left-arm and46 right-arm idle frames. Final totals are
16,597 left-arm and10,280 right-arm idle frames. All312 arm timelines passed
the45-frame maximum and unchanged-before-cutoff checks;15 focused tests pass.
