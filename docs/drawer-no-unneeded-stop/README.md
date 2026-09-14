# Conditional lift stopping

Uniform drawer planning previously inserted the 0.5s braking and0.3s restart
ramps unconditionally. It now first computes the relative timing using the
recorded-speed lift. When the drawer is already open at natural peak arrival,
the original left source clock advances by one frame across the lift peak.
Only late-opening schedules use the smoothed stop and recompute their actual
lift duration. Uniform positions and pair spacing remain unchanged; lift
prerequisite duration may differ across variants because only one needs ramps.
Right closing waits and their smoothing remain active in either mode.

All78 sources /156 variants replanned successfully.61 variants now pass the peak
without stopping.209 tests passed,4 skipped. Original source26/27/29/30 were
rendered in both variants and passed numeric Action/State and all-camera frame
validation. This includes reported outputs100–103, whose source clocks advance
exactly1 frame per output frame around the peak with the drawer already open.

Run root: /home/coder/share/drawer-no-unneeded-stop-20260915
All156 episodes are now regenerated and finalized:82,569 frames,468 videos,
156 Action/State Parquet files. Every camera video decoded successfully; source
clock numeric comparisons, global metadata and stage checks passed.61 outputs
have no unnecessary lift stop. The four originally reported outputs100–103
are among them. Idle masks were re-exported against the new clocks, including
continuous pre-close waits and per-arm final-rest caps of45 supervised frames.

Local review: http://127.0.0.1:38771/
Local package: /Users/shuyuan/Downloads/drawer-no-stop-review-156-20260915
Main-camera copies:640x362,30FPS,H264 CRF23,85,445,713 bytes with all frames kept.
Old reviews and their notes remain unchanged. The new review has an independent
localStorage key and a next-uninterrupted-lift button. Training mask integration
is still not connected. Numerical checks do not certify flawless compositing;
source exposure/color and segmentation quality notes remain visible for review.

## Training dataset on Hugging Face

Private dataset: Shiki42/piperx-put-cube-in-drawer-20260908-87ep-ctr.
The independent LeRobot v3 export materializes retime.left_idle and
retime.right_idle boolean columns in every Parquet row, with matching feature
declarations, global stats and per-episode stats. True excludes the corresponding
arm's seven action dimensions. The training loader must explicitly apply them.
156 episodes,82,569 frames,468 camera videos; original columns and video hashes
unchanged. Arrow loader passed; remote sizes, LFS hashes and downloaded samples
verified. See hf-dataset-upload.json for immutable HF revision and details.
