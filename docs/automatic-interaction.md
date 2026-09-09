# Automatic interaction and constrained retiming

Implementation in progress on Coder A. No manually supplied boxes, clicks, object
IDs, or event frames are accepted in the normal video interface. Robot signals
are reserved for trimming, independent validation and collision scheduling.

Task contracts:
- Drawer: right opens while left picks up and approaches a safe original pose;
  left insertion waits for opening completion. Right closing waits for left
  withdrawal. Both moving drawer geometry and held cube must be considered.
- Letters/workpieces: independent manipulation, left priority, right waits only
  where necessary; waiting poses must remain safe as left traverses its future.
- Preserve recorded poses and source pixels; no invented IK trajectory that
  cannot be represented in the source video.
- Export one synchronized version per source episode; preserve two seconds of
  available terminal stillness. Numeric and wrist streams follow their arm.

Geometry: RoboVisualize dual.py base spacing 0.49 m, parallel +X facing bases,
left at +Y. assets/piper_x_description.urdf is the user-provided local snapshot
including the gripper-base 90-degree mounting correction. Meshes and FK are
loaded from RoboVisualize. Cross-arm margin starts at 0.02 m; transition sampling
at most 0.25 degrees / 0.25 mm. This is a sampled geometry check, not proof of
continuous physical safety; scene/held-object verification is additionally needed.

Gates: first demonstrate video-only events and candidate evidence on two episodes
of each task, then segmentation/compositing, then full drawer trim/retime and
public HF upload. Failed confidence/geometry checks must be recorded and must
not be reported as successful edited episodes.

The gripper signal is total jaw aperture in mm (observed up to approximately
70 mm). Each URDF finger has a 35 mm prismatic stroke. Collision FK converts
total aperture to half-aperture before calling the existing single-finger API.
Integration tests verify both 35 and 70 mm values, and reject overlapping bases.

Development runtime: Python 3.11, PyTorch 2.8.0+cu128, torchvision 0.23.0+cu128,
transformers 5.16.1, Pinocchio 2.7.0. Install PyTorch using the CUDA 12.8 index
on Coder A; the current default CUDA 13 wheels exceed its installed driver.

## Current real-video evidence

Six held-out sample videos (episodes 0/1 of each task) are extracted automatically.
The first geometry-only reports correctly remain `success: false`: wrist/cable
confusion, orientation-dependent apparent aperture, and tracking loss across
occlusion prevent reliable event verification. Neither GroundingDINO nor sparse
Qwen3-VL-4B localization passed visual inspection; neither is a runtime authority.
SAM2 is publicly accessible; SAM3 access was denied and is not used.

The SAM2 backend now streams frames with a bounded cache after whole-video
preprocessing caused three experiments to exit 137 under concurrent memory load.
Object prompts and gripper prompts are generated automatically. Missing visible
coordinates remain NaN and must not count as measured correlation evidence.

A task-specific RGB aperture regressor is trained from existing recorded sensor
values (no manual labels), while inference accepts only images. Episodes 0/1 are
excluded from both optimization and checkpoint selection; every seventh remaining
episode is validation. Scripts and checkpoints preserve split provenance.

Source collision audit at 2 cm: sampled letters/workpiece episodes 0/1 have no
violations. Drawer samples have near-gripper separations down to about 1.8 mm;
these do not establish mesh collision but do not pass a 2 cm clearance criterion.
The final drawer schedule must resolve or explicitly reject such configurations.

The first RGB aperture model passed white-table validation (roughly 1–2 mm MAE)
but failed the dark-table episodes 0/1 (roughly 5–10 mm MAE, missed opening events).
This is a measured domain shift, not a successful Phase 1 result. A revised
local-contrast preprocessing and event-weighted training run is being evaluated.
Its checkpoint contract explicitly records preprocessing to prevent applying
new image normalization to an old RGB model.

## New causal checks and measured scheduling results

Opening is corroborating evidence, not a mandatory large-amplitude trigger:
letters episode 0's first manipulation has only about 2 mm recorded jaw change,
and deposition does not show a large opening. Observed object transport followed
by stationary deposition and gripper separation is therefore required even when
aperture is weak. A track is rejected if its original object remains visible.

For independent tasks, both sample episodes passed swept mesh checks with left
priority: letters 1048→909 and 1048→878 source frames; workpieces 1063→881 and
952→784. These are joint scheduling checks, not visual-edit completion claims.

Drawer sample 0: automatic handle tracking identifies an open dwell around
frame 159 and closing near 435. Forward/backward cube tracking estimates pickup
303, with observed uncertainty interval [303,318], and confirms release around
382. The image-space entry gate is around 344; a conservative drawer-volume
proxy moves the latest safe held wait earlier, to around 338 in the first audit.
The proxy is an estimated scene volume, not a measured drawer CAD model.

Additional checks now include true mask-to-object proximity, one-to-one terminal
letter matching, negative prompts from other objects, large-SAM2 retries for
identity conflicts, and pickup uncertainty intervals. Large SAM2 recovered the
letter T trajectory after the small model switched to N. Cabinet-relative
point motion and normalized interior-area motion are separate drawer hypotheses,
since one recorded episode moves the entire cabinet while closing.

A* scheduling matches exhaustive small-grid optima and the earlier exact dynamic
program's four real-sample output lengths, while reducing their search times.
Batch errors retain full tracebacks and per-video failure reports; failed batches
return a nonzero status after processing the remaining videos.

The short-term camera tracker was found to follow robot motion, producing a
false approximately 40 px shift in a letter sample. It was replaced with
reference-frame ORB matching and long-term background consistency. On the three
sample-0 videos, accepted background shifts are approximately 1–2 px; synthetic
camera/foreground-motion separation tests pass. Gripper identity is additionally
constrained by entry-edge motion support to prevent same-looking arm swaps.

Reports now expose automatic gates (arm count, object coverage, persistent track
visibility, original-location departure, task roles and drawer state). Passing
these per-video checks is distinct from the subsequent visual-compositing and
full-dataset release gates. Models trained for aperture experiments remain
auxiliary research artifacts; the video workflow does not read sensor labels.

Whole-arm segmentation now anchors end-effector identity. The end-effector is
located by geodesic distance inside a cable-filtered arm silhouette; direct
single-gripper tracking had latched onto deposited objects or the other arm.
Missing object coordinates can be explained as robot occlusion, but predicted
attachment points are never used as measured motion evidence. Candidate episodes
are selected jointly with unique-object and same-arm non-overlap constraints.

Recorded gripper excursions beyond the URDF stroke now enlarge the collision
margin rather than disappearing in clipping. All five existing sample schedules
passed this stricter audit. A read-only trim analysis of the complete drawer
source found 667 removable initial frames and no excess terminal hold; all 87
strictly measured tails are shorter than two seconds. Any added retime terminal
hold must be explicitly labeled as repeated source boundary poses/frames.
