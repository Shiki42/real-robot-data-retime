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
