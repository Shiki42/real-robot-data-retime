# Photometric evidence and source-verified compositing

This round prioritizes correctness, not speed. No human masks or physical
contact/separation labels are available. Support coverage, confidence and origin
checks are diagnostic evidence, not a measured accuracy rate.

## The remaining right-arm failure was partly a bad reference

Inspection of drawer_0 frames 560, 610 and 625 showed that the original absolute
RGB difference reference included tabletop revealed after the arm moved away.
Exposure/white-balance changes between the beginning and end of the clip also
changed the reference. Painting those pixels as robot foreground would be wrong.

On the previously fixed RobotSeg candidate masks, the right-arm longest failing
support run was 17 frames with the raw reference. Exposure normalization alone
reduced it to one frame without changing any predicted mask. The remaining
reference still contained ambiguous changes. A darkening-only experiment also
included cast shadows in a workpiece clip, so it was not adopted as the final
rule.

The final implementation reuses `match_background_colors`, using exclusions
from the independent motion detector, never from the mask being audited. It
retains broad exposure-normalized candidate geometry for discovery, but uses
separate positive evidence for prompts and auditing:

- sufficiently dark pixels with a substantial darkening relative to the initial
  exposure-normalized scene;
- exclude moderate color-preserving attenuation, which can be shadow or gray
  foreground, from positive audit evidence;
- mark other changed dark pixels ambiguous rather than declaring them background;
- keep opaque-support, ambiguous and unassessed counts explicit.

This is a cue for the dark robot in these scenes. Reflective parts and gray
material can remain ambiguous, and deep shadows can still satisfy the positive
cue. It is not a semantic foreground classifier. It does not establish semantic pixel
truth. The detector still needs sufficiently many observed support pixels and
frames. Empty masks and deliberately damaged masks continue to fail.

`robot_prompt` retains a wide context box but places positive points only on
supported pixels, and unsupported reentry candidates no longer supply positive
points. The gripper cache includes this support in its input hash. Original RGB
is not modified for neural segmentation by the motion-evidence stage.

## Controlled results

The same saved masks were audited with different references. No predictions were
changed just to improve these numbers.

| Input/masks | Raw support gate | Final photometric support gate | Empty-mask control |
|---|---|---|---|
| drawer_0, previously reseeded candidate | Fail | Pass | Fail |
| workpiece_0, original RobotSeg masks | Pass | Fail | Fail |
| workpiece_1, original RobotSeg masks | Pass | Pass | Fail |

The original, unrepaired drawer masks still fail the new test on the left arm.
Thus this is not unconditional acceptance or a lowered coverage threshold.
The workpiece_0 failure and unknown pixels are retained; the rule does not make
all videos pass. For drawer_0, 421 frame/arm pairs remain unassessed by this cue.
A support-gate pass is not complete visual validation.

## Normal pipeline and an actual parallel preview

The new evidence is used in `interaction.pipeline.run`, including mask auditing
and automatic mask repair. `motion_photometry.npz` records coefficients and
pixel counts. A frozen code snapshot was used for the final integrated drawer_0
run, reusing source-verified object measurements and revalidating all interaction
hypotheses. This integrated run uses the normal **SAM2** backend; it is not
claimed as a new RobotSeg accuracy result.

The final analysis passed all eight automatic interaction gates. Source-linked
rendering produced 462 frames at 30 FPS (15.4 s) from the 634-frame source. The
existing rendered-origin check made 328 clear observations and found zero
reintroduced-origin observations. These checks do not prove correct object
boundaries, depth ordering, or physical collision freedom. The video-only plan
uses projected silhouettes and no joint-space verification.

The preview still has visible object-edge/occlusion issues. It remains
`rendered_pending_visual_review`, with `validated_for_compositing: false`.
No dataset publication or final visual acceptance is claimed.

## Rendering safeguards and color-reference correction

`scripts/render_interaction_checkpoint.py` reuses the existing scheduler,
registration and compositor. It refuses an unfinished checkpoint, failed gates,
a different source video, or changed frame geometry/registration. Automatic
render-check failure returns a nonzero CLI status. Rendering writes to a new
output directory and does not overwrite the analysis checkpoint.

Local patch color fitting previously sampled the entire surrounding ring,
including possible robot/object or moving-drawer pixels. It now requires an
explicit background-reference mask. The compositor excludes both source-clock
foregrounds and moving scene regions. If too few background pixels remain, it
keeps the already globally color-matched source patch and records that the
additional local adjustment was skipped. A synthetic regression demonstrates
that a black foreground ring cannot whiten a colored object patch.

This fixes a sampling error; it did not eliminate every bright edge in the real
preview. A trial that selected a different early clean-plate reference introduced
an undesirable tint and was removed. The final default preserves the initial
scene reference. Committed upstream fixes through `efa49b7` were also merged,
including bounded CoTracker feature memory, exact paired-source replay and visible
boundary fragments. Uncommitted work in the separate main checkout was not used.

## Reproduce and inspect

Use the existing experiment runtime and pinned models. Normal analysis is:

```bash
python main.py --input input.mp4 --analysis-only --debug-dir /path/to/new-analysis
```

To inspect raw/corrected motion evidence for existing arm measurements:

```bash
python scripts/audit_photometric_motion.py \
  --input /path/to/drawer_0.mp4 --arms /path/to/drawer_0-arms \
  --candidate /path/to/drawer_0-reseed-agreement --output /path/to/new-audit
```

To render a completed, successful checkpoint without repeating neural inference:

```bash
python scripts/render_interaction_checkpoint.py \
  --input /path/to/drawer_0.mp4 --analysis /path/to/completed-analysis \
  --output /path/to/new-render
```

Artifacts are on Coder A under `/home/coder/share/retime-accuracy-20260910`:

- `*-opaque-motion`: raw references, positive support, ambiguity and mask controls;
- `runtime-photometric-final`: frozen integrated-analysis source and hash manifest;
- `drawer_0-final-photometric`: final interaction checkpoint and diagnostic videos;
- `runtime-render-safe-color`: frozen rendering source and hash manifest;
- `drawer_0-render-safe-color/parallel.mp4`: current 15.4-second candidate;
- `drawer_0-render-color-fixed`: rejected alternate-background experiment, not the default.

The local comparison viewer uses a read-only, authenticated Coder port forward
and a loopback-only HTTP server. It is not a public upload, and no video was
copied into the local repository.

Machine-readable reports: [evidence receipt](photometric-verification-20260910.json).

Validation: 115 tests passed, 5 skipped. Regression tests cover exposure drift,
background reveal, shadow-like ambiguity, missing-mask rejection, reliable prompt
points, foreground-free color fitting, checkpoint completion, source identity,
registration integrity and nonzero render failure status.
