# Workpiece and letter episode refinement

The accepted drawer clip is the visual reference. The goal is coherent foreground
and object identity, not minimum runtime. There are no human segmentation or
contact labels; automatic checks below are diagnostics, not measured accuracy.

## Correct scene ownership

Previously every object was painted with its assigned arm at every source time.
An untouched right-arm object could therefore be painted over the independently
retimed left arm. In the workpiece diagnostic, source frames left=165/right=564
exposed this ordering error. It can look like an object attached to the wrong arm
even when the underlying object identity is unchanged.

Stationary objects now render below both arms. Only objects between their inferred
pickup and release frames join the corresponding moving arm layer. Stationary
object pixels occluded by robots at their donor source time are excluded. Drawer
contents after release remain owned by the drawer scene.

The visual planner reuses the same moving-foreground helper, so a carried object
extending outside its arm mask is included in swept silhouette checks. Stationary
objects do not block approach. These are image-space constraints, not joint-space
or physical collision guarantees. Longer waits can be necessary to preserve this
constraint; the faster workpiece trial that ignored carried extents was rejected.

## Letter mask contamination

A fresh letter analysis initially accepted only three interactions. SAM's whole-arm
mask still contained approximately 97% of the already deposited M at source frame
255, pulling the derived fingertip onto that letter. Its estimated release was
then delayed to frame 339, conflicting with the next I pickup near frame 323.

For the letter task, independently tracked letter masks are now subtracted from
whole-arm masks before deriving final gripper geometry and evaluating interaction
hypotheses. Existing gripper extraction is reused. No release threshold was
relaxed and the temporal scoring rules were not changed. A direct diagnostic
restored the estimated M release to frame 255. Missing object observations remain
missing; the implementation does not interpolate hidden trajectories.

The refinement runs after object tracking, including when reinterpreting saved
measurements, and records its parent measurement producer. The compositor restores
carried object pixels through its explicit ownership rule. A regression test
confirms that a detached object cannot pull the fingertip away from the robot.

## Verification

- 118 tests passed, 5 skipped. New tests cover pre-pickup and post-release object
  occlusion by the opposite arm, carried-object layering and planning clearance.
- The accepted drawer was rendered again at the same source mapping: all 462
  decoded frames are pixel-identical to the accepted clip.
- Workpiece was reanalysed from the source video with current photometric motion
  evidence and SAM2. Four interactions and all six current gates passed.
- The new workpiece render has 759 frames at 30 FPS (25.3 seconds), zero moving
  foreground overlap pixels, and zero origin duplicates in 1,478 clear checks.
- Letter revalidation accepted all four interactions and all six gates. Its render
  has 849 frames at 30 FPS (28.3 seconds), zero moving foreground overlap pixels,
  and zero origin duplicates in 2,109 clear checks. The clean plate has 48 inpainted
  pixels; no unresolved scene-patch pixels were reported. This is explicitly not
  a claim of pixel-perfect segmentation.
- Letter review sampled 16 distributed output frames and source frames around the
  M release/I pickup transition. All four letters remain present in the final layout.
- Workpiece review sampled 16 distributed output frames plus targeted source/mask
  comparisons. Source motion blur and some fine cable-edge imperfections remain.

Artifacts and logs are private on Coder A under
`/home/coder/share/retime-accuracy-20260910/object-iterations`.
`runtime-ownership` preserves the rendering source and file hashes.
`workpiece_1-final` and `letters_0-final` contain the source-verified renders.
`letters_0-object-free-analysis` contains the successful letter revalidation;
the preceding three-interaction failure remains in `letters_0-analysis`.
`runtime-letter-exclusion` preserves the source used for letter revalidation and
rendering. The final committed code only removes an extra blank line from this
snapshot. Full source/render reports and hashes are in
[the machine-readable receipt](object-episode-iterations-20260910.json).
`drawer-regression/decoded-comparison.json` records the drawer regression.

The local MP4 files are `workpiece_1-v2.mp4` and `letters_0-v2.mp4`; delivery only remuxes
H.264 packets with MP4 fast-start metadata. Browser previews use VP9/WebM. Neither
operation changes the frame mapping. The original accepted drawer artifact is
preserved.

## Reproduce

Use the existing pinned model environment. For either `workpiece_1` or `letters_0`:

```bash
python main.py --input /path/to/source.mp4 --analysis-only \
  --debug-dir /path/to/new-analysis
python scripts/render_interaction_checkpoint.py --input /path/to/source.mp4 \
  --analysis /path/to/new-analysis --output /path/to/new-render
```

The second command rejects incomplete or failed analyses and mismatched source
videos/registration. It writes a new directory and preserves the analysis.
These visual-only renders are not edited robot-control datasets.
