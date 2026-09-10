# AIST / Molmo research pilots

These entry points preserve the inspected pilot experiments without replacing
`main.py`, `batch.py`, or the current RobotSeg/TAPIR pipeline.
They are experimental source-clock and source-pixel tools, not validated robot policies.

## Install and workspace

```bash
python -m pip install -e '.[pilot,test]'
export RETIME_WORK_DIR=/path/to/pilot-work
```

Keep data, model weights and outputs outside the checkout. Every stage also accepts
`--work-dir`. Python 3.11+ and FFmpeg with H.264 encoding and libdav1d AV1 decoding
are required. CPU stages do not load SAM3.

## AIST: conservative timing diagnosis

```bash
python experiments/aist_molmo/analyze_aist.py /path/to/episode_0.hdf5 \
  --output /path/to/new-aist-report
```

If the HDF5 lacks `frame_rate`, provide `--fps 50` only when that nominal
dataset profile has been verified. Explicit FPS cannot override conflicting file
metadata. JPEG cameras are decoded to RGB; state/action stay in source units.
Generated timestamps are nominal control times, not verified camera exposure times.
Metric depth requires an explicit calibrated scale.

The report contains raw-unit tolerances, possible gripper coupling intervals and
source maps. Taxonomy labels do not decide whether an episode is parallelizable.
The contact heuristic is intentionally conservative and can over-lock separate
objects; it is not a general contact detector.

## Molmo: reproduce the two-episode study

The profile is fixed to `allenai/28112025-block-02`, revision
`957f65a1fa5c86b129b2f7c3e52a3b46ea976215`, episodes 15 and 8 at 30 Hz.
It downloads the small numeric/metadata shard and one approximately 321 MB shared
top video, then makes two frame-count-checked 640x360 clips. Files are pinned and
checked against available Hub LFS digests. Re-encoding hashes may differ across
FFmpeg versions; the prepared clip hashes are recorded in the manifest.

```bash
python experiments/aist_molmo/prepare_molmo.py --work-dir "$RETIME_WORK_DIR"
```

For the GPU stage, separately install the official SAM3 code and dependencies.
The inspected pilot used SAM3 revision
`660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7`, PyTorch 2.8.0 and
torchvision 0.23.0. The SAM3 predictor import must be available in the environment.
SAM3 is not bundled in this repository. See the official
[SAM3 installation instructions](https://github.com/facebookresearch/sam3).

```bash
python experiments/aist_molmo/track_foreground.py \
  --work-dir "$RETIME_WORK_DIR" --checkpoint /path/to/sam3.pt
python experiments/aist_molmo/plan_variants.py --work-dir "$RETIME_WORK_DIR"
python experiments/aist_molmo/sweep_phases.py --work-dir "$RETIME_WORK_DIR"
python experiments/aist_molmo/validate_plans.py --work-dir "$RETIME_WORK_DIR"
python experiments/aist_molmo/render_previews.py --work-dir "$RETIME_WORK_DIR"
python experiments/aist_molmo/render_previews.py --work-dir "$RETIME_WORK_DIR" --screened
```

`SAM3_CHECKPOINT` can replace `--checkpoint`. Rendering also accepts
`--episode 15` or `--episode 8`.
The common output root is `$RETIME_WORK_DIR/experiments/molmo_video_v4/`.

The stages produce:
- source JPEGs and packed SAM3 masks;
- strict left-first, right-first and parallel source maps;
- a separate short-idle parameter family;
- a uniform phase sweep with a projected silhouette exclusion test;
- numeric checks, comparison videos, contact sheets and SHA-256 receipts.

The recombination adapter calls the existing `build_uniform_schedule_plan`.
It maps that schedule onto non-contiguous per-arm source indices and preserves
shared entry/exit boundaries. The shared ending here is goal completion/parking,
not a screw-insertion contact event. Each parameter family reuses the same
per-arm path across its schedules; different idle-tolerance families should not
be described as identical paths.

## Profile limitations and cache reuse

SAM3 may join both arms through the common base. This pilot partitions its semantic
foreground using fixed root regions and geodesic distance. Those regions, the
640x360 image size, left-stage reference, and scene ownership region are specific
to the inspected camera and tasks. They are not calibrated geometry or a
task-independent object ownership model.

The renderer assumes a fixed camera and a right-then-left serial source scene.
It can retain visible shadow/edge artifacts. Its lower panels show both actual
source frames so that the edits can be audited. Projection overlap is marked in
the video. No 3D collision, object dynamics, wrist-view reconstruction or real
rollout success is certified.

Use a fresh work directory after changing source video, checkpoint, or profile
geometry. The historical cache layout is preserved for reproducibility; it is not
a general cache-invalidation framework. In particular, do not reuse old
`source_jpeg` or `robots.npz` files with a changed input.

The original direct-parallel candidates had projected arm overlaps. The screened
versions are reported separately in [the measured pilot notes](../../docs/aist-molmo-pilots.md).
Do not count playback/end-frame holds as task time saved.

## Verification

```bash
python -m pytest tests/test_aist.py tests/test_aist_timing.py \
  tests/test_idle_dp.py tests/test_optional_scheduler.py \
  tests/test_decoupled_clock.py tests/test_sam3_video_adapter.py tests/test_retime.py
```

The SAM3 adapter tests use a fake predictor; they do not download a model.
`validate_plans.py` checks source maps and per-axis first/second difference maxima
on the real numeric pilot files. These checks have a narrower scope than physical
execution validation.
