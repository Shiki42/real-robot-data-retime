# AIST / Molmo pilot provenance and measured scope

This branch adds the AIST adapter, bounded idle compaction, optional source-clock
search, a thin adapter to the existing uniform phase scheduler, and reproducible
pilot stages. It does not introduce the original decoupling/phase-grid concept:
those primitives already exist in `retime.py`.

The pilot was originally developed against `7d466f0`; this organized version is
based on `49b546c` and leaves the current main video pipeline intact.
Temporary patch scripts, private platform paths, weights, downloaded datasets and
rendered videos are deliberately outside the commit.

## Sources

- [AIST-Bimanual](https://aistairc.github.io/aist_bimanip_site/): small external
  raw-HDF5 pilots. Some episodes lack FPS attributes; an explicit nominal profile
  must be recorded. Some taxonomy/task descriptions differ from actual single-arm
  behavior, so labels are not a parallelism certificate.
- [ALOHA coffee](https://huggingface.co/datasets/lerobot/aloha_static_coffee):
  episodes 47, 20 and 8 were tested; conservative coupling proposals remained
  unchanged. The observed motion-overlap estimates were not achieved speedups.
- [Molmo subset](https://huggingface.co/datasets/allenai/28112025-block-02),
  revision `957f65a1fa5c86b129b2f7c3e52a3b46ea976215`: 16 numeric episodes
  screened, then episodes 15 and 8 selected for the recombination previews.

## Delivered preview measurements

All times below are frames / 30 Hz. The synthetic display hold after one panel
ends is excluded. These are measurements of experimental source-clock plans,
not real-robot execution results.

| Episode | Profile / schedule | Source seconds | Candidate seconds | Projected overlap frames |
| --- | --- | ---: | ---: | ---: |
| 15 | strict / left-first or right-first | 67.27 | 79.53 | 0 |
| 15 | strict / direct parallel | 67.27 | 44.30 | 91 |
| 15 | strict / phase-screened parallel | 67.27 | 48.90 | 0 |
| 15 | short-idle / direct parallel | 67.27 | 35.37 | 137 |
| 15 | short-idle / phase-screened parallel | 67.27 | 47.83 | 0 |
| 8 | strict / left-first or right-first | 61.27 | 65.40 | 0 |
| 8 | strict / direct parallel | 61.27 | 38.33 | 283 |
| 8 | strict / phase-screened parallel | 61.27 | 46.53 | 0 |
| 8 | short-idle / direct parallel | 61.27 | 33.00 | 291 |
| 8 | short-idle / phase-screened parallel | 61.27 | 45.40 | 0 |

The 12 delivered videos passed full decode/frame-count checks and local SHA-256
transfer checks. Four phase-screened plans passed source-index and per-axis
first/second-difference peak checks. Mask availability was complete in these two
clips; availability is not segmentation accuracy.

The phase screen uses 80x45 masks dilated by one cell and excludes the shared base.
It is a conservative 2D image-space test, not a 3D collision test. Candidate
rendering uses source pixels with an inspected serial-scene ownership hypothesis.
There are remaining shadow/boundary artifacts, and object contact/dynamics have
not been validated. These previews must not be promoted to training-ready or
physically verified demonstrations on the strength of a speedup number.

Speed increase is `T_source / T_candidate - 1`; duration reduction is
`1 - T_candidate / T_source`. A 50% speed increase is a 33.3% shorter duration,
not a halving of duration. Decoupling also needs alternate-order goal/contact
consistency evidence; speedup alone is insufficient.

See [commands and input contracts](../experiments/aist_molmo/README.md).
