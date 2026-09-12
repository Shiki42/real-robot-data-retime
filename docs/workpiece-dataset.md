# Full workpiece dataset processing

Input: `Shiki42/piperx-workpiece-storage-0909-62ep-raw`, revision
`5a924d4b954def471c5f57b7896886057788fc8a`: 62 episodes and 53,366 frames.
The work directory is `/home/coder/share/workpiece-full-20260912` on Coder A.
Source files remain unchanged. Output is a separate LeRobot v3 dataset.

The pipeline preserves measured preparation motion, alternating pickups and
non-preemptive execution. Full-arm URDF separation is above 0.05 m with a
0.59 m base spacing. Waiting candidates share FK and mesh-pair caches. It first
searches shared and independent stage padding, then outside source-path parking
poses if necessary. No new robot poses are synthesized. Second approaches are
searched from the previous completed placement, including motion before late
visual approach labels. Startup is inferred before the first pickup from
measured joint motion rather than capped by an early visual annotation.

The main camera `right_environment_1` is exported as `top` and recomposited.
Wrist cameras follow their own arm's source clock; opposite-arm appearance in
those wrist images is not recomposited. State and action vectors preserve their
respective source interpolation. Clearance certificates apply to the exported
measured-state interpolation, not controller-dynamics or real-robot replay.

Run from the repository with its dependencies and Robo Visualize available:

```sh
export PYTHONPATH="$PWD/src:/path/to/robo-visualize/src"
python scripts/prepare_workpiece_dataset.py \
  --source /path/to/pinned/raw-dataset --work /path/to/work \
  --revision 5a924d4b954def471c5f57b7896886057788fc8a \
  --urdf assets/piper_x_description.urdf --mesh-root /path/to/piperx/assets
python scripts/process_workpiece_dataset.py \
  --manifest /path/to/work/manifest.json --output /path/to/work/dataset \
  --analysis-workers 2 --process-workers 1
python scripts/validate_workpiece_dataset.py \
  --source /path/to/pinned/raw-dataset --dataset /path/to/work/dataset \
  --urdf assets/piper_x_description.urdf --mesh-root /path/to/piperx/assets
```

`prepare` supports `--reuse-manifest` for previously verified inputs. Each stage
writes its own status and log. Successful packages are reused on restart;
unresolved validation failures remain explicit and never enter the accepted
export. Finalization refuses running or unfinished entries. Accepted output IDs
are contiguous; `retime.source_episode_index` and both source-frame columns
retain the original identities. Evidence and code fingerprints accompany each
accepted episode. No upload is performed by these commands.

The final validator checks source interpolation, preparation onsets, pickup
order, global indices, timestamps, video decoding and dimensions, and continuous
mesh clearance again using the exported float32 states. Dataset and episode
statistics are rebuilt, including the final global index statistics.

Batch recovery retains the original acceptance checks. Measured arm overlap can
propose a separate bin visit when a fingertip is clipped; persistent, uncovered
changed bin pixels must still verify deposition. Direct tip visits are kept
separate to avoid merging successive trips. Stationary scene branches are
excluded only from localization topology, while source robot pixels remain
intact. Expensive identity retries run after initial verification failure.
Exact photometric evidence is cached by frame content and implementation hashes.
Origin checks exclude foreground pixels so a small moving occluder cannot be
mistaken for a duplicate part; visible unoccluded duplicates still fail.

Rejected episodes retain portable status and available analysis/planning evidence
under `meta/rejections`, referenced by the processing report.
