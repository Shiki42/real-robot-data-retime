"""Assemble release metadata only after every episode has passed current checks."""
from pathlib import Path
import json,hashlib,shutil
from real_robot_data_retime.automatic_dataset import finalize
from real_robot_data_retime.automatic_validation import validate
r=Path('/home/coder/share/retime-interaction-20260909');out=r/'drawer-retimed-b054795';repo='Shiki42/piperx-put-cube-in-drawer-retime'
receipts=[]
for ep in range(87):
 path=out/f'meta/retime_receipts/episode_{ep:03d}.json';receipt=json.loads(path.read_text());receipts.append(receipt)
 audit=json.loads((r/f'numeric-b054795/episode_{ep:03d}.json').read_text())
 if not audit['passed'] or audit['receipt_sha256']!=hashlib.sha256(path.read_bytes()).hexdigest():raise ValueError(f'stale numeric audit {ep}')
 visual=json.loads((r/f'visual-review-b054795/episode_{ep:03d}_accepted.json').read_text())
 if not visual['passed'] or visual['receipt_sha256']!=hashlib.sha256(path.read_bytes()).hexdigest():raise ValueError(f'stale visual review {ep}')
summary=finalize(r/'drawer-trimmed',out,repo);report=validate(r/'drawer-trimmed',out)
shutil.copy2(r/'drawer-trimmed/trim_manifest.json',out/'trim_manifest.json')
ledger=sum(len(x['plan'].get('preserved_original_pair_edges',[])) for x in receipts)
card=f'''---
language:
- en
task_categories:
- robotics
tags:
- robotics
- lerobot
- bi_piperx
- synthetic-video
---
# PiperX cube-in-drawer: automatically retimed

This LeRobot v3 dataset contains all 87 episodes ({summary['frames']:,} frames at 30 FPS) derived from [Travor278/piperx-put-cube-in-drawer-20260908-87ep](https://huggingface.co/datasets/Travor278/piperx-put-cube-in-drawer-20260908-87ep), source revision `58bbbd720f6e78d162b8f4bc7077759d34c5162f`.

The main camera is a spatial and temporal composite of recorded pixels. Left-arm cube pickup/lifting can overlap right-arm drawer opening; insertion waits for opening and closing waits for confirmed release and withdrawal. Interaction events and segmentation prompts are inferred from video without user clicks, boxes, object selections or frame labels. Robot state/action data inform trimming, scheduling and verification.

Initial static trimming removed 667 frames, retaining 53,796 measured frames from 54,463 source frames. Source recordings have less than two seconds of terminal stillness; trimming preserves all original tails. Each retimed episode appends 60 repeated-boundary frames explicitly marked by `retime.synthetic_hold` to provide a two-second terminal hold. These are synthetic repeats, not additional sensor observations.

## Views and numeric alignment

Only `observation.images.top` is spatially edited. Each wrist view follows its own arm's source clock and retains its original camera view. State, action and arm-specific telemetry are selected independently from the corresponding recorded arm rows. `retime.left_source_frame` and `retime.right_source_frame` index the trimmed source; `meta/retime_source_indices` also records original raw offsets. The composite top view has no single physical sensor timestamp.

## Geometry and contact policy

Scheduling uses the supplied RoboVisualize PiperX meshes, 0.49 m base spacing, parallel +X orientation and the supplied URDF gripper-mount rotation. New pose combinations undergo swept mesh checks. When original paired recordings already violate modeled clearance, only exact adjacent original paired edges may be retained, explicitly listed in each episode's receipt. These outputs must not be described as absolutely collision-free; this is an edited training dataset, not a physical execution certificate.

## Verification and provenance

`validation.json` records whole-dataset numeric, source-clock, dependency, endpoint, video and wrist-frame checks. Episode receipts include heuristic interaction confidence, release confirmation, mask audits, rendering reports, source trim information, synthetic hold metadata and the original-contact ledger. Human-readable visual review records and file hashes are included under `meta/release`.

The pipeline uses task-specific scene priors: the right arm operates the drawer and the left arm deposits one colored cube. Confidence values are heuristic. Occluded object positions are not labeled as visibly measured motion. Temporal clean-plate reconstruction may use explicitly reported inpainting where no source pixel is observed. Minor silhouette, shadow and exposure seams can remain. Visual review samples critical frames rather than certifying every pixel of every video. See per-episode receipts for details.

Code: [Shiki42/real-robot-data-retime](https://github.com/Shiki42/real-robot-data-retime), branch `shuyuan/automatic-interaction`. Exact producer fingerprints are retained in receipts. No license is declared in the source dataset card; this derivative does not introduce a license grant for the source recordings.
'''
(out/'README.md').write_text(card)
release=out/'meta/release';release.mkdir(exist_ok=True)
shutil.copy2(r/'release-source-inventory.json',release/'source-inventory.json')
for ep in range(87):
 for suffix in ['sampling','accepted']:
  p=r/f'visual-review-b054795/episode_{ep:03d}_{suffix}.json';shutil.copy2(p,release/p.name)
manifest={}
for path in sorted(out.rglob('*')):
 if path.is_file() and '.cache' not in path.parts and path.name!='SHA256SUMS.json':
  with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
  manifest[path.relative_to(out).as_posix()]=dict(sha256=digest,size=path.stat().st_size)
(out/'SHA256SUMS.json').write_text(json.dumps(manifest,indent=2));print(summary,flush=True)
