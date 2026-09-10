"""Reverify current inference; reuse only byte-identical render inputs and code."""
from pathlib import Path
import json,hashlib,shutil,sys,numpy as np
from real_robot_data_retime.interaction.pipeline import run
from real_robot_data_retime.automatic_dataset import process_episode,analysis_identity
import real_robot_data_retime.automatic_dataset as implementation
r=Path('/home/coder/share/retime-interaction-20260909');ep=int(sys.argv[1]);cache=Path(sys.argv[2]);out=r/'drawer-retimed-b054795';work=r/'analysis-b054795';debug=work/f'episode_{ep:03d}';debug.mkdir(parents=True,exist_ok=True)
for name in ['tracks.npz','segmentation.npz','drawer_point_tracks.npz','measurements.json']:
 if (cache/name).exists():shutil.copy2(cache/name,debug/name)
video=r/f'drawer-trimmed/videos/observation.images.top/chunk-000/file-{ep:03d}.mp4';report=run(video,debug,'drawer',reuse_measurements=debug)
if not report['success']:raise ValueError(f'episode {ep} current inference failed')
identity=analysis_identity(video,Path(implementation.__file__).parent);(debug/'analysis_identity.txt').write_text(identity)
timeline=json.loads((debug/'interaction_timeline.json').read_text());old=r/'drawer-retimed-final';receipt_path=old/f'meta/retime_receipts/episode_{ep:03d}.json';old_debug=r/f'release-final-analysis/episode_{ep:03d}';reuse=False
if receipt_path.exists():
 receipt=json.loads(receipt_path.read_text());old_source=r/'validation-9daf9a3/src/real_robot_data_retime';new_source=Path(implementation.__file__).parent
 changed=[p.relative_to(new_source).as_posix() for p in new_source.rglob('*.py') if (old_source/p.relative_to(new_source)).read_bytes()!=p.read_bytes()]
 same_code=changed==['interaction/evidence.py']
 same_masks=False;same_registration=False
 if (old_debug/'segmentation.npz').exists() and (old_debug/'tracks.npz').exists():
  with np.load(debug/'segmentation.npz') as a,np.load(old_debug/'segmentation.npz') as b:
   same_masks=set(a.files)==set(b.files) and all(np.array_equal(a[k],b[k]) for k in a.files)
  with np.load(debug/'tracks.npz') as a,np.load(old_debug/'tracks.npz') as b:same_registration=np.array_equal(a['registration'],b['registration'])
 renderer=receipt['compositing']['implementation_fingerprint']
 reuse=(same_code and same_masks and same_registration and timeline==receipt['interaction']['timeline'] and renderer=='c862e60ba113f9a36289f3a73699464696655d8562b21f9783c72bdd54a2e392')
 if reuse:
  parent=hashlib.sha256(receipt_path.read_bytes()).hexdigest()
  files=[f'data/chunk-000/file-{ep:03d}.parquet',f'meta/retime_source_indices/episode_{ep:03d}.npz']+[f'videos/observation.images.{cam}/chunk-000/file-{ep:03d}.mp4' for cam in ['top','left_wrist','right_wrist']]
  for rel in files:
   dst=out/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(old/rel,dst)
  receipt['analysis_identity']=identity;receipt['interaction']=dict(timeline=timeline,report=report,measurements=json.loads((debug/'measurements.json').read_text()),robot_mask_audit=json.loads((debug/'robot_mask_audit.json').read_text()))
  receipt['reuse_verification']=dict(previous_receipt_sha256=parent,identical_timeline=True,identical_segmentation_arrays=True,identical_registration=True,implementation_changes=changed,policy='Only inference evidence selection changed; all actual selected events, masks and planning/rendering inputs remain exactly equal')
  dst=out/f'meta/retime_receipts/episode_{ep:03d}.json';dst.parent.mkdir(parents=True,exist_ok=True);dst.write_text(json.dumps(receipt,indent=2))
if not reuse:
 receipt=process_episode(r/'drawer-trimmed',r/'drawer',out,work,r/'validation-b054795/assets/piper_x_description.urdf','/home/coder/share/robo-visualize/src/robo_visualize/arms/piperx/assets',ep)
print(dict(episode=ep,frames=receipt['length'],reused_verified_output=reuse),flush=True)
