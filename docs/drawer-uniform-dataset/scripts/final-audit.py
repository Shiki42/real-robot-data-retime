import json,hashlib,shutil
from pathlib import Path
import numpy as np,pyarrow.parquet as pq
from real_robot_data_retime.trim import source_episodes,read_episode
from real_robot_data_retime.interaction.measurements import producer_fingerprint
r=Path('/home/coder/share/drawer-uniform-dataset-20260912');c=json.loads((r/'final-config.json').read_text());out=Path(c['output']);info=json.loads((out/'meta/info.json').read_text());rows=source_episodes(out);assert len(rows)==174 and info['total_episodes']==174 and info['fps']==30;assert [x['episode_index'] for x in rows]==list(range(174));total=0;results=[]
for row in rows:
 ep=row['episode_index'];table=read_episode(out,info,row);n=len(table);assert table['index'].to_pylist()==list(range(total,total+n));assert row['dataset_from_index']==total and row['dataset_to_index']==total+n;total+=n
 receipt=json.loads((out/f'meta/retime_receipts/episode_{ep:03d}.json').read_text());assert receipt['plan']['producer']==producer_fingerprint();assert receipt['visual_review']['passed']
 validation=json.loads((r/'validation'/f'source-{ep%87:03d}.json').read_text());v=next(x for x in validation['results'] if x['output']==ep);assert validation['passed'] and v['frames']==n and v['numeric_exact'];results.append(v)
 for key in ['top','left_wrist','right_wrist']:
  assert row[f'videos/observation.images.{key}/to_timestamp']==n/30
assert total==info['total_frames']==97769
files=list((out/'videos').rglob('*.mp4'));assert len(files)==522;assert len(list((out/'data').rglob('*.parquet')))==174
manual=[]
for ep in [60,35,61]:
 a=Path(c['analyses'][str(ep)]);note=json.loads((a/'manual-annotation.json').read_text());note.update(status='accepted',validation=dict(passed=True,outputs=[ep,ep+87],numeric_and_three_cameras=True,sampled_stage_visual_review=True));manual.append(note)
ledger=dict(limit=5,used=3,entries=manual,abandoned_diagnostics='Mask dilation and generic scene/ownership probes remain under their separate probe directories and are not part of this dataset or producer.')
(out/'manual-interventions.json').write_text(json.dumps(ledger,indent=2));shutil.copy2('docs/drawer-uniform-dataset/trim-summary.json',out/'trim-summary.json');shutil.copy2('docs/drawer-uniform-dataset/trim-frames.csv',out/'trim-frames.csv');shutil.copy2(r/'visual-review.json',out/'visual-review.json')
report=dict(passed=True,source_episodes=87,episodes=174,frames=total,fps=30,duration_seconds=total/30,camera_videos=522,parquet_files=174,producer=producer_fingerprint(),code_commit='a29cc6b',trim_head_removed=667,trim_tail_removed=0,manual_source_count=3,results=results,visual_scope='All174 outputs reviewed at lift, end of wait, postrelease and final frames; not exhaustive every-frame certification. Minor segmentation seams, exposure differences and missing-depth ordering remain limitations.',unknown_depth_order='existing stable_right_foreground',overlap_pixel_frames=sum(x['overlap_pixel_frames'] for x in results),depth_valid_overlap_pixel_frames=sum(x['depth_valid_overlap_pixel_frames'] for x in results))
(out/'validation.json').write_text(json.dumps(report,indent=2));(r/'final-audit.json').write_text(json.dumps({k:v for k,v in report.items() if k!='results'},indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ['results','visual_scope']},indent=2))
