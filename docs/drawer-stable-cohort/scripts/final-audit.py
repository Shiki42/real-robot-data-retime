import json,shutil,numpy as np,pyarrow.parquet as pq
from pathlib import Path
from collections import Counter
from real_robot_data_retime.trim import source_episodes,read_episode
from real_robot_data_retime.interaction.measurements import producer_fingerprint
R=Path('/home/coder/share/drawer-stable-cohort-20260912');c=json.loads((R/'config.json').read_text());out=Path(c['output']);info=json.loads((out/'meta/info.json').read_text());rows=source_episodes(out);ids=c['source_indices'];assert len(rows)==2*len(ids)==118;total=0;counter=Counter();results=[]
for row in rows:
 ep=row['episode_index'];table=read_episode(out,info,row);n=len(table);assert table['index'].to_pylist()==list(range(total,total+n));assert row['dataset_from_index']==total and row['dataset_to_index']==total+n;total+=n
 receipt=json.loads((out/f'meta/retime_receipts/episode_{ep:03d}.json').read_text());source=receipt['source_episode_index'];counter[source]+=1;assert receipt['plan']['source_cohort']==ids and receipt['plan']['producer']==producer_fingerprint();assert source==ids[ep%len(ids)] and receipt['plan']['variant']==ep//len(ids)
 validation=json.loads((R/'validation'/f'source-{source:03d}.json').read_text());v=next(x for x in validation['results'] if x['output']==ep);assert validation['passed'] and v['frames']==n and v['numeric_exact'];results.append(v)
 assert receipt['visual_review']['passed']
 for camera in ['top','left_wrist','right_wrist']:assert row[f'videos/observation.images.{camera}/to_timestamp']==n/30
assert set(counter)==set(ids) and set(counter.values())=={2};assert total==info['total_frames']==62651;assert len(list((out/'videos').rglob('*.mp4')))==354
selection=json.loads((R/'selection.json').read_text());assert not set(counter)&set(selection['excluded_sources']);timing=json.loads((R/'timing-summary.json').read_text());report=dict(numerical_and_structural_passed=True,source_episodes=59,episodes=118,frames=total,fps=30,camera_videos=354,source_episode_indices=ids,excluded_source_indices=selection['excluded_sources'],producer=producer_fingerprint(),code_commit='d2f4ba2',timing={k:v for k,v in timing.items() if k!='results'},visual_review_scope='Both variants of every retained source reviewed at lift, wait end, postrelease and final frame. This is task-semantics review, not flawless image-quality certification. Exposure/color and mask seam notes remain for human review.',results=results)
(out/'validation.json').write_text(json.dumps(report,indent=2));(R/'final-audit.json').write_text(json.dumps({k:v for k,v in report.items() if k!='results'},indent=2))
for name in ['selection.json','timing-summary.json','trim-summary.json','visual-review.json','destination-audit.json','destination-disposition.json']:shutil.copy2(R/name,out/name)
shutil.copy2('docs/drawer-stable-cohort/excluded-sources.csv',out/'excluded-sources.csv')
old=json.loads(Path('/home/coder/share/drawer-uniform-dataset-20260912/final-dataset/manual-interventions.json').read_text());entries=[x for x in old['entries'] if x.get('source_episode',x.get('source')) in ids];assert len(entries)==2;(out/'manual-interventions.json').write_text(json.dumps(dict(limit=5,used=2,entries=entries),indent=2))
print(json.dumps({k:v for k,v in report.items() if k not in ['results','visual_review_scope','source_episode_indices','excluded_source_indices']},indent=2))
