import json,shutil
from pathlib import Path
from real_robot_data_retime.trim import source_episodes,read_episode
R=Path('/home/coder/share/drawer-10px-cohort-20260912');c=json.loads((R/'config.json').read_text());out=Path(c['output']);ids=c['source_indices'];info=json.loads((out/'meta/info.json').read_text());rows=source_episodes(out);assert len(rows)==2*len(ids);results=[];total=0
for row in rows:
 ep=row['episode_index'];t=read_episode(out,info,row);n=len(t);assert t['index'].to_pylist()==list(range(total,total+n));assert row['dataset_from_index']==total and row['dataset_to_index']==total+n;total+=n
 source=ids[ep%len(ids)];v=json.loads((R/'validation'/f'source-{source:03d}.json').read_text());assert v['passed'];record=next(x for x in v['results'] if x['output']==ep);assert record['frames']==n and record['numeric_exact'];results.append(record)
assert total==info['total_frames'];assert len(list((out/'videos').rglob('*.mp4')))==3*len(rows)
report=dict(passed=True,source_episodes=len(ids),episodes=len(rows),frames=total,camera_videos=3*len(rows),fps=30,visual_review='Pending human review; numerical validation does not certify compositing quality.',results=results)
(R/'validation.json').write_text(json.dumps(report,indent=2));shutil.copy2(R/'validation.json',out/'validation.json');shutil.copy2(R/'selection.json',out/'selection.json')
print(json.dumps({k:v for k,v in report.items() if k!='results'},indent=2))
