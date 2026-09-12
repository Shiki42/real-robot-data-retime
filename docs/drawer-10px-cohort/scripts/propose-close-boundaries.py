import json,numpy as np,pyarrow.parquet as pq
from pathlib import Path
from real_robot_data_retime.timeline.holds import stationary_pose_mask
from real_robot_data_retime.retime import boolean_ranges
r=Path('/home/coder/share/drawer-10px-cohort-20260912/dataset');c=json.loads((r/'uniform_manifest.json').read_text())['source_config'];rows=[]
for rank,src in enumerate(c['source_indices']):
 rec=json.loads((r/f'meta/retime_receipts/episode_{rank:03d}.json').read_text());st=rec['plan']['stages'];maps=np.load(r/f'meta/retime_source_indices/episode_{rank:03d}.npz');gate=int(np.searchsorted(maps['left'],st['withdrawal_source_frame']));close=int(np.searchsorted(maps['right'],st['close_source_frame']));begin=int(np.floor(maps['right'][gate]));end=st['close_source_frame'];t=pq.read_table(Path(c['source'])/f'data/chunk-000/file-{src:03d}.parquet');state=np.array(t['observation.state'].to_pylist())[:,7:];action=np.array(t['action'].to_pylist())[:,7:];quiet=stationary_pose_mask(state,action,30)
 runs=[x for x in boolean_ranges(quiet) if x.end>begin and x.start<end and x.end-x.start>=3];boundary=begin
 if runs:boundary=min(end,max(begin,runs[-1].end-3))
 grip_events=[]
 for name,values in [('action',action[:,6]),('state',state[:,6])]:
  baseline=values[begin];ids=np.flatnonzero(np.abs(values[begin:end]-baseline)>=1.0)
  if len(ids):
   hit=begin+int(ids[0]);start=hit
   while start>begin and abs(values[start]-values[start-1])>0.02:start-=1
   grip_events.append(dict(signal=name,onset=start,threshold_crossing=hit));boundary=min(boundary,start)
 if src==4:boundary=454
 boundary=max(st['open_source_frame'],min(end,boundary));out=int(np.searchsorted(maps['right'],boundary));rows.append(dict(source=src,output=rank,gate=gate,closing=close,gate_source=begin,close_source=end,close_preparation_source_frame=boundary,preparation_output=out,quiet_runs=[[x.start,x.end] for x in runs],grip_events=grip_events))
Path('/tmp/all-close-boundaries.json').write_text(json.dumps(rows,indent=2))
for x in rows:
 if x['closing']-x['gate']>15:print(x['source'],x['output'],x['gate'],x['preparation_output'],x['closing'],x['grip_events'])
