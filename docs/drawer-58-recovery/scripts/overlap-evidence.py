import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from real_robot_data_retime.compositing.depth import AlignedDepth
from real_robot_data_retime.compositing.ownership import arm_foreground,exclude_placed_objects
from real_robot_data_retime.staged import load_joints
from real_robot_data_retime.collision.piperx import PiperXClearance
r=Path('/home/coder/share/drawer-58-recovery-20260911');cfg=json.loads((r/'config-v1.json').read_text());rows=json.loads((r/'classification.json').read_text());trim=json.loads((Path(cfg['source'])/'trim_manifest.json').read_text())['episodes'];out=[]
if (r/'overlap-evidence.json').exists():
 raise FileExistsError('Use a new output root; existing evidence must not be overwritten')
for ep in [7,14,18,22,32,44,60]:
 row=rows[ep];ana=Path(cfg['analyses'][str(ep)]);tl=json.loads((ana/'interaction_timeline.json').read_text());e=tl['episodes'][0]
 if 'peak' not in row:continue
 peak=row['peak'];s,a,*_=load_joints(Path(cfg['source'])/f'data/chunk-000/file-{ep:03d}.parquet',cfg['urdf'],cfg['mesh_root'],tl);c=PiperXClearance(s[:,:7],s[:,7:],Path(cfg['urdf']),Path(cfg['mesh_root']),margin_m=.005)
 with np.load(ana/'segmentation.npz') as seg,np.load(ana/'tracks.npz') as tr:
  shape=tuple(seg['frame_shape']);w=int(shape[1]);robots=np.unpackbits(seg['robots'],axis=-1,count=w).astype(bool);objects=np.unpackbits(seg['objects'],axis=-1,count=w).astype(bool);trans=tr['registration'].copy()
 exclude_placed_objects(robots,objects,tl['episodes']);refs=pq.read_table(Path(cfg['raw_source'])/f'data/chunk-000/file-{ep:03d}.parquet',columns=['observation.depth.top'])['observation.depth.top'].to_pylist()[trim[ep]['start']:trim[ep]['stop']];depth=AlignedDepth(cfg['raw_source'],refs,trans,shape)
 lm=arm_foreground(robots,objects,tl['episodes'],0,peak);z=depth(peak);records=[]
 for t in range(tl['drawer_motion']['open_frame']+1):
  overlap=lm & robots[t,1]
  if not overlap.any():continue
  rz=depth(t);valid=overlap&(z>0)&(rz>0);gap=np.abs(z.astype(float)-rz.astype(float))[valid]
  records.append(dict(right=t,pixels=int(overlap.sum()),valid=int(valid.sum()),depth_gap_percentiles_mm=np.percentile(gap,[0,5,50,95,100]).tolist() if len(gap) else [],mesh_clear=c.configuration_safe(peak,t)))
 depth.close();out.append(dict(episode=ep,peak=peak,overlaps=records));(r/'overlap-evidence.json').write_text(json.dumps(out,indent=2));print(ep,len(records),'mesh failures',sum(not v['mesh_clear'] for v in records),'max overlap',max([v['pixels'] for v in records],default=0),flush=True)
