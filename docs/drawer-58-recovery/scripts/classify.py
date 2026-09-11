import json
from pathlib import Path
import numpy as np
from real_robot_data_retime.staged import load_joints
from real_robot_data_retime.collision.piperx import PiperXClearance
from real_robot_data_retime.collision.drawer import DrawerGeometry, drawer_sweep, outside_box
from real_robot_data_retime.timeline.smooth import held_grasp_interval
from real_robot_data_retime.timeline.drawer_wait import opening_wait_candidates
from real_robot_data_retime.compositing.ownership import exclude_placed_objects
root=Path('/home/coder/share/drawer-58-recovery-20260911')
cfg=json.loads(Path('/home/coder/share/drawer-uniform-fix-20260911/main-config.json').read_text())
cfg.update(work=str(root/'work-v1'),output=str(root/'dataset-v1'),urdf='/home/coder/share/real-robot-data-retime-drawer-58-recovery/assets/piper_x_description.urdf')
(root/'config-v1.json').write_text(json.dumps(cfg,indent=2))
if (root/'classification.json').exists():
 raise FileExistsError('Use a new output root; existing evidence must not be overwritten')
old=json.loads(Path('/home/coder/share/drawer-uniform-fix-20260911/main-preflight.json').read_text())
rows=[]
for previous in old['results']:
 ep=previous['episode']; analysis=Path(cfg['analyses'][str(ep)])
 timeline=json.loads((analysis/'interaction_timeline.json').read_text());event=timeline['episodes'][0];motion=timeline['drawer_motion']
 state,action,*_=load_joints(Path(cfg['source'])/f'data/chunk-000/file-{ep:03d}.parquet',cfg['urdf'],cfg['mesh_root'],timeline)
 row=dict(episode=ep,previous_passed=previous['passed'],event=event,motion=motion)
 try: grasp,end,aperture=held_grasp_interval(state,action,event,30)
 except ValueError as error:
  row['holding_error']=str(error);rows.append(row);continue
 checker=PiperXClearance(state[:,:7],state[:,7:],Path(cfg['urdf']),Path(cfg['mesh_root']),margin_m=.005)
 tcp=np.array([[pose[4] for pose in arm] for arm in checker.poses]);geometry=DrawerGeometry.from_mapping(cfg['scene_geometry'])
 volume=drawer_sweep(tcp[1],motion['pull_start'],motion['open_frame'],**geometry.box_kwargs)
 peak=grasp+int(np.argmax(tcp[0,grasp:end,2]));peakz=float(tcp[0,peak,2])
 ids=np.arange(len(state));band=(ids>=grasp)&(ids<end)&(tcp[0,:,2]>=peakz-.002)
 rows_at_peak=[];eligible=np.zeros(len(state),bool)
 for frame in np.flatnonzero(band):
  t=int(frame);commanded=checker._pose(action[t,:7],0)
  checks=dict(ramp_support=t-8>=event['approach_start'] and t+4<end,aperture=bool(max(state[t,6],action[t,6])<=aperture),state_object=outside_box(tcp[0,t],volume,radius=geometry.held_object_radius_m),state_arm=checker.arm_clears_volume(0,t,volume,margin=.005),action_object=outside_box(commanded[4],volume,radius=geometry.held_object_radius_m),action_arm=checker.pose_clears_volume(commanded,volume,margin=.005))
  eligible[t]=all(checks.values());rows_at_peak.append(dict(frame=t,z=float(tcp[0,t,2]),checks=checks))
 with np.load(analysis/'segmentation.npz') as seg:
  w=int(seg['frame_shape'][1]);robots=np.unpackbits(seg['robots'],axis=-1,count=w).astype(bool);objects=np.unpackbits(seg['objects'],axis=-1,count=w).astype(bool)
 exclude_placed_objects(robots,objects,timeline['episodes'])
 image_allowed=opening_wait_candidates(band,robots,objects,timeline['episodes'],event['approach_start'],motion['open_frame'],ep/174,30)
 for record in rows_at_peak:record['checks']['opening_image']=bool(image_allowed[record['frame']])
 row.update(grasp=grasp,held_end=end,aperture=aperture,peak=peak,peak_z=peakz,candidates=rows_at_peak,eligible_after_geometry=np.flatnonzero(eligible).tolist(),eligible_after_image=np.flatnonzero(eligible&image_allowed).tolist())
 if not previous['passed'] and previous.get('diagnostic'):row['schedule']=json.loads(Path(previous['diagnostic']).read_text())
 rows.append(row)
 (root/'classification.json').write_text(json.dumps(rows,indent=2))
 print(ep,previous['passed'],grasp,end,peak,'near',len(rows_at_peak),'geometry',int(eligible.sum()),'image',int((eligible&image_allowed).sum()),flush=True)
