import json,cv2,numpy as np,pyarrow.parquet as pq
from pathlib import Path
from real_robot_data_retime.trim import analyze_dataset
r=Path('/home/coder/share/drawer-uniform-dataset-20260912');raw=Path('/home/coder/share/retime-interaction-20260909/drawer');trim=Path('/home/coder/share/retime-interaction-20260909/drawer-trimmed');report=analyze_dataset(raw,tail_seconds=1000000);old=json.loads((trim/'trim_manifest.json').read_text());info=json.loads((trim/'meta/info.json').read_text());results=[]
for a,b in zip(report['episodes'],old['episodes']):
 assert (a['start'],a['stop'])==(b['start'],b['stop']) and a['tail_removed']==0
 ep=a['episode_index'];src=pq.read_table(raw/f'data/chunk-000/file-{ep:03d}.parquet');dst=pq.read_table(trim/f'data/chunk-000/file-{ep:03d}.parquet');n=a['stop']-a['start'];assert len(dst)==n
 for key in ['action','observation.state']:assert src[key].slice(a['start'],n).equals(dst[key])
 assert dst['frame_index'].to_pylist()==list(range(n));assert np.allclose(dst['timestamp'].to_numpy(),np.arange(n)/30,atol=1e-5)
 cameras={}
 for cam in ['top','left_wrist','right_wrist']:
  cap=cv2.VideoCapture(str(trim/f'videos/observation.images.{cam}/chunk-000/file-{ep:03d}.mp4'));assert cap.get(cv2.CAP_PROP_FRAME_COUNT)==n and cap.get(cv2.CAP_PROP_FPS)==30;count=0
  while True:
   ok,frame=cap.read()
   if not ok:break
   count+=1
  cap.release();assert count==n;cameras[cam]=count
 results.append(dict(episode=ep,head_removed=a['head_removed'],tail_removed=0,frames=n,numeric_exact=True,decoded_camera_frames=cameras));print(ep,n,flush=True)
report.update(head_removed=sum(x['head_removed'] for x in report['episodes']),tail_removed=0,output_frames=sum(x['frames'] for x in results),reused_trim=str(trim),verified=results);(r/'trim-verification.json').write_text(json.dumps(report,indent=2));assert report['head_removed']==667 and report['output_frames']==53796
