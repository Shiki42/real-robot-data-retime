import json,cv2,numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
R=Path('/home/coder/share/drawer-stable-cohort-20260912');C=json.loads(Path('/home/coder/share/drawer-uniform-dataset-20260912/final-config.json').read_text())
def run(ep):
 cv2.setNumThreads(1);report=json.loads((R/'per-source'/f'{ep:03d}.json').read_text());samples={x['frame']:x for x in report['samples']};a=Path(C['analyses'][str(ep)]);seg=np.load(a/'segmentation.npz');sw=int(seg['frame_shape'][1]);robots=np.unpackbits(seg['robots'],axis=-1,count=sw).any(axis=1);cap=cv2.VideoCapture(C['source']+f'/videos/observation.images.top/chunk-000/file-{ep:03d}.mp4');n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));ref=None;rows=[]
 for t in range(n):
  ok,im=cap.read();assert ok
  if t not in samples:continue
  im=cv2.resize(im,(640,362));s=samples[t];x,y,w,h=s['box'];hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV);mask=((hsv[:,:,0]>=103)&(hsv[:,:,0]<=135)&(hsv[:,:,1]>100)&(hsv[:,:,2]>25)).astype(np.uint8);roi=np.zeros(mask.shape,np.uint8);roi[max(0,int(y+h*.45)):min(362,y+h+18),max(0,x-5):min(640,x+w+5)]=1;mask&=roi;nc,lab,stats,centers=cv2.connectedComponentsWithStats(mask,8);candidates=[k for k in range(1,nc) if 25<=stats[k,4]<=1200 and .5<=stats[k,2]/stats[k,3]<=2]
  if not candidates:continue
  if ref is None:
   k=max(candidates,key=lambda k:stats[k,4]);ref=dict(center=centers[k],area=int(stats[k,4]),offset=(centers[k]-[x,y])/[w,h]);score=0
  else:
   expected=np.array([x,y])+ref['offset']*[w,h];candidates=[k for k in candidates if .5<=stats[k,4]/ref['area']<=1.8 and np.linalg.norm(centers[k]-expected)<20]
   if not candidates:continue
   k=min(candidates,key=lambda k:np.linalg.norm(centers[k]-expected));score=float(np.linalg.norm(centers[k]-expected))
  overlap=float(cv2.resize(robots[t].astype(np.uint8),(640,362),interpolation=cv2.INTER_NEAREST)[lab==k].mean())
  if overlap>.25:continue
  shift=s['camera_shift'];delta=None if shift is None else centers[k]-ref['center']-shift
  rows.append(dict(frame=t,center=centers[k].tolist(),area=int(stats[k,4]),robot_overlap=overlap,relative=None if delta is None else delta.tolist(),pixels=None if delta is None else float(np.linalg.norm(delta)),background_inliers=s['background_inliers']))
 cap.release();valid=[x for x in rows if x['pixels'] is not None];result=dict(source=ep,count=len(valid),max_pixels=max([x['pixels'] for x in valid],default=None),p90_pixels=float(np.percentile([x['pixels'] for x in valid],90)) if valid else None,last=valid[-1] if valid else None,samples=rows);(R/'knob').mkdir(exist_ok=True);(R/'knob'/f'{ep:03d}.json').write_text(json.dumps(result,indent=2));return {k:v for k,v in result.items() if k!='samples'}
if __name__=='__main__':
 rows=[]
 with ProcessPoolExecutor(max_workers=4) as p:
  for f in as_completed([p.submit(run,i) for i in range(87)]):
   row=f.result();rows.append(row);(R/'knob-summary.json').write_text(json.dumps(sorted(rows,key=lambda x:x['source']),indent=2));print(row['source'],row['max_pixels'],row['p90_pixels'],flush=True)
