import json,cv2,numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
R=Path('/home/coder/share/drawer-stable-cohort-20260912');C=json.loads(Path('/home/coder/share/drawer-uniform-dataset-20260912/final-config.json').read_text());K=json.loads((R/'clear-marker.json').read_text())
def run(ep):
 cv2.setNumThreads(1);screen=json.loads((R/'per-source'/f'{ep:03d}.json').read_text());a=Path(C['analyses'][str(ep)]);seg=np.load(a/'segmentation.npz');sw=int(seg['frame_shape'][1]);robots=np.unpackbits(seg['robots'],axis=-1,count=sw).any(axis=1);tr=np.load(a/'tracks.npz')['registration'];cap=cv2.VideoCapture(C['source']+f'/videos/observation.images.top/chunk-000/file-{ep:03d}.mp4');n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));ok,first=cap.read();assert ok;first=cv2.resize(first,(640,362));x,y,w,h=screen['reference_box'];x0=max(0,x-12);y0=max(0,y-12);x1=min(640,x+w+12);y1=min(362,y+h+12);template=cv2.cvtColor(first,cv2.COLOR_BGR2GRAY)[y0:y1,x0:x1].astype(np.float32)/255;targets={n-1}
 if K[ep]['samples']:targets.add(max(K[ep]['samples'],key=lambda s:s['pixels'])['frame'])
 results=[]
 for t in sorted(targets):
  cap.set(cv2.CAP_PROP_POS_FRAMES,t);ok,im=cap.read();assert ok;im=cv2.resize(im,(640,362));gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)[y0:y1,x0:x1].astype(np.float32)/255;mask=cv2.resize(robots[t].astype(np.uint8),(640,362),interpolation=cv2.INTER_NEAREST);mask=1-cv2.dilate(mask,np.ones((7,7),np.uint8));mask=mask[y0:y1,x0:x1]*255;warp=np.eye(2,3,dtype=np.float32)
  try:score,warp=cv2.findTransformECC(template,gray,warp,cv2.MOTION_AFFINE,(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,150,1e-6),mask,5)
  except cv2.error as error:
   results.append(dict(frame=t,fit_error=str(error)));continue
  points=np.array([[x-x0,y-y0],[x+w-x0,y-y0],[x+w-x0,y+h-y0],[x-x0,y+h-y0]],float);delta=points@warp[:,:2].T+warp[:,2]-points;s=next((v for v in screen['samples'] if v['frame']==t),None);bg=None if s is None else s['camera_shift'];bg=np.array(bg) if bg is not None else -tr[t,:2,2]*640/sw;delta-=bg
  results.append(dict(frame=t,score=float(score),corner_rms=float(np.sqrt(np.mean(np.sum(delta**2,axis=1)))),center_shift=np.mean(delta,axis=0).tolist(),warp=warp.tolist(),background_method='independent_features' if s is not None and s['camera_shift'] is not None else 'recorded_registration'))
 cap.release();return dict(source=ep,results=results)
if __name__=='__main__':
 rows=[]
 with ProcessPoolExecutor(max_workers=4) as p:
  for f in as_completed([p.submit(run,i) for i in range(87)]):
   row=f.result();rows.append(row);(R/'pose-check.json').write_text(json.dumps(sorted(rows,key=lambda x:x['source']),indent=2));print(row['source'],[(x['frame'],round(x.get('score',0),3),round(x.get('corner_rms',0),2)) for x in row['results']],flush=True)
