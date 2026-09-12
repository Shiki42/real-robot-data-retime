import json,cv2,numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
R=Path('/home/coder/share/drawer-stable-cohort-20260912')
CFG=json.loads(Path('/home/coder/share/drawer-uniform-dataset-20260912/final-config.json').read_text())
def run(ep):
 cv2.setNumThreads(1)
 a=Path(CFG['analyses'][str(ep)]);tl=json.loads((a/'interaction_timeline.json').read_text());seg=np.load(a/'segmentation.npz');sw=int(seg['frame_shape'][1]);robots=np.unpackbits(seg['robots'],axis=-1,count=sw).any(axis=1);tracks=np.load(a/'tracks.npz');trans=tracks['registration'];cap=cv2.VideoCapture(CFG['source']+f'/videos/observation.images.top/chunk-000/file-{ep:03d}.mp4');n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));orb=cv2.ORB_create(nfeatures=1400,fastThreshold=8,edgeThreshold=8);matcher=cv2.BFMatcher(cv2.NORM_HAMMING);samples=[];selected={};ref=None
 for t in range(n):
  ok,im=cap.read();assert ok
  if t%6 and t!=n-1:continue
  im=cv2.resize(im,(640,362));hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV);red=(((hsv[:,:,0]<12)|(hsv[:,:,0]>170))&(hsv[:,:,1]>120)&(hsv[:,:,2]>70)).astype(np.uint8);nc,labs,stats,centers=cv2.connectedComponentsWithStats(red,8);ids=[k for k in range(1,nc) if stats[k,4]>1500]
  if not ids:continue
  k=max(ids,key=lambda k:stats[k,4]);mask=labs==k;x,y,w,h,area=map(int,stats[k]);center=centers[k]
  if ref is None:
   ref=dict(center=center,area=area,box=[x,y,w,h]);selected[0]=im.copy();ignore=np.zeros((362,640),np.uint8);ignore[max(0,y-35):min(362,y+h+100),max(0,x-40):min(640,x+w+70)]=1
   ignore|=cv2.resize(robots[t].astype(np.uint8),(640,362),interpolation=cv2.INTER_NEAREST);ignore=cv2.dilate(ignore,np.ones((13,13),np.uint8));kp0,desc0=orb.detectAndCompute(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),(1-ignore)*255)
  if not .85<=area/ref['area']<=1.2:continue
  ignore=np.zeros((362,640),np.uint8);ignore[max(0,y-35):min(362,y+h+100),max(0,x-40):min(640,x+w+70)]=1;ignore|=cv2.resize(robots[t].astype(np.uint8),(640,362),interpolation=cv2.INTER_NEAREST);ignore=cv2.dilate(ignore,np.ones((13,13),np.uint8));kp,desc=orb.detectAndCompute(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),(1-ignore)*255)
  shift=None;inliers=0
  if desc is not None and desc0 is not None and len(desc)>=2:
   pairs=[]
   for match in matcher.knnMatch(desc0,desc,k=2):
    if len(match)==2 and match[0].distance<60 and match[0].distance<.75*match[1].distance:pairs.append((kp0[match[0].queryIdx].pt,kp[match[0].trainIdx].pt))
   if len(pairs)>=8:
    pairs=np.array(pairs);delta=pairs[:,1]-pairs[:,0];median=np.median(delta,axis=0);valid=np.linalg.norm(delta-median,axis=1)<2;inliers=int(valid.sum())
    if inliers>=8:shift=np.median(delta[valid],axis=0)
  registered=center-ref['center']+trans[t,:2,2]*640/sw
  relative=None if shift is None else center-ref['center']-shift
  row=dict(frame=t,area_ratio=area/ref['area'],box=[x,y,w,h],camera_shift=None if shift is None else shift.tolist(),background_inliers=inliers,relative_shift=None if relative is None else relative.tolist(),relative_pixels=None if relative is None else float(np.linalg.norm(relative)),registered_pixels=float(np.linalg.norm(registered)))
  samples.append(row);selected[t]=im.copy()
 cap.release();valid=[x for x in samples if x['relative_pixels'] is not None];ranked=sorted(samples,key=lambda x:x['relative_pixels'] if x['relative_pixels'] is not None else x['registered_pixels'],reverse=True);worst=ranked[0];last=samples[-1];frames=[0,worst['frame'],last['frame']];tiles=[]
 for t in frames:
  im=selected[t].copy();cv2.putText(im,f'source {ep} frame {t}',(5,22),0,.65,(0,0,255),2);tiles.append(cv2.resize(im,(320,181)))
 (R/'contacts').mkdir(exist_ok=True);cv2.imwrite(str(R/'contacts'/f'source-{ep:03d}.jpg'),np.hstack(tiles),[cv2.IMWRITE_JPEG_QUALITY,70])
 result=dict(source=ep,reference_box=ref['box'],valid_samples=len(valid),sample_count=len(samples),max_relative_pixels=max([x['relative_pixels'] for x in valid],default=None),p90_relative_pixels=float(np.percentile([x['relative_pixels'] for x in valid],90)) if valid else None,max_registered_pixels=max(x['registered_pixels'] for x in samples),worst=worst,last=last,samples=samples)
 (R/'per-source').mkdir(exist_ok=True);(R/'per-source'/f'{ep:03d}.json').write_text(json.dumps(result,indent=2));return {k:v for k,v in result.items() if k!='samples'}
if __name__=='__main__':
 rows=[]
 with ProcessPoolExecutor(max_workers=4) as pool:
  for f in as_completed([pool.submit(run,i) for i in range(87)]):
   row=f.result();rows.append(row);(R/'screening.json').write_text(json.dumps(sorted(rows,key=lambda x:x['source']),indent=2));print(row['source'],row['p90_relative_pixels'],row['max_relative_pixels'],row['max_registered_pixels'],flush=True)
