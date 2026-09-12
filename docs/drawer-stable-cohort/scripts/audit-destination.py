import json,cv2,numpy as np,sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from real_robot_data_retime.tasks.drawer_constraints import discover_drawer_interior
from real_robot_data_retime.background.clean_plate import dilate
R=Path('/home/coder/share/drawer-stable-cohort-20260912');C=json.loads((R/'config.json').read_text())
def run(ep):
 cv2.setNumThreads(1);a=Path(C['analyses'][str(ep)]);timeline=json.loads((a/'interaction_timeline.json').read_text());event=timeline['episodes'][0];seg=np.load(a/'segmentation.npz');sw=int(seg['frame_shape'][1]);robots=np.unpackbits(seg['robots'],axis=-1,count=sw)[:,0].astype(bool);obj=np.unpackbits(seg['objects'][event['object_id']],axis=-1,count=sw).astype(bool);registration=np.load(a/'tracks.npz')['registration'];video=Path(C['source'])/f'videos/observation.images.top/chunk-000/file-{ep:03d}.mp4';cap=cv2.VideoCapture(str(video));images=[]
 for t in [0,timeline['drawer_motion']['open_frame']]:
  cap.set(cv2.CAP_PROP_POS_FRAMES,t);ok,img=cap.read();assert ok;img=cv2.resize(img,(sw,int(seg['frame_shape'][0])));img=cv2.warpAffine(img,registration[t,:2],(sw,int(seg['frame_shape'][0])),borderMode=cv2.BORDER_REFLECT);images.append(img)
 cap.release();interior=discover_drawer_interior(images[1]);hsv=cv2.cvtColor(images[0],cv2.COLOR_BGR2HSV);pixels=hsv[obj[0]];hue=float(np.median(pixels[pixels[:,1]>85,0]));threshold=max(30,int(obj[0].sum()/8));result=[]
 for v in range(2):
  target=C['source_indices'].index(ep)+v*len(C['source_indices']);m=np.load(Path(C['output'])/f'meta/retime_source_indices/episode_{target:03d}.npz');cap=cv2.VideoCapture(str(Path(C['output'])/f'videos/observation.images.top/chunk-000/file-{target:03d}.mp4'));hits=[];checked=0
  for frame,l in enumerate(m['left']):
   ok,img=cap.read();assert ok
   if not event['pickup_frame']<=l<event['release_frame']:continue
   img=cv2.resize(img,(sw,int(seg['frame_shape'][0])));hs=cv2.cvtColor(img,cv2.COLOR_BGR2HSV);delta=np.abs(hs[:,:,0].astype(float)-hue);blue=(np.minimum(delta,180-delta)<25)&(hs[:,:,1]>85)&(hs[:,:,2]>30);lo,hi=int(np.floor(l)),int(np.ceil(l));allowed=dilate(robots[lo]|robots[hi]|obj[lo]|obj[hi],8);suspect=(blue&interior&~allowed).astype(np.uint8);nc,labels,stats,_=cv2.connectedComponentsWithStats(suspect,8);largest=int(stats[1:,4].max(initial=0));checked+=1
   if largest>=threshold:hits.append(dict(output_frame=frame,left_source=float(l),area=largest))
  cap.release();result.append(dict(output=target,checked_frames=checked,threshold_pixels=threshold,hits=hits))
 row=dict(source=ep,results=result);(R/'destination-audit').mkdir(exist_ok=True);(R/'destination-audit'/f'{ep:03d}.json').write_text(json.dumps(row,indent=2));print(ep,[len(x['hits']) for x in result],flush=True);return row
if __name__=='__main__':
 if len(sys.argv)>1:run(int(sys.argv[1]))
 else:
  with ProcessPoolExecutor(max_workers=4) as p:rows=list(p.map(run,C['source_indices']))
  (R/'destination-audit.json').write_text(json.dumps(rows,indent=2))
