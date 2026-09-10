import sys,json,cv2,numpy as np,hashlib
from pathlib import Path
r=Path('/home/coder/share/retime-interaction-20260909');ep=int(sys.argv[1]);out=r/'drawer-retimed-b054795';receipt_path=out/f'meta/retime_receipts/episode_{ep:03d}.json';receipt=json.loads(receipt_path.read_text());timeline=receipt['interaction']['timeline'];event=timeline['episodes'][0];motion=timeline['drawer_motion'];deps=receipt['plan']['dependencies'];maps=np.load(out/f'meta/retime_source_indices/episode_{ep:03d}.npz');left,right=maps['left'],maps['right'];n=len(left)
def mapped(clock,t):return min(n-1,int(np.searchsorted(clock,t)))
pickup=mapped(left,event['pickup_frame']);release=mapped(left,event['release_frame']);close=mapped(right,motion['close_start']);ids=[0,max(0,pickup-8),pickup,min(n-1,pickup+12),mapped(right,motion['open_frame']),mapped(left,deps['safe_wait_frame']+1),max(0,release-8),min(n-1,release+12),mapped(left,deps['withdrawal_frame']),min(n-1,close+8),min(n-1,close+35),receipt['plan']['synthetic_terminal_hold']['start_frame']-1];ids=sorted(ids)
video=out/receipt['compositing']['output'];cap=cv2.VideoCapture(str(video));tiles=[]
for t in ids:
 cap.set(cv2.CAP_PROP_POS_FRAMES,t);ok,frame=cap.read()
 if not ok:raise ValueError(f'cannot decode episode{ep} frame{t}')
 frame=cv2.resize(frame,(424,240));cv2.rectangle(frame,(0,0),(424,24),(0,0,0),-1);cv2.putText(frame,f'EP{ep} out{t} L{left[t]} R{right[t]}',(5,17),0,.5,(255,255,255),1);tiles.append(frame)
cap.release();folder=r/'visual-review-b054795';folder.mkdir(exist_ok=True);cv2.imwrite(str(folder/f'episode_{ep:03d}.jpg'),np.concatenate([np.concatenate(tiles[i:i+4],axis=1) for i in range(0,12,4)]),[cv2.IMWRITE_JPEG_QUALITY,80]);(folder/f'episode_{ep:03d}_sampling.json').write_text(json.dumps(dict(episode=ep,output_frames=ids,receipt_sha256=hashlib.sha256(receipt_path.read_bytes()).hexdigest(),video_sha256=hashlib.sha256(video.read_bytes()).hexdigest()),indent=2))
