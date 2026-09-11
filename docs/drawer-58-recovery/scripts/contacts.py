import cv2,json,numpy as np
from pathlib import Path
r=Path('/home/coder/share/drawer-58-recovery-20260911');cfg=json.loads((r/'config-v1.json').read_text());rows=json.loads((r/'classification.json').read_text());panels=[]
if (r/'source-contacts.jpg').exists():
 raise FileExistsError('Use a new output root; existing evidence must not be overwritten')
for ep in [0,7,14,18,22,60]:
 row=next((x for x in rows if x['episode']==ep),None)
 if row is None:continue
 e=row['event'];m=row['motion'];frames=[e['pickup_frame'],row.get('peak',e['grasp_frame']),m['pull_start'],m['open_frame']]
 cap=cv2.VideoCapture(str(Path(cfg['source'])/f'videos/observation.images.top/chunk-000/file-{ep:03d}.mp4'));tiles=[]
 for f in frames:
  cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,img=cap.read();assert ok
  img=cv2.resize(img,(424,240));cv2.putText(img,f'ep {ep} frame {f}',(8,22),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,255),2);tiles.append(img)
 cap.release();panels.append(np.concatenate(tiles,axis=1))
cv2.imwrite(str(r/'source-contacts.jpg'),np.concatenate(panels,axis=0))
