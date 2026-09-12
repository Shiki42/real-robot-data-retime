"""Render source-stage evidence for the committed boundary table."""
import json,cv2,numpy as np
from pathlib import Path
root=Path('/home/coder/share/drawer-10px-cohort-20260912');d=Path('docs/drawer-10px-cohort/review');rows=list(json.loads((d/'close-preparation-boundaries.json').read_text())['sources'].items());out=d/'phase-evidence';out.mkdir(exist_ok=True)
for start in range(0,len(rows),4):
 tiles=[]
 for source,row in rows[start:start+4]:
  ep=row['reviewed_output'];gate=row['withdrawal_output'];close=row['closing_output'];m=np.load(root/f'dataset/meta/retime_source_indices/episode_{ep:03d}.npz');prep=max(gate,int(np.searchsorted(m['right'],row['close_preparation_source_frame'])));frames=[gate,(gate+prep)//2,prep,close];cap=cv2.VideoCapture(str(root/f'local-review-package/review/videos/episode_{ep:03d}.mp4'));line=[]
  for label,frame in zip(['gate','wait','prep','close'],frames):
   cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,img=cap.read();assert ok;tile=np.full((201,320,3),255,np.uint8);tile[20:]=cv2.resize(img,(320,181));cv2.putText(tile,f'src{source} out{ep} f{frame} {label}',(3,14),0,.4,(0,0,0),1);line.append(tile)
  cap.release();tiles.append(np.hstack(line))
 cv2.imwrite(str(out/f'page-{start//4:02d}.jpg'),np.vstack(tiles))
print('Evidence pages',len(list(out.glob('*.jpg'))))
