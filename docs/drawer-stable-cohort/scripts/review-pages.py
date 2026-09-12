import cv2,numpy as np,json
from pathlib import Path
r=Path('/home/coder/share/drawer-stable-cohort-20260912');ids=json.loads((r/'config.json').read_text())['source_indices'];p=r/'review'
for start in range(0,len(ids),4):
 group=ids[start:start+4];files=[p/f'source-{i:03d}.jpg' for i in group];out=p/f'page-{start//4:02d}.jpg'
 if all(f.exists() for f in files) and not out.exists():cv2.imwrite(str(out),np.vstack([cv2.imread(str(f)) for f in files]),[cv2.IMWRITE_JPEG_QUALITY,55]);print(out.name,group)
