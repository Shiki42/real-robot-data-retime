import cv2,numpy as np,json
from pathlib import Path
r=Path('/home/coder/share/drawer-10px-cohort-20260912');ids=json.loads((r/'selection.json').read_text())['restored_sources'];d=r/'restored-review';d.mkdir(exist_ok=True)
for i in range(0,len(ids),3):
 group=ids[i:i+3];files=[r/'review'/f'source-{ep:03d}.jpg' for ep in group]
 if all(p.exists() for p in files):cv2.imwrite(str(d/f'page-{i//3:02d}.jpg'),np.vstack([cv2.imread(str(p)) for p in files]))
print('pages',len(list(d.glob('*.jpg'))))
