import cv2,numpy as np
from pathlib import Path
r=Path('/home/coder/share/drawer-uniform-dataset-20260912/review')
for start in range(0,87,4):
 files=[r/f'source-{i:03d}.jpg' for i in range(start,min(start+4,87))];out=r/f'page-{start//4:02d}.jpg'
 if all(p.exists() for p in files) and not out.exists():
  cv2.imwrite(str(out),np.vstack([cv2.imread(str(f)) for f in files]),[cv2.IMWRITE_JPEG_QUALITY,55]);print(out.name)
