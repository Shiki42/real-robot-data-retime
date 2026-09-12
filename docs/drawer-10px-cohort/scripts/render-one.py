import json,sys,time,runpy,cv2
from pathlib import Path
from real_robot_data_retime.uniform_drawer import render_episode
cv2.setNumThreads(2)
r=Path('/home/coder/share/drawer-10px-cohort-20260912');c=json.loads((r/'config.json').read_text());ep=int(sys.argv[1]);start=time.monotonic();render_episode(c,ep);v=runpy.run_path(str(r/'validate-output.py'));v['validate'](ep,c);print('VERIFIED',ep,time.monotonic()-start,flush=True)
