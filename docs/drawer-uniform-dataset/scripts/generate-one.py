import json,sys,time,runpy
from pathlib import Path
from real_robot_data_retime.uniform_drawer import plan_episode,render_episode
r=Path('/home/coder/share/drawer-uniform-dataset-20260912');c=json.loads((r/'final-config.json').read_text());ep=int(sys.argv[1]);start=time.monotonic();plan_episode(c,ep);print('planned',ep,flush=True);render_episode(c,ep);print('rendered',ep,flush=True);v=runpy.run_path(str(r/'validate-output.py'));v['validate'](ep,c);print('VERIFIED',ep,time.monotonic()-start,flush=True)
