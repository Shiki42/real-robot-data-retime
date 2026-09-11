import json,subprocess,sys,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
r=Path('/home/coder/share/drawer-uniform-dataset-20260912');logs=r/'generation-logs';logs.mkdir(exist_ok=False)
def run(ep):
 start=time.monotonic()
 with (logs/f'episode-{ep:03d}.log').open('w') as log:
  result=subprocess.run([sys.executable,str(r/'generate-one.py'),str(ep)],stdout=log,stderr=subprocess.STDOUT)
 return dict(source=ep,passed=result.returncode==0,exit_code=result.returncode,seconds=time.monotonic()-start)
rows=[]
with ThreadPoolExecutor(max_workers=4) as pool:
 for future in as_completed([pool.submit(run,ep) for ep in range(87)]):
  row=future.result();rows.append(row);tmp=r/'generation-status.tmp.json';tmp.write_text(json.dumps(sorted(rows,key=lambda x:x['source']),indent=2));tmp.replace(r/'generation-status.json');print(row,flush=True)
