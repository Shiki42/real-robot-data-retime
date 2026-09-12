import json,subprocess,sys,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
r=Path('/home/coder/share/drawer-10px-cohort-20260912');c=json.loads((r/'config.json').read_text());p=json.loads((r/'preflight.json').read_text());assert len(p['results'])==len(c['source_indices']) and all(x['passed'] for x in p['results']);logs=r/'render-logs';logs.mkdir(exist_ok=False)
def run(ep):
 start=time.monotonic()
 with (logs/f'source-{ep:03d}.log').open('w') as log:result=subprocess.run([sys.executable,str(r/'render-one.py'),str(ep)],stdout=log,stderr=subprocess.STDOUT)
 return dict(source=ep,passed=result.returncode==0,exit_code=result.returncode,seconds=time.monotonic()-start)
rows=[]
with ThreadPoolExecutor(max_workers=6) as pool:
 for f in as_completed([pool.submit(run,ep) for ep in c['source_indices']]):
  row=f.result();rows.append(row);tmp=r/'render-status.tmp.json';tmp.write_text(json.dumps(sorted(rows,key=lambda x:x['source']),indent=2));tmp.replace(r/'render-status.json');print(row,flush=True)
