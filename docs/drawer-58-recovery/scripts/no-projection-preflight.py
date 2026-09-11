import json,time,traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
from real_robot_data_retime.uniform_drawer import plan_episode,producer_fingerprint
R=Path('/home/coder/share/drawer-58-no-projection-20260912')
def run(ep):
 cfg=json.loads((R/'config.json').read_text());start=time.monotonic()
 try:
  plan_episode(cfg,ep)
  result=dict(episode=ep,passed=True)
 except ValueError as error:
  result=dict(episode=ep,passed=False,error_type=type(error).__name__,reason=str(error))
  (R/f'error-{ep:03d}.log').write_text(traceback.format_exc())
 result['seconds']=time.monotonic()-start
 path=Path(cfg['work'])/f'episode_{ep:03d}/feasibility.json'
 if path.exists():result['feasibility']=json.loads(path.read_text())
 return result
if __name__=='__main__':
 if (R/'preflight.json').exists():raise FileExistsError('Existing cohort must not be overwritten')
 rows=[]
 with ProcessPoolExecutor(max_workers=3) as pool:
  for future in as_completed([pool.submit(run,ep) for ep in range(87)]):
   row=future.result();rows.append(row)
   (R/'preflight.json').write_text(json.dumps(dict(producer=producer_fingerprint(),scope='Planning ablation only: both 2D rejection sites removed; all other existing checks retained; not rendered or fully collision validated',results=sorted(rows,key=lambda x:x['episode'])),indent=2))
   print(row['episode'],row['passed'],row.get('reason',''),flush=True)
