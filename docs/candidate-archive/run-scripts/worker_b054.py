from pathlib import Path
import json,subprocess,os,sys,time
r=Path('/home/coder/share/retime-interaction-20260909');worker=int(sys.argv[1]);batch=r/'release-b054795';claims=batch/'claims';claims.mkdir(parents=True,exist_ok=True);env=dict(os.environ,PYTHONPATH=str(r/'validation-b054795/src')+':/home/coder/share/robo-visualize/src')
def write(p,obj):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj));tmp.replace(p)
while not (batch/'STOP').exists():
 selected=None
 for ep in [1,4,49]+[i for i in range(87) if i not in [1,4,49]]:
  if (claims/f'{ep:03d}').exists():continue
  old=r/f'release-33aa255/claims/{ep:03d}/status.json'
  if old.exists() and json.loads(old.read_text())['status']=='running':continue
  if ep in json.loads((r/'pre9daf-episodes.json').read_text()) and not (r/'repaint-pre9daf.done').exists():continue
  paths=[r/f'{folder}/episode_{ep:03d}' for folder in ['release-earliest','release-surface','release-final-analysis','release-current']]
  valid=[p for p in paths if (p/'measurements.json').exists() and ((p/'job-status.json').exists() or (p/'analysis_identity.txt').exists())]
  if not valid:continue
  cache=valid[0]
  if (cache/'job-status.json').exists() and json.loads((cache/'job-status.json').read_text())['exit_code']!=0:continue
  claim=claims/f'{ep:03d}'
  try:claim.mkdir()
  except FileExistsError:continue
  selected=ep;break
 if selected is None:
  if len(list(claims.iterdir()))==87:break
  time.sleep(10);continue
 ep=selected;status=dict(episode=ep,status='running',version='b054795',worker=worker,cache=str(cache));write(claim/'status.json',status);start=time.time()
 with (batch/f'episode_{ep:03d}.log').open('w') as log:result=subprocess.run(['/home/coder/share/real-robot-data-retime/.venv/bin/python',str(r/'process_b054.py'),str(ep),str(cache)],env=env,stdout=log,stderr=subprocess.STDOUT)
 status.update(status='complete' if result.returncode==0 else 'failed',exit_code=result.returncode,seconds=time.time()-start);write(claim/'status.json',status);print(status,flush=True)
