from pathlib import Path
import json,hashlib,time,subprocess
from real_robot_data_retime.automatic_validation import validate_pending_episode
import real_robot_data_retime.automatic_validation as validator
r=Path('/home/coder/share/retime-interaction-20260909');out=r/'numeric-b054795';out.mkdir(exist_ok=True);vhash=hashlib.sha256(Path(validator.__file__).read_bytes()).hexdigest()
while not (out/'STOP-v2').exists():
 for p in sorted((r/'release-b054795/claims').glob('*/status.json')):
  status=json.loads(p.read_text());ep=status['episode'];done=out/f'episode_{ep:03d}.json'
  if status['status']!='complete':continue
  receipt=r/f'drawer-retimed-b054795/meta/retime_receipts/episode_{ep:03d}.json'
  digest=hashlib.sha256(receipt.read_bytes()).hexdigest()
  if done.exists() and json.loads(done.read_text())['receipt_sha256']==digest:continue
  result=validate_pending_episode(r/'drawer-trimmed',r/'drawer-retimed-b054795',ep)
  subprocess.run(['/home/coder/share/real-robot-data-retime/.venv/bin/python',str(r/'review_b054_sheet.py'),str(ep)],check=True)
  result.update(passed=True,validation_sha256=vhash,receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest())
  done.write_text(json.dumps(result,indent=2));print(ep,result['frames'],'passed',flush=True)
 if len(list(out.glob('episode_*.json')))==87:break
 time.sleep(10)
