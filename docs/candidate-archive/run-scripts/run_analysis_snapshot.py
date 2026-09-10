import os,sys,json,subprocess,shutil,fcntl,hashlib
from pathlib import Path
from real_robot_data_retime.interaction.measurements import producer_fingerprint
r=Path('/home/coder/share/retime-interaction-20260909');ep=int(sys.argv[1]);fingerprint=producer_fingerprint();snapshot=r/'experiments'/fingerprint;source=Path('/home/coder/share/real-robot-data-retime/src/real_robot_data_retime');cache=Path(sys.argv[2]) if len(sys.argv)>2 else r/f'final-analysis/episode_{ep:03d}';output=r/(sys.argv[3] if len(sys.argv)>3 else 'release-corrected-analysis')/f'episode_{ep:03d}'
output.parent.mkdir(parents=True,exist_ok=True)
episode_lock=(output.parent/f'.{ep:03d}.analysis.lock').open('a')
fcntl.flock(episode_lock,fcntl.LOCK_EX)
status_file=output/'job-status.json'
if status_file.exists():
 status=json.loads(status_file.read_text());sys.exit(status['exit_code'])
(r/'experiments').mkdir(exist_ok=True)
with (r/'experiments/snapshot.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 if not snapshot.exists():
  staging=r/'experiments'/f'.{fingerprint}.{os.getpid()}'
  shutil.copytree(source,staging/'src/real_robot_data_retime',ignore=shutil.ignore_patterns('__pycache__'))
  tree=staging/'src/real_robot_data_retime';digest=hashlib.sha256()
  for path in sorted(tree.rglob('*.py')):digest.update(path.relative_to(tree).as_posix().encode());digest.update(path.read_bytes())
  if digest.hexdigest()!=fingerprint:raise RuntimeError('source changed while snapshotting')
  staging.rename(snapshot)
code="import sys; from pathlib import Path; from real_robot_data_retime.interaction.pipeline import run; r=Path('/home/coder/share/retime-interaction-20260909'); ep=int(sys.argv[1]); print(run(r/f'drawer-trimmed/videos/observation.images.top/chunk-000/file-{ep:03d}.mp4',Path(sys.argv[3]),'drawer',reuse_measurements=Path(sys.argv[2])))"
env=dict(os.environ,PYTHONPATH=str(snapshot/'src')+':/home/coder/share/robo-visualize/src');print('snapshot',fingerprint,flush=True);result=subprocess.run(['/home/coder/share/real-robot-data-retime/.venv/bin/python','-c',code,str(ep),str(cache),str(output)],env=env);
report=json.loads((output/'report.json').read_text()) if (output/'report.json').exists() else {}
output.mkdir(exist_ok=True)
(output/'job-status.json').write_text(json.dumps(dict(exit_code=result.returncode,success=result.returncode==0 and report.get('success',False),fingerprint=fingerprint)))
sys.exit(result.returncode)
