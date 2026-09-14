import json,hashlib,shutil,tarfile
from pathlib import Path
import numpy as np
r=Path('/home/coder/share/drawer-no-unneeded-stop-20260915');out=r/'dataset';d=r/'local-review-package/review';v=json.loads((r/'validation.json').read_text());m=json.loads((d/'manifest.json').read_text());masks=json.loads((d/'idle-masks.json').read_text());assert v['passed'] and m['episodes']==156 and len(masks['episodes'])==156;assert v['frames']==m['frames'];checks=[]
for row,mask in zip(m['episodes_data'],masks['episodes'],strict=True):
 ep=row['output'];rec=json.loads((out/f'meta/retime_receipts/episode_{ep:03d}.json').read_text());s=rec['plan']['stages'];maps=np.load(out/f'meta/retime_source_indices/episode_{ep:03d}.npz');left,right=maps['left'],maps['right'];assert row['frames']==mask['frames']==len(left);ids=np.flatnonzero(left==s['peak_source_frame']);peak=int(ids[0]);smooth=s['smooth_stop_required']
 if not smooth:assert len(ids)==1 and np.all(np.diff(left)[peak-3:peak+3]==1) and right[peak]>=s['open_source_frame']
 assert len(mask['right_excess_wait'])<=1
 for arm in ['left','right']:
  idle=np.zeros(len(left),bool)
  for start,end in mask[arm]:idle[start:end]=True
  rest=mask[arm+'_terminal_rest'];assert idle[rest['mask_start']:].all() and (~idle[rest['start']:]).sum()<=45
 checks.append(dict(output=ep,source=row['source'],frames=len(left),uninterrupted_lift=not smooth,peak_frame=peak,opening_frame=rec['plan']['stage_validation']['drawer_open_frame']))
report=dict(passed=True,episodes=156,frames=v['frames'],uninterrupted_lift_outputs=sum(x['uninterrupted_lift'] for x in checks),mask_checks='all312 arm final-rest caps <=45 frames; right excess waits continuous',results=checks)
(r/'final-validation.json').write_text(json.dumps(report,indent=2))
for name in ['selection.json','trim-summary.json','final-validation.json']:shutil.copy2(r/name,out/name);shutil.copy2(r/name,d/name)
for name in ['idle-masks.json','close-preparation-boundaries.json']:shutil.copy2(d/name,out/name)
shutil.copy2('/home/coder/share/drawer-10px-cohort-20260912/dataset/manual-interventions.json',out/'manual-interventions.json')
(d/'README.txt').write_text('156 episodes: conditional lift stopping. Full dataset on Coder A. Independent review notes. All source maps, videos, Action/State and idle masks use the new output frame clocks. Original archives and reviews preserved. Numerical validation is not flawless visual-quality certification.\n')
for root in [out,d]:
 files=sorted(p for p in root.rglob('*') if p.is_file() and p.name!='SHA256SUMS');(root/'SHA256SUMS').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(root))+'\n' for p in files))
archive=d.parent/'review.tar'
with tarfile.open(archive,'w') as t:t.add(d,arcname='review')
delivery=dict(episodes=156,frames=v['frames'],uninterrupted_lift_outputs=report['uninterrupted_lift_outputs'],review_bytes=m['bytes'],archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),dataset=str(out),review_archive=str(archive));(r/'delivery.json').write_text(json.dumps(delivery,indent=2));print(json.dumps(delivery,indent=2))
