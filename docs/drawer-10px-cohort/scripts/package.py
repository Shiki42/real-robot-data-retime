import json,shutil,hashlib,tarfile,csv
from pathlib import Path
R=Path('/home/coder/share/drawer-10px-cohort-20260912');out=R/'dataset';dest=R/'local-review-package/review';m=json.loads((dest/'manifest.json').read_text());s=json.loads((R/'selection.json').read_text());assert m['episodes']==156 and m['frames']==83187;assert sum(x['restored'] for x in m['episodes_data'])==38
restored=[x for x in m['episodes_data'] if x['restored']];(dest/'restored-episodes.json').write_text(json.dumps(restored,indent=2))
for name in ['trim-summary.json','timing-summary.json','excluded-sources.csv']:
 shutil.copy2(R/name,out/name);shutil.copy2(R/name,dest/name)
manual=json.loads(Path('/home/coder/share/drawer-stable-cohort-20260912/dataset/manual-interventions.json').read_text());(out/'manual-interventions.json').write_text(json.dumps(manual,indent=2))
(dest/'README.txt').write_text('156 episodes, 78 sources, 38 restored outputs.\nScreen: background-relative marker displacement P90 <=10px at 640px analysis width.\n30 FPS, all 83187 frames preserved.\nUse the next-restored button to review added samples.\nAction/State and three-camera full dataset remain on Coder A.\nThis review has independent localStorage; previous review notes are preserved.\n')
for root in [out,dest]:
 files=sorted(p for p in root.rglob('*') if p.is_file() and p.name!='SHA256SUMS');(root/'SHA256SUMS').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(root))+'\n' for p in files))
archive=dest.parent/'review.tar'
with tarfile.open(archive,'w') as tar:tar.add(dest,arcname='review')
report=dict(episodes=156,sources=78,restored_outputs=38,frames=83187,review_bytes=m['bytes'],archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),dataset=str(out),review_archive=str(archive))
(R/'delivery.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
