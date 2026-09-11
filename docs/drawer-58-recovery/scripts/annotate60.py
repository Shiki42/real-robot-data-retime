import json,shutil,hashlib
from pathlib import Path
r=Path('/home/coder/share/drawer-58-recovery-20260911');c=json.loads((r/'config-v1.json').read_text());source=Path(c['analyses']['60']);target=r/'analysis/episode_060'
shutil.copytree(source,target)
p=target/'interaction_timeline.json';timeline=json.loads(p.read_text());event=timeline['episodes'][0];assert event['approach_start']==323
event['approach_start']=196
annotation=dict(source_episode=60,field='episodes[0].approach_start',old=323,new=196,reason='Last stationary interval was already at the cube; visual stop incorrectly discarded the approach and pre-closure aperture. State/action show first sustained left motion at edge 196 after idle, visible approach through frames 240-290, closure 315-320, visual pickup 332.',evidence=['source60-grasp.jpg','raw and trimmed episode 060 state/action'],source_timeline_sha256=hashlib.sha256((source/'interaction_timeline.json').read_bytes()).hexdigest(),unchanged=['source video','source telemetry','pickup 332','release 462','object identity','segmentation','scene geometry'],validation='pending plan/render; not counted as success')
p.write_text(json.dumps(timeline,indent=2));(target/'manual-annotation.json').write_text(json.dumps(annotation,indent=2));(r/'manual-interventions.json').write_text(json.dumps(dict(limit=5,used=1,entries=[annotation]),indent=2));c['analyses']['60']=str(target);c['work']=str(r/'manual60-work');c['output']=str(r/'manual60-dataset');(r/'manual60-config.json').write_text(json.dumps(c,indent=2))
