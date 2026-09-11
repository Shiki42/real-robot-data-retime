import json,sys
from pathlib import Path
from real_robot_data_retime.timeline import visual
from real_robot_data_retime.timeline.scheduler import NoSafeSchedule
from real_robot_data_retime.uniform_drawer import plan_episode
root=Path('/home/coder/share/drawer-58-recovery-20260911');cfg=json.loads((root/'config-v1.json').read_text());cfg['work']=str(root/'schedule-probe');original=visual.schedule_sources
if (root/'schedule-evidence.json').exists():
 raise FileExistsError('Use a new output root; existing evidence must not be overwritten')
records=[]
def diagnose(n,m,safe,**kwargs):
 context=sys._getframe(1).f_back.f_locals
 try:return original(n,m,safe,**kwargs)
 except NoSafeSchedule as error:
  trace=error.__traceback__
  while trace.tb_next:trace=trace.tb_next
  costs=trace.tb_frame.f_locals.get('costs',{})
  src=context['sources'];points=sorted(costs,key=lambda p:sum(p),reverse=True)[:4]
  result=dict(episode=ep,peak=context['gate'],open=context['a'],close=context['b'],withdrawal=context['withdrawal'],reachable=len(costs),frontier=[])
  for i,j in points:
   edges=[]
   for di,dj in [(1,1),(1,0),(0,1)]:
    ni,nj=i+di,j+dj
    if ni>=n or nj>=m:continue
    edges.append(dict(next=[float(src[0][ni]),float(src[1][nj])],can_wait=(bool(kwargs['can_wait'](0,i)) if di==0 else bool(kwargs['can_wait'](1,j)) if dj==0 else True),dependency=bool(kwargs['dependency'](ni,nj)),safe=bool(safe(i,j,ni,nj)),clear=[bool(context['clear'](p,q)) for p,q in [(i,j),(ni,nj),(i,nj),(ni,j)]]))
   result['frontier'].append(dict(source=[float(src[0][i]),float(src[1][j])],edges=edges))
  records.append(result);(root/'schedule-evidence.json').write_text(json.dumps(records,indent=2));print(result,flush=True);raise
visual.schedule_sources=diagnose
for ep in [72,74,77,78,80]:
 try:plan_episode(cfg,ep)
 except NoSafeSchedule as error:print(ep,str(error),flush=True)
