"""Materialize reviewed per-arm idle labels in a separate LeRobot v3 export."""
import json,shutil,hashlib
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from real_robot_data_retime.stats import feature_statistics

SOURCE=Path('/home/coder/share/drawer-no-unneeded-stop-20260915/dataset')
DEST=Path('/home/coder/share/piperx-put-cube-in-drawer-20260908-87ep-ctr')
REPO='Shiki42/piperx-put-cube-in-drawer-20260908-87ep-ctr'

def main():
    shutil.copytree(SOURCE,DEST)
    masks=json.loads((SOURCE/'idle-masks.json').read_text())
    info=json.loads((DEST/'meta/info.json').read_text())
    stats=json.loads((DEST/'meta/stats.json').read_text())
    keys=['retime.left_idle','retime.right_idle']
    aggregate={key:[] for key in keys};episode_stats={};total=0
    assert len(masks['episodes'])==info['total_episodes']==156
    for record in masks['episodes']:
        ep=record['output'];path=DEST/f'data/chunk-000/file-{ep:03d}.parquet';table=pq.read_table(path);original=table
        assert len(table)==record['frames'] and set(table['episode_index'].to_pylist())=={ep}
        episode_stats[ep]={}
        for arm,key in zip(['left','right'],keys):
            values=np.zeros(len(table),bool);last=0
            for start,end in record[arm]:
                assert last<=start<end<=len(table);values[start:end]=True;last=end
            rest=record[arm+'_terminal_rest'];assert values[rest['mask_start']:].all();assert (~values[rest['start']:]).sum()<=45
            table=table.append_column(key,pa.array(values));aggregate[key].append(values[:,None]);episode_stats[ep][key]=feature_statistics(values[:,None])
        pq.write_table(table,path)
        readback=pq.read_table(path);assert readback.select(original.column_names).equals(original)
        for key in keys:assert readback[key].equals(table[key])
        total+=len(table)
    for path in (DEST/'meta/episodes').rglob('*.parquet'):
        table=pq.read_table(path);ids=table['episode_index'].to_pylist()
        for key in keys:
            for stat in episode_stats[ids[0]][key]:
                dtype=table.schema.field('stats/retime.synthetic_hold/'+stat).type
                table=table.append_column('stats/'+key+'/'+stat,pa.array([episode_stats[ep][key][stat] for ep in ids],type=dtype))
        pq.write_table(table,path)
    for key in keys:
        info['features'][key]=dict(dtype='bool',shape=[1],names=None)
        stats[key]=feature_statistics(np.concatenate(aggregate[key]))
    info['repo_id']=REPO
    (DEST/'meta/info.json').write_text(json.dumps(info,indent=2)+'\n')
    (DEST/'meta/stats.json').write_text(json.dumps(stats,indent=2)+'\n')
    assert total==info['total_frames']==82569
    videos=list((DEST/'videos').rglob('*.mp4'));assert len(videos)==468
    for path in videos:
        assert hashlib.sha256(path.read_bytes()).digest()==hashlib.sha256((SOURCE/path.relative_to(DEST)).read_bytes()).digest()
    report=dict(passed=True,repo_id=REPO,codebase_version='v3.0',episodes=156,frames=total,fps=30,videos=468,original_columns_unchanged=True,video_sha256_unchanged=True,idle_true_counts={k:int(np.concatenate(v).sum()) for k,v in aggregate.items()},training_hook='Consumer must apply idle masks explicitly; LeRobot does not automatically exclude custom idle fields from loss.')
    (DEST/'idle-mask-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    (DEST/'README.md').write_text('''---
tags:
- lerobot
- robotics
- piperx
- counterfactual-retiming
---
# piperx-put-cube-in-drawer-20260908-87ep-ctr

LeRobot v3.0 dataset:156 episodes from78 retained original sources (two timing
variants each),82,569 frames at30FPS. The87ep suffix identifies the original
87-source collection; this export contains156 counterfactual episodes.

Three synchronized RGB cameras: observation.images.top (848x480),
observation.images.left_wrist and observation.images.right_wrist (640x480).
Action and observation.state are float32[14]: left6 joint angles plus gripper,
then right6 joint angles plus gripper. Joint positions are degrees; grippers mm.
All frames, Action/State and source clocks come from the validated no-unneeded-stop
regeneration. If the drawer is open when the left arm reaches its lift peak,
it proceeds without inserted braking/stopping/restarting.61 outputs use this path.

## Idle masks for action loss

Each frame Parquet includes boolean retime.left_idle and retime.right_idle.
True means EXCLUDE that arm's seven action dimensions from loss. False means
retain supervision. A training loader must explicitly consume these fields;
custom LeRobot features alone do not change the loss automatically.

```python
# Per-action-timestep element mask; apply separately to each action-chunk step.
left_keep = ~batch["retime.left_idle"].bool()
right_keep = ~batch["retime.right_idle"].bool()
# Broadcast each arm's keep value across its own seven action dimensions.
# Combine with the training loader's padding mask before normalized reduction.
```

Necessary left waiting for drawer opening and right waiting for left withdrawal
remain supervised. Excess pre-close waits are continuous intervals ending before
required closing preparation. After each arm settles in its final rest pose,
at most1.5 seconds (45 frames) remain supervised. Existing idle frames stay idle.
idle-masks.json contains exact zero-based half-open intervals and per-arm rest
boundaries. Per-frame fields, feature declarations, global stats and per-episode
stats were generated from the same masks and verified.

Original source: Travor278/piperx-put-cube-in-drawer-20260908-87ep,
revision58bbbd720f6e78d162b8f4bc7077759d34c5162f. Source selection uses cabinet
marker displacement P90<=10px in a640px analysis image. Nine sources excluded.
Existing Trim removed615 initial frames from the78 retained sources; no tail cut.
Code repository: https://github.com/Shiki42/real-robot-data-retime
Generation code9ea4e87; regenerated delivery6c0b163.

Numerical/frame validation is not flawless compositing certification. Exposure,
color and segmentation-edge artifacts remain possible. Source maps, receipts,
selection and validation evidence accompany the dataset. Original data preserved.
''')
    files=sorted(p for p in DEST.rglob('*') if p.is_file() and p.name!='SHA256SUMS')
    (DEST/'SHA256SUMS').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(DEST))+'\n' for p in files))
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
