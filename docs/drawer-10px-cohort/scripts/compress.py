import json,csv,subprocess,hashlib,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
source=Path('/home/coder/share/drawer-10px-cohort-20260912/dataset');root=Path('/home/coder/share/drawer-10px-cohort-20260912/local-review-package');dest=root/'review';(dest/'videos').mkdir(parents=True,exist_ok=True)
def run(ep):
 src=source/f'videos/observation.images.top/chunk-000/file-{ep:03d}.mp4';dst=dest/f'videos/episode_{ep:03d}.mp4';rec=json.loads((source/f'meta/retime_receipts/episode_{ep:03d}.json').read_text());n=rec['length']
 subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(src),'-map','0:v:0','-an','-vf','scale=640:-2:flags=lanczos','-c:v','libx264','-preset','medium','-crf','23','-threads','2','-pix_fmt','yuv420p','-vsync','0','-movflags','+faststart',str(dst)],check=True)
 meta=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=width,height,avg_frame_rate,nb_read_frames','-of','json',str(dst)]))['streams'][0];assert int(meta['nb_read_frames'])==n and meta['avg_frame_rate']=='30/1' and (meta['width'],meta['height'])==(640,362)
 return dict(output=ep,source=rec['source_episode_index'],variant=rec['plan']['variant'],position=rec['plan']['stages']['uniform']['position'],frames=n,fps=30,width=640,height=362,filename=f'videos/episode_{ep:03d}.mp4',source_bytes=src.stat().st_size,bytes=dst.stat().st_size,sha256=hashlib.sha256(dst.read_bytes()).hexdigest(),trim_head_removed=rec['trim']['start'],quality_notes=rec.get('visual_review',{}).get('quality_notes',[]))
restored=set(json.loads((root.parent/'selection.json').read_text())['restored_sources'])
notes={42:['原始源含明显单帧色偏，请重点审核。'],41:['原始源存在曝光或颜色变化，请审核合成接缝。'],56:['原始源存在曝光或颜色变化，请审核合成接缝。'],69:['原始源存在曝光或颜色变化，请审核合成接缝。'],79:['原始源存在曝光或颜色变化，请审核合成接缝。'],47:['请重点审核机械臂分割及合成边缘。'],61:['请重点审核机械臂分割及合成边缘。'],71:['请重点审核机械臂分割及合成边缘。']}
rows=[]
with ThreadPoolExecutor(max_workers=6) as pool:
 for future in as_completed([pool.submit(run,i) for i in range(156)]):
  row=future.result();row['restored']=row['source'] in restored;row['quality_notes']=notes.get(row['source'],[]);rows.append(row);print(row['output'],row['bytes'],flush=True)
rows.sort(key=lambda x:x['output']);assert len(rows)==156
report=dict(episodes=156,frames=sum(x['frames'] for x in rows),fps=30,encoding=dict(codec='h264',crf=23,preset='medium',scale='640x362',frame_dropping=False),source_dataset=str(source),source_bytes=sum(x['source_bytes'] for x in rows),bytes=sum(x['bytes'] for x in rows),episodes_data=rows)
(dest/'manifest.json').write_text(json.dumps(report,indent=2));
with (dest/'episode-map.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=['output','source','variant','position','frames','fps','filename','trim_head_removed'],extrasaction='ignore',lineterminator='\n');writer.writeheader();writer.writerows(rows)
(dest/'SHA256SUMS').write_text('\n'.join(x['sha256']+'  '+x['filename'] for x in rows)+'\n');print('DONE',report['source_bytes'],report['bytes'],flush=True)
