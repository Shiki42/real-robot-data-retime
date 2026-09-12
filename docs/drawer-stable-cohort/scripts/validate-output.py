import json,sys,cv2,numpy as np,pyarrow as pa,pyarrow.parquet as pq
from pathlib import Path
from real_robot_data_retime.timeline.smooth import sample_rows
from real_robot_data_retime.timeline.uniform import validate_stage_schedule
from real_robot_data_retime.uniform_drawer import source_map_digest
from real_robot_data_retime.interaction.measurements import producer_fingerprint
from real_robot_data_retime.automatic_validation import wrist_pixel_error
R=Path('/home/coder/share/drawer-stable-cohort-20260912')
def validate(ep,c):
 source=Path(c['source']);output=Path(c['output']);records=[];tiles=[]
 original=pq.read_table(source/f'data/chunk-000/file-{ep:03d}.parquet');positions=[]
 for v in range(2):
  target=c['source_indices'].index(ep)+v*len(c['source_indices']);receipt=json.loads((output/f'meta/retime_receipts/episode_{target:03d}.json').read_text());p=receipt['plan'];table=pq.read_table(output/f'data/chunk-000/file-{target:03d}.parquet')
  with np.load(output/f'meta/retime_source_indices/episode_{target:03d}.npz') as m:maps={k:m[k] for k in m.files}
  left,right=maps['left'],maps['right'];n=len(left);assert n==len(table) and p['producer']==producer_fingerprint();assert source_map_digest(left,right)==p['source_map_digest']
  for key in ['action','observation.state']:
   values=np.array(original[key].to_pylist());expected=np.c_[sample_rows(values[:,:7],left),sample_rows(values[:,7:],right)];expected=np.array(pa.array(expected.tolist(),type=original.schema.field(key).type).to_pylist());assert np.array_equal(expected,np.array(table[key].to_pylist()))
  assert table['frame_index'].to_pylist()==list(range(n)) and set(table['episode_index'].to_pylist())=={target};assert np.allclose(table['timestamp'].to_numpy(),np.arange(n)/30,atol=1e-5)
  assert np.array_equal(maps['raw_left'],left+receipt['trim']['start']) and np.array_equal(maps['raw_right'],right+receipt['trim']['start'])
  stage=validate_stage_schedule(left,right,p['stages']);assert receipt['compositing']['automatic_origin_audit']['passed']
  cameras={};event=receipt['interaction']['timeline']['episodes'][0];s=p['stages'];selected=[s['stop_output_frames'][0],s['stop_output_frames'][-1],min(int(np.searchsorted(left,event['release_frame']+10)),n-1),n-1];review={}
  for camera in ['top','left_wrist','right_wrist']:
   path=output/f'videos/observation.images.{camera}/chunk-000/file-{target:03d}.mp4';cap=cv2.VideoCapture(str(path));assert cap.get(cv2.CAP_PROP_FRAME_COUNT)==n and cap.get(cv2.CAP_PROP_FPS)==30;count=0
   while True:
    ok,img=cap.read()
    if not ok:break
    if camera=='top' and count in selected:review[count]=img
    count+=1
   cap.release();assert count==n;cameras[camera]=dict(frames=count,fps=30)
   if camera!='top':cameras[camera]['pixel_check']=wrist_pixel_error(source/f'videos/observation.images.{camera}/chunk-000/file-{ep:03d}.mp4',path,left if camera=='left_wrist' else right,samples=16)
  row=[]
  for k in selected:
   img=cv2.resize(review[k],(320,181));cv2.putText(img,f'src{ep} out{target} f{k}',(5,17),0,.45,(0,0,255),1);row.append(img)
  tiles.append(np.hstack(row));positions.append(s['uniform']['position']);records.append(dict(source=ep,output=target,frames=n,numeric_exact=True,cameras=cameras,stage=stage,uniform_position=positions[-1],origin_audit=receipt['compositing']['automatic_origin_audit'],overlap_pixel_frames=receipt['compositing']['arm_overlap_pixel_frames'],depth_valid_overlap_pixel_frames=receipt['compositing']['valid_metric_overlap_pixel_frames']))
 assert abs(positions[1]-positions[0]-.5)<1e-12
 review_dir=R/'review';review_dir.mkdir(exist_ok=True);cv2.imwrite(str(review_dir/f'source-{ep:03d}.jpg'),np.vstack(tiles),[cv2.IMWRITE_JPEG_QUALITY,65])
 dst=R/'validation';dst.mkdir(exist_ok=True);(dst/f'source-{ep:03d}.json').write_text(json.dumps(dict(passed=True,results=records),indent=2));return records
if __name__=='__main__':
 c=json.loads(Path(sys.argv[1]).read_text());print(json.dumps(validate(int(sys.argv[2]),c)),flush=True)
