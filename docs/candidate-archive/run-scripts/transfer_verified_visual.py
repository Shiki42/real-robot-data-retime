from pathlib import Path
import json,hashlib,sys
r=Path('/home/coder/share/retime-interaction-20260909');ep=int(sys.argv[1]);old=r/f'visual-review-final/episode_{ep:03d}_accepted.json';new=r/f'visual-review-b054795/episode_{ep:03d}_sampling.json';receipt=r/f'drawer-retimed-b054795/meta/retime_receipts/episode_{ep:03d}.json'
a=json.loads(old.read_text());b=json.loads(new.read_text());data=json.loads(receipt.read_text())
if data['reuse_verification']['previous_receipt_sha256']!=a['receipt_sha256'] or a['video_sha256']!=b['video_sha256'] or a['output_frames']!=b['output_frames']:raise ValueError('prior visual review is not identical')
if b['receipt_sha256']!=hashlib.sha256(receipt.read_bytes()).hexdigest():raise ValueError('stale sampling')
result=dict(**b,passed=True,reviewer='Codex visual inspection of identical previously reviewed video bytes',scope=a['scope'],notes=a['notes'],previous_visual_review_sha256=hashlib.sha256(old.read_bytes()).hexdigest(),previous_receipt_sha256=a['receipt_sha256'])
(r/f'visual-review-b054795/episode_{ep:03d}_accepted.json').write_text(json.dumps(result,indent=2));print(ep,'identical visual review carried forward')
