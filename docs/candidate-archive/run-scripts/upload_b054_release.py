"""Publish a fully verified dataset, then download and hash the pinned revision."""
from pathlib import Path
import hashlib,json
from huggingface_hub import HfApi,hf_hub_download
r=Path('/home/coder/share/retime-interaction-20260909');out=r/'drawer-retimed-b054795';repo='Shiki42/piperx-put-cube-in-drawer-retime'
report=json.loads((out/'validation.json').read_text())
if not report['passed'] or report['episodes']!=87:raise ValueError('all87 validation required')
manifest=json.loads((out/'SHA256SUMS.json').read_text())
for rel,item in manifest.items():
 with (out/rel).open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
 if digest!=item['sha256'] or (out/rel).stat().st_size!=item['size']:raise ValueError(f'local release changed: {rel}')
api=HfApi()
if api.whoami()['name']!='Shiki42':raise ValueError('unexpected HF account')
api.create_repo(repo,repo_type='dataset',private=False,exist_ok=True)
commit=api.upload_folder(repo_id=repo,repo_type='dataset',folder_path=out,commit_message='Publish all 87 automatically retimed and verified drawer episodes',ignore_patterns=['.cache/**'])
revision=commit.oid
expected=set(manifest)|{'SHA256SUMS.json'};remote=set(api.list_repo_files(repo,repo_type='dataset',revision=revision));extras=remote-expected-{'.gitattributes'}
if expected-remote or extras:raise ValueError(f'remote inventory differs; missing {expected-remote}; extra {extras}')
verified=[]
for rel in sorted(expected):
 path=Path(hf_hub_download(repo_id=repo,filename=rel,repo_type='dataset',revision=revision,local_dir=r/'hf-final-verification'))
 with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
 if rel=='SHA256SUMS.json':
  with (out/rel).open('rb') as stream:wanted=hashlib.file_digest(stream,'sha256').hexdigest()
 else:wanted=manifest[rel]['sha256']
 if digest!=wanted:raise ValueError(f'uploaded hash mismatch: {rel}')
 verified.append(rel)
 print('verified',len(verified),len(expected),rel,flush=True)
result=dict(repo_id=repo,revision=revision,public=not api.dataset_info(repo).private,files=len(verified),hashes_verified=True)
if not result['public']:raise ValueError('dataset is not public')
(r/'hf-final-upload-verification.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
