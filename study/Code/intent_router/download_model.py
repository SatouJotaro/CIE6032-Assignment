"""Download public model at an immutable revision into this project."""
import json
from pathlib import Path
from huggingface_hub import model_info, snapshot_download

ROOT = Path(__file__).resolve().parent
MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'

if __name__ == '__main__':
    lock = ROOT / 'model_lock.json'
    revision = json.loads(lock.read_text())['revision'] if lock.exists() else model_info(MODEL).sha
    info = {'model_id': MODEL, 'revision': revision, 'local_path':'models/Qwen2.5-0.5B-Instruct'}
    lock.write_text(json.dumps(info, indent=2), encoding='utf-8')
    print('Downloading', info, flush=True)
    snapshot_download(MODEL, revision=revision, local_dir=ROOT/info['local_path'],
                      allow_patterns=['*.json','*.safetensors','vocab.json','merges.txt','LICENSE','README.md'], max_workers=2)
    print('Download complete', flush=True)
