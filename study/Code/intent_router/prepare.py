"""Prepare official MASSIVE 1.1 zh-CN; never extract arbitrary archive paths."""
import collections
import hashlib
import json
from pathlib import Path
import tarfile
import unicodedata

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'

def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]

def normalize(text):
    return ''.join(unicodedata.normalize('NFKC', text).lower().split())

def main():
    archive = DATA / 'raw/massive-1.1.tar.gz'
    with tarfile.open(archive) as tf:
        members = [m for m in tf.getmembers() if m.isfile() and m.name.endswith('/zh-CN.jsonl')]
        assert len(members) == 1, 'Expected one Chinese source file'
        raw = tf.extractfile(members[0]).read()
        (DATA / 'raw/zh-CN.jsonl').write_bytes(raw)
        licenses = [m for m in tf.getmembers() if m.isfile() and m.name.split('/')[-1] == 'LICENSE']
        assert licenses, 'Missing dataset license'
        (DATA / 'raw/LICENSE').write_bytes(tf.extractfile(licenses[0]).read())
    rows = [json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
    assert len({str(r['id']) for r in rows}) == len(rows)
    splits = {s: [] for s in ['train', 'dev', 'test']}
    for r in rows:
        assert r['locale'] == 'zh-CN' and r['utt'].strip()
        splits[r['partition']].append({k: r[k] for k in ['id', 'utt', 'intent']})
    assert {s: len(rs) for s, rs in splits.items()} == {'train':11514, 'dev':2033, 'test':2974}
    labels = sorted({r['intent'] for r in splits['train']})
    assert len(labels) == 60
    for s, rs in splits.items():
        assert set(r['intent'] for r in rs) <= set(labels)
        (DATA / f'{s}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rs), encoding='utf-8')
    save(DATA / 'labels.json', labels)
    groups = {}
    for s, rs in splits.items():
        for r in rs:
            groups.setdefault(normalize(r['utt']), []).append((s, str(r['id']), r['intent']))
    cross = [g for g in groups.values() if len({x[0] for x in g}) > 1]
    audit = {
        'source': 'https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz',
        'dataset_version': '1.1', 'locale':'zh-CN', 'license':'CC-BY-4.0',
        'archive_sha256':hashlib.file_digest(archive.open('rb'), 'sha256').hexdigest(),
        'source_sha256':hashlib.sha256(raw).hexdigest(),
        'counts':{s:len(rs) for s,rs in splits.items()},
        'label_counts':{s:dict(sorted(collections.Counter(r['intent'] for r in rs).items())) for s,rs in splits.items()},
        'normalization':'NFKC, lowercase, remove whitespace; does not detect semantic duplicates',
        'duplicate_text_groups':sum(len(g)>1 for g in groups.values()),
        'conflicting_label_groups':sum(len({x[2] for x in g})>1 for g in groups.values()),
        'cross_split_text_groups':len(cross), 'cross_split_members':cross,
        'policy':'Official splits preserved; report additional dev metrics excluding normalized text seen in train. Test held for final evaluation.'
    }
    save(DATA / 'audit.json', audit)
    print(json.dumps({k:audit[k] for k in ['counts','duplicate_text_groups','conflicting_label_groups','cross_split_text_groups']}, ensure_ascii=False))

if __name__ == '__main__':
    main()
