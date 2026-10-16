"""Pin added stack code and dependencies; never touch old freeze artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
BASE='c4e4eb8c2a9d87d42303e2c0021e168942fda33a'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify(root,value):
    if value.get('schema')!='seq-combo-stack-freeze-v1' or value.get('base')!=BASE or value.get('seeds')!=[42,2024,2025]:raise ValueError('freeze identity mismatch')
    for name,pin in value['files'].items():
        p=root/name
        if p.is_symlink() or sha(p)!=pin:raise ValueError('frozen file drift: '+name)
    for seed in (42,2024,2025):
        p=root/f'configs/seq_combo_stack/p-seq-combo-stack-s{seed}.json'
        from vkgr.retrieval_seq_combo_stack_run import validate_config
        c=json.loads(p.read_text());validate_config(c)
        if c['seed']!=seed:raise ValueError('config seed mismatch')
    return True

def payload(root):
    old=subprocess.check_output(['git','ls-tree','-r','--name-only',BASE],cwd=root,text=True).splitlines()
    for name in old:
        if name.startswith(('src/','scripts/','tests/','configs/')):
            original=subprocess.check_output(['git','show',BASE+':'+name],cwd=root)
            if (root/name).read_bytes()!=original:raise ValueError('old source changed: '+name)
    names=[]
    for directory in ('src','scripts','tests','configs'):
        names.extend(str(p.relative_to(root)) for p in (root/directory).rglob('*') if p.is_file() and (p.suffix in ('.py','.json')) and '__pycache__' not in p.parts)
    names=[n for n in names if n!='configs/seq_combo_stack/freeze-v1.json']
    names.append('docs/superpowers/specs/2026-10-17-seq-combo-stack-card.md')
    return {'schema':'seq-combo-stack-freeze-v1','base':BASE,'seeds':[42,2024,2025],'files':{n:sha(root/n) for n in sorted(names)}}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--write',action='store_true');a=ap.parse_args();root=Path(__file__).resolve().parents[1];p=root/'configs/seq_combo_stack/freeze-v1.json'
    if a.write:
        with p.open('x') as h:json.dump(payload(root),h,indent=2,sort_keys=True)
    v=json.loads(p.read_text());verify(root,v);print(json.dumps({'status':'freeze_verified','files':len(v['files']),'sha256':sha(p)}))
if __name__=='__main__':main()
