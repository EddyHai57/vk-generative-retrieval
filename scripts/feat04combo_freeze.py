"""Immutable code/config freeze. No datasets or checkpoint files included."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def validate_configs(rows):
    if len(rows)!=3 or [x.get('seed') for x in rows]!=[42,2024,2025]:raise ValueError('exact three registered seeds required')
    base=rows[0]
    for x in rows:
        if x!={**base,'seed':x['seed']}:raise ValueError('seed is the only configurable difference')
        if (x['task_id'],x['feature'],x['schema'],x['temperature'],x['learning_rate'],x['logq'],x['max_epochs'],x['min_epochs'],x['patience'])!=('FEAT04-COMBO','f4_combo_c_d_e','p-feat04combo-run-v1',.1,.001,True,100,5,3):raise ValueError('registered combination changed')
def verify(root,value):
    for name,pin in value['files'].items():
        p=root/name
        if p.is_symlink() or sha(p)!=pin:raise ValueError('frozen file mismatch: '+name)
    return True

def payload(root):
    rows=[json.loads((root/f'configs/feat04combo/p-feat04combo-cde-s{s}.json').read_text()) for s in (42,2024,2025)];validate_configs(rows)
    changes=subprocess.check_output(['git','diff','--name-status','--no-renames','5d51228'],cwd=root,text=True).splitlines()
    if any(not line.startswith('A\t') for line in changes):raise ValueError('an old file changed; only additions are permitted')
    names=subprocess.check_output(['git','ls-files'],cwd=root,text=True).splitlines()
    names=[n for n in names if n.startswith(('src/','scripts/','tests/','configs/')) and n!='configs/feat04combo/freeze-v1.json']
    return {'schema':'feat04combo-freeze-v1','components':['C','D','E'],'seeds':[42,2024,2025],'files':{n:sha(root/n) for n in sorted(names)},'old_frozen_files_unchanged':True,'report_rule':'original plan 1.2; combination-minus-E descriptive; C9 report-only'}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--write',action='store_true');args=ap.parse_args();root=Path(__file__).resolve().parents[1];p=root/'configs/feat04combo/freeze-v1.json'
    if args.write:
        with p.open('x') as f:json.dump(payload(root),f,sort_keys=True,indent=2)
    value=json.loads(p.read_text());verify(root,value);validate_configs([json.loads((root/f'configs/feat04combo/p-feat04combo-cde-s{s}.json').read_text()) for s in (42,2024,2025)])
    print(json.dumps({'status':'freeze_verified','files':len(value['files']),'freeze_sha256':sha(p)}))
if __name__=='__main__':main()
