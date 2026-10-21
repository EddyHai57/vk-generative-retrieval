"""Isolated fresh-copy/-B/cache-prefix mutation checks; no real data or training."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

repo = Path(__file__).resolve().parents[2]
mutations = [
 ('four-gate-bool', 'analysis/sgns_gpu_full.py', "raise ValueError('four-gate total receipt producer/adapter not frozen; full training STOP')", "return {'verified':True}"),
 ('existing-source-drift', 'analysis/sgns_gpu_full.py', "or any(source['files'].get(k) != v for k,v in base.items())", 'or False'),
 ('physical-row-groups', 'analysis/sgns_gpu_full.py', "[parquet.metadata.row_group(i).num_rows for i in range(parquet.metadata.num_row_groups)] != record['row_groups']", 'False'),
 ('consumed-json-hash', 'analysis/sgns_gpu_full.py', "if hashlib.sha256(raw).hexdigest() != sha:\n        raise ValueError('consumed JSON SHA differs')", "if False:\n        raise ValueError('consumed JSON SHA differs')"),
 ('job-stdout-binding', 'analysis/sgns_gpu_full.py', "or receipt_from_bytes(blobs['logs/stdout.log']) != receipt", 'or False'),
 ('expected-job-semantic-comparison', 'analysis/sgns_gpu_full.py', 'or any(encoded(run.get(k)) != encoded(v) for k,v in expected_job.items())', 'or False'),
 ('terminal-signal', 'analysis/sgns_gpu_full.py', "or terminal.get('signal') is not None", 'or False'),
 ('terminal-wall-cap', 'analysis/sgns_gpu_full.py', "or terminal.get('wall_cap_exceeded') is not False", 'or False'),
 ('snapshot-stability', 'analysis/sgns_gpu_full.py', 'if any(secure(path/name).read_bytes() != raw for name,raw in blobs.items()):', 'if False:'),
 ('verifier-five-epochs', 'src/vkgr/sgns_full_verify.py', "if audit['epochs_completed']!=5:", "if audit['epochs_completed']!=4:"),
]
results = []
for name,relative,old,new in mutations:
    with tempfile.TemporaryDirectory(prefix='full-sgns-mutation-'+name+'-') as temp:
        root = Path(temp).resolve()/'copy'
        root.mkdir()
        for folder in ('analysis','src','tests','configs'):
            shutil.copytree(repo/folder,root/folder,ignore=shutil.ignore_patterns('__pycache__'))
        path = root/relative; raw = path.read_text()
        assert raw.count(old) == 1,(name,raw.count(old))
        path.write_text(raw.replace(old,new))
        env = {**os.environ,'PYTHONPATH':str(root/'src'),'PYTHONDONTWRITEBYTECODE':'1',
               'PYTHONPYCACHEPREFIX':str(Path(temp)/'unique-pycache')}
        run = subprocess.run([sys.executable,'-B','-m','unittest','tests.test_sgns_gpu_full'],cwd=root,env=env,capture_output=True,text=True)
        assert run.returncode != 0, 'survived: '+name
        results.append({'mutation':name,'killed':True,'isolated':True})
print(json.dumps({'results':results,'all_killed':True,'isolation':'new copy + -B + PYTHONDONTWRITEBYTECODE=1 + unique cache prefix'},sort_keys=True))
