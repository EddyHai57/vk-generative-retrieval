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
 ('bridge-claimed-exact-bytes', 'analysis/sgns_full_input_bridge.py', "if queue.exists() or (queue.parent.parent/'rejected'/queue.name).exists() or claimed.read_bytes()!=change['raw']['after']:\n        raise ValueError('exact overridden original claimed queue entry required')", "if False:\n        raise ValueError('exact overridden original claimed queue entry required')"),
 ('bridge-source-ledger', 'analysis/sgns_full_input_bridge.py', 'if actual!=expected:', 'if False:'),
 ('bridge-extra-source-path', 'analysis/sgns_full_input_bridge.py', 'set(files)!=set(old)|added', 'False'),
 ('bridge-fixed-plan-pin', 'analysis/sgns_full_input_bridge.py', "if record.get('sha256')!=pins[name+'_sha256']:", 'if False:'),
 ('bridge-sole-ram-change', 'analysis/sgns_full_input_bridge.py', "or {k:v for k,v in before.items() if k!='ram_mib'}!={k:v for k,v in after.items() if k!='ram_mib'}", 'or False'),
 ('bridge-old-metadata', 'analysis/sgns_full_input_bridge.py', "or spec.get('inputs')!=[r['path'] for r in inputs]", 'or False'),
 ('bridge-prepare-hook', 'analysis/sgns_gpu_full.py', 'bridge_context = bridge_preflight(args,source)', 'bridge_context = None'),
 ('bridge-consumer-hook', 'analysis/sgns_gpu_full.py', 'verify_manifest(value,source)', 'pass'),
 ('bridge-atomic-run-state', 'analysis/sgns_full_input_bridge.py', "(ops/'runs'/run_id).exists()", 'False'),
 ('bridge-atomic-claimed-state', 'analysis/sgns_full_input_bridge.py', "list((ops/'queue/claimed').glob('*-'+run_id+'.json'))", 'False'),
 ('bridge-atomic-rejected-state', 'analysis/sgns_full_input_bridge.py', "list((ops/'queue/rejected').glob('*-'+run_id+'.json'))", 'False'),
 ('bridge-atomic-second-check', 'analysis/sgns_full_input_bridge.py', 'unchanged();os.replace(temp,queue)', 'os.replace(temp,queue)'),

 ('four-gate-bool', 'analysis/sgns_gpu_full.py', "return consume(path,sha256,scratch)", "return {'verified':True}"),
 ('acceptance-unfrozen-code', 'analysis/sgns_subset_acceptance.py', 'if required and not ready:', 'if False:'),
 ('acceptance-consumed-record', 'analysis/sgns_subset_acceptance.py', "if sha(raw)!=pin(record['sha256']):", 'if False:'),
 ('acceptance-ignored-shadow', 'analysis/sgns_subset_acceptance.py', 'if actual!=expected:', 'if False:'),
 ('acceptance-module-origin', 'analysis/sgns_subset_acceptance.py', "except ValueError:raise ValueError('module origin outside original pinned tree') from None", 'except ValueError:return'),
 ('acceptance-terminal-signal', 'analysis/sgns_subset_acceptance.py', "or terminal.get('signal') is not None", 'or False'),
 ('acceptance-terminal-wall-cap', 'analysis/sgns_subset_acceptance.py', "or terminal.get('wall_cap_exceeded') is not False", 'or False'),
 ('input-producer-hook', 'analysis/sgns_subset_acceptance_a1.py', 'checked_input=check_input_producer(accepted,source,env,code)', "checked_input={'evidence':{}}"),
 ('input-producer-full-metadata', 'analysis/sgns_subset_acceptance_a1.py', "or any(s.encoded(run.get(k))!=s.encoded(val) for k,val in expected.items())", 'or False'),
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
        run = subprocess.run([sys.executable,'-B','-m','unittest','tests.test_sgns_gpu_full','tests.test_sgns_subset_acceptance','tests.test_sgns_full_input_bridge'],cwd=root,env=env,capture_output=True,text=True)
        assert run.returncode != 0, 'survived: '+name
        results.append({'mutation':name,'killed':True,'isolated':True})
print(json.dumps({'results':results,'all_killed':True,'isolation':'new copy + -B + PYTHONDONTWRITEBYTECODE=1 + unique cache prefix'},sort_keys=True))
