"""Readonly probe: /private/tmp existence, repo reachability, server python/torch/numpy, uid. No writes. (DEPLOY-01 deploy01_probe_read.py)"""
import json,os,subprocess,sys
from pathlib import Path
S=CFG['server']
r=subprocess.run(['git','ls-remote',S['repo_url']],capture_output=True,text=True,timeout=60)
v=subprocess.run([S['python'],'-c','import sys,torch,numpy;print(sys.version.split()[0],torch.__version__,numpy.__version__)'],capture_output=True,text=True,timeout=60)
pt=Path('/private/tmp')
print(json.dumps({'private_tmp_exists':pt.exists(),'private_tmp_is_symlink':pt.is_symlink(),'private_tmp_target':os.path.realpath(pt) if pt.exists() else None,
 'ls_remote_rc':r.returncode,'ls_remote':r.stdout[-800:],'ls_remote_err':r.stderr[-300:],'remote_py':v.stdout.strip(),'remote_py_err':v.stderr[-300:],'uid':os.getuid(),
 'git_version':subprocess.run(['git','--version'],capture_output=True,text=True).stdout.strip()},indent=1))
