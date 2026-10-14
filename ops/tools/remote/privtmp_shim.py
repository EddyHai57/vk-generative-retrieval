"""Writer: create NEW test-only shim CFG work_dir/shim-privtmp/sitecustomize.py that remaps tempfile dir='/private/tmp'
(macOS convention hard-coded in G tests) to NEW CFG work_dir/privtmp. Used ONLY via PYTHONPATH for the G regression run;
never on any job. Writes only those two new dirs. (DEPLOY-01 deploy01_privtmp_shim_remote.py)"""
import json,subprocess,sys
from pathlib import Path
W=Path(CFG['server']['work_dir']);S=W/'shim-privtmp';T=W/'privtmp'
S.mkdir(mode=0o700);T.mkdir(mode=0o700)
(S/'sitecustomize.py').write_text('''import tempfile as _t
_orig=_t._sanitize_params
def _p(prefix,suffix,dir):
    if dir is not None and str(dir).rstrip('/')=='/private/tmp':dir=%r
    return _orig(prefix,suffix,dir)
_t._sanitize_params=_p
''' % str(T))
pre=subprocess.run([sys.executable,'-c','import importlib.util;s=importlib.util.find_spec("sitecustomize");print(s.origin if s else None)'],capture_output=True,text=True).stdout.strip()
chk=subprocess.run([sys.executable,'-c','import tempfile;d=tempfile.TemporaryDirectory(dir="/private/tmp");print(d.name);d.cleanup()'],capture_output=True,text=True,env={'PYTHONPATH':str(S)})
print(json.dumps({'shim':str(S),'target':str(T),'preexisting_sitecustomize':pre,'check':chk.stdout.strip(),'check_err':chk.stderr[-300:]}))
