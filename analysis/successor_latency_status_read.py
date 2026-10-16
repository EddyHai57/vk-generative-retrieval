"""Pinned read-only latency receipt entry."""
import hashlib,importlib.util,json,sys
from pathlib import Path
p=Path('/workspace/vkgr-successor-ops-v1/code/analysis/successor_control.py')
if hashlib.sha256(p.read_bytes()).hexdigest()!='df08e2fb3fbf9422432e36fe89b7609e2ba5ff3ecca3900a7f53638fe2d4327c':raise ValueError('control source mismatch')
sys.path.insert(0,str(p.parent))
s=importlib.util.spec_from_file_location('control',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
print(json.dumps(m.status(),sort_keys=True))
