"""Pinned module bridge for writer transport (python -c has no __file__)."""
import hashlib
import importlib.util
from pathlib import Path
import sys

CORE=Path('/workspace/vkgr-seqm1-night-20261016/code/analysis/seqm1_night.py')
PIN='9e2b1a2e5114882e25a56233883ccacdebed4235be6f53e457c3eac3d8c7edda'

def load_core(path,pin):
    if hashlib.sha256(path.read_bytes()).hexdigest()!=pin:raise ValueError('night module pin mismatch')
    sys.path.insert(0,str(path.parent))
    spec=importlib.util.spec_from_file_location('seqm1_night',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

if __name__=='__main__':
    sys.path.insert(0,'/workspace/vkgr-infra03-ops/ops')
    load_core(CORE,PIN).main()
