"""Revalidate frozen science and clean launcher before running the original entry."""
import importlib.util
from pathlib import Path
import runpy
import sys
sys.dont_write_bytecode = True

def load_bridge():
    path=Path(__file__).with_name('gtok02a1_launch_bridge.py')
    spec=importlib.util.spec_from_file_location('gtok02a1_launch_bridge',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def main(argv=None):
    args=list(sys.argv[1:] if argv is None else argv)
    bridge=load_bridge();ops=bridge.guard()
    sys.path[:0]=[str(ops.DEP/'src'),str(ops.DEP/'scripts')]
    while args and args[0]=='-B':args.pop(0)
    if len(args)==2 and args[0]=='--probe':
        script=str(ops.DEP/'scripts/gtok02a1_cuda_check.py')
        sys.argv=[script,args[1]];runpy.run_path(script,run_name='__main__');return
    if args==['--preflight']:
        import json
        print(json.dumps(bridge.readonly_preflight(ops),sort_keys=True,allow_nan=False));return
    if len(args)>=3 and args[:2]==['-m',ops.ENTRY] and args[2] in ('train','verify'):
        sys.argv=[ops.ENTRY,*args[2:]]
        runpy.run_module(ops.ENTRY,run_name='__main__');return
    raise ValueError('unregistered guarded science entry')
if __name__=='__main__':main()
