"""Guard the entire frozen deployment before entering M2 training/verification."""
import runpy,sys
sys.dont_write_bytecode=True
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import seqm2_ops as ops
if __name__=='__main__':
 ops.identity();ops.modules();ops.control_identity();a=sys.argv[1:]
 if len(a)<3 or a[:2]!=['-m',ops.ENTRY] or a[2] not in ('train','verify'):raise ValueError('unregistered science entry')
 sys.argv=[ops.ENTRY,*a[2:]];runpy.run_module(ops.ENTRY,run_name='__main__')
