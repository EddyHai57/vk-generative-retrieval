"""Immutable source/config inventory for M2; code freeze follows one review."""
import hashlib,json,subprocess,sys
from pathlib import Path
BASE='b089b8f2da3b72d115f3df743f1e5461392270f6'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def build(root):
 root=Path(root);names=subprocess.check_output(['git','-C',str(root),'ls-tree','-r','--name-only',BASE],text=True).splitlines()
 for name in names:
  expected=hashlib.sha256(subprocess.check_output(['git','-C',str(root),'show',BASE+':'+name])).hexdigest()
  if sha(root/name)!=expected:raise ValueError('immutable M1 baseline changed: '+name)
 paths=sorted({*root.glob('src/vkgr/*.py'),*root.glob('scripts/seqm2*.py'),*root.glob('tests/test_seqm2*.py'),*root.glob('configs/seqm2/p-*.json')})
 return {'schema':'seqm2-freeze-v1','base':BASE,'reviewed_source_commit':subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),'files':{str(p.relative_to(root)):sha(p) for p in paths},'seeds':[42,2024,2025],'soft_optimizer_steps':100,'hard_coverage':'diagnostic_only','exact_only':True}
def verify(root,record):
 root=Path(root)
 if record['schema']!='seqm2-freeze-v1' or record['base']!=BASE or record['seeds']!=[42,2024,2025]:raise ValueError('freeze identity mismatch')
 for name,pin in record['files'].items():
  if sha(root/name)!=pin:raise ValueError('frozen file drift: '+name)
 actual=sorted({str(p.relative_to(root)) for p in [*root.glob('src/vkgr/*.py'),*root.glob('scripts/seqm2*.py'),*root.glob('tests/test_seqm2*.py'),*root.glob('configs/seqm2/p-*.json')]})
 if sorted(record['files'])!=actual:raise ValueError('freeze inventory incomplete')
 return record
if __name__=='__main__':
 root=Path(__file__).resolve().parents[1];path=root/'configs/seqm2/freeze-v1.json'
 if sys.argv[1:]==['--write']:
  value=build(root)
  with path.open('x') as f:json.dump(value,f,sort_keys=True,indent=2)
  print(json.dumps({'status':'written','files':len(value['files'])}))
 elif sys.argv[1:]==['--check']:verify(root,json.loads(path.read_text()));print('freeze_verified')
 else:raise SystemExit('use --write after review or --check')
