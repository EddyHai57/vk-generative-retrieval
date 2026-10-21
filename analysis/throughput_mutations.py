"""Isolated local assertion mutations, one fresh copy and cache prefix per case."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

CASES={
 'tensor_snapshot':('throughput_tensors.py',"json.dumps(rows, sort_keys=True, separators=(',', ':'), allow_nan=False) != row_key","rows != expected",'test_throughput_tensors.py'),
 'tensor_memory':('throughput_tensors.py','(config.max_in_flight_bytes or config.memory_limit_bytes) and tensor_bytes_bound is None','config.max_in_flight_bytes and tensor_bytes_bound is None','test_throughput_tensors.py'),

 'tensor_order':('throughput_tensors.py','rows = _STORE.rows(indices, length=length)','rows = _STORE.rows(list(reversed(indices)), length=length)','test_throughput_tensors.py'),
 'tensor_padding':('throughput_tensors.py','_BUILDER.build(part)','_BUILDER.build(part + part)','test_throughput_tensors.py'),
 'tensor_namespace':('throughput_tensors.py',"namespace != self.namespace or self.validator.identity() != self.namespace","False",'test_throughput_tensors.py'),
 'tensor_arguments':('throughput_tensors.py','or not self.validator.validate_call(rows, args, kwargs)','or False','test_throughput_tensors.py'),
 'sequence_future':('throughput_sequences.py',"self.items[cut:j].astype(int).tolist()","self.items[cut:j+1].astype(int).tolist()",'test_throughput_sequences.py'),
 'sequence_order':('throughput_sequences.py','for index in indices:','for index in reversed(indices):','test_throughput_sequences.py'),
 'sequence_mask':('throughput_sequences.py',"past=sorted(set(map(int,self.items[start:j][self.likes[start:j]])))",'past=[]','test_throughput_sequences.py'),
 'sequence_key':('throughput_sequences.py',"if p.is_symlink()or h.hexdigest()!=records[p.name]:","if False:",'test_throughput_sequences.py'),
 'worker_rng':('throughput_workers.py','before=rng_state();result=function(value)','before=rng_state();random.random();result=function(value)','test_throughput_workers.py'),
 'worker_order':('throughput_workers.py','yield result','yield -1','test_throughput_workers.py'),
 'worker_cursor':('throughput_workers.py','probe=copy.deepcopy(cursor)','probe=cursor','test_throughput_workers.py'),
 'bhkm_tie':('throughput_bhkm.py','np.lexsort((child, np.repeat(item_ids, k), distances.ravel()))','np.lexsort((-child, np.repeat(item_ids, k), distances.ravel()))','test_throughput_bhkm.py'),
 'bhkm_capacity':('throughput_bhkm.py','capacity[:n % k] += 1','capacity[:n % k] += 0','test_throughput_bhkm.py'),
 'bhkm_dimension':('throughput_bhkm.py','for dimension in range(rows.shape[1]):','for dimension in reversed(range(rows.shape[1])):','test_throughput_bhkm.py'),
 'bhkm_rng':('throughput_bhkm.py','init = rng.choice','rng.random(); init = rng.choice','test_throughput_bhkm.py'),
 'week_order':('throughput_weekly.py','chain.from_iterable(checked())','chain.from_iterable(reversed(list(checked())))','test_throughput_weekly.py'),
 'week_holdout':('throughput_weekly.py','0<=w<=25','0<=w<=26','test_throughput_weekly.py'),
 'exact_tie':('throughput_exact.py','np.lexsort((combined_ids,-combined_scores))','np.lexsort((-combined_ids,-combined_scores))','test_throughput_exact.py'),
 'exact_precision':('throughput_exact.py','dtype=np.float64,optimize=False','dtype=np.float32,optimize=False','test_throughput_exact.py'),
 'decode_order':('throughput_decode.py','output.append(value)','output.insert(0,value)','test_throughput_decode.py'),
 'decode_identity':('throughput_decode.py',"if not callable(getattr(_SESSION, 'identity', None)) or _SESSION.identity() != state_sha:","if False:",'test_throughput_decode.py'),
 'probe_limit':('throughput_probe.py',"memory_limit_bytes=int(ram*.5)","memory_limit_bytes=int(ram*.9)",'test_throughput_probe.py'),
 'probe_gpu':('throughput_probe.py','decode_workers=2 if vram>=45000 else 1','decode_workers=4 if vram>=45000 else 2','test_throughput_probe.py'),
}


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--group',choices=('sequence','worker','bhkm','week','exact','decode','probe','tensor'),required=True);a=p.parse_args(argv)
    root=Path(__file__).resolve().parents[1];out=Path(a.output);out.mkdir(parents=True,exist_ok=False);records=[]
    cases=[('baseline',None)]+[(k,v)for k,v in CASES.items()if k.startswith(a.group)]
    for label,spec in cases:
        temp=Path(tempfile.mkdtemp(prefix='vkgr-throughput-'+label+'-'))
        shutil.copytree(root/'src',temp/'src');(temp/'tests').mkdir()
        for p in (root/'tests').glob('test_throughput_*.py'):shutil.copy2(p,temp/'tests'/p.name)
        shutil.copy2(root/'tests/test_tensor_cache.py',temp/'tests/test_tensor_cache.py')
        (temp/'tests/fixtures').mkdir()
        for p in (root/'tests/fixtures').glob('throughput_frozen_*.py'):shutil.copy2(p,temp/'tests/fixtures'/p.name)
        shutil.copytree(root/'configs/throughput',temp/'configs/throughput')
        (temp/'analysis').mkdir()
        shutil.copy2(root/'analysis/throughput_probe.py',temp/'analysis/throughput_probe.py')
        if spec:
            file,before,after,pattern=spec;target=temp/'src/vkgr'/file;text=target.read_text()
            if before not in text:raise ValueError('absent mutation anchor')
            target.write_text(text.replace(before,after))
        else:pattern='test_throughput_'+{'sequence':'sequences','worker':'workers','bhkm':'bhkm','week':'weekly','exact':'exact','decode':'decode','probe':'probe','tensor':'tensors'}[a.group]+'.py'
        env={**os.environ,'PYTHONPATH':str(temp/'src')+os.pathsep+str(temp/'tests'),
            'PYTHONDONTWRITEBYTECODE':'1','PYTHONPYCACHEPREFIX':str(temp/'unique-pycache'),'VKGR_CACHE_PREFIX':str(temp/'unique-cache')}
        with (out/(label+'.log')).open('xb')as f:
            r=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s','tests','-p',pattern,'-v'],cwd=temp,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=120)
        text=(out/(label+'.log')).read_text();killed='FAIL:'in text and 'AssertionError'in text
        # A mutation that makes the RNG contract fail-closed is an expected
        # ValueError targeted by the worker tests, never a syntax/import kill.
        if label=='tensor_memory':killed='input/output reservation exceeds in-flight cap'in text and 'ERROR:'in text
        if label=='worker_rng':killed='worker consumed random state'in text and 'ERROR:'in text
        if label=='worker_cursor':killed='live cursor differs from planned ordinal'in text and 'ERROR:'in text
        if label=='bhkm_capacity':killed='external greedy assignment left rows unassigned'in text and 'ERROR:'in text
        ok=r.returncode==0 if spec is None else r.returncode!=0 and killed
        records.append({'case':label,'returncode':r.returncode,'accepted':ok,'isolated_copy':True,'unique_cache':True,'bytecode_disabled':True})
    receipt={'group':a.group,'results':records,'pass':all(r['accepted']for r in records)}
    with (out/'receipt.json').open('x')as f:json.dump(receipt,f,indent=2)
    print(json.dumps(receipt));return 0 if receipt['pass']else 1


if __name__=='__main__':raise SystemExit(main())
