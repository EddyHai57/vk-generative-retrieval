"""Synthetic CUDA train/verify with frozen real shapes VRAM probe; no real inputs."""
import json
import os
from pathlib import Path
import subprocess
import sys

def main():
    out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
    env={**os.environ,'PYTHONPATH':'src','OMP_NUM_THREADS':'1'}
    probe=subprocess.run([sys.executable,'-B','scripts/seq_combo_stack_vram_smoke.py','--device','cuda'],capture_output=True,text=True,env=env,timeout=120)
    if probe.returncode:raise ValueError('VRAM probe failed: '+probe.stderr[-500:])
    memory=json.loads(probe.stdout.splitlines()[-1]);row=memory['variants'][0]
    if not row['finite_loss']:raise ValueError('nonfinite synthetic probe')
    from vkgr import retrieval_seq_combo_stack_run as r
    import torch
    torch.set_num_threads(1)
    config={**r.default_config(),'max_epochs':6,'min_epochs':2,'batch_size':16}
    data=r.synthetic_data();value=r.run_session(out/'synthetic',data,config,'cuda')
    proof=r.verify_run(out/'synthetic/manifest.json',value['manifest_sha256'],data,config,'cuda')
    if proof['status']!='verified':raise ValueError('synthetic CUDA verify not verified')
    result={'status':'cuda_engineering_verified','real_data_read':False,'commit':value['binding']['source']['git_commit'],'source_sha':value['binding']['source']['source_sha'],'vram':memory,'synthetic_verify':proof,'manifest_sha256':value['manifest_sha256']}
    with (out/'engineering.json').open('x') as f:json.dump(result,f,sort_keys=True)
    print(json.dumps(result,sort_keys=True))
if __name__=='__main__':main()
