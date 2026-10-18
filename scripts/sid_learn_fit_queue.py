"""Metadata-only unique S2 fit-stage publication; no retry or scientific work.
The CUDA reservation is explicitly unmeasured once; fit/replay use measured proof.
"""
import copy,json,sys
from pathlib import Path
sys.dont_write_bytecode=True
ops=None
if '__file__' in globals():
 sys.path.insert(0,str(Path(__file__).resolve().parent))
 import sid_learn_ops as ops

def runtime(req):
 global ops
 if ops is None:
  sys.path.insert(0,str(Path(req['deployment'])/'scripts'))
  import sid_learn_ops as loaded_ops
  ops=loaded_ops
 return ops
STAGES={'sid-learn-input-preflight-v1':'preflight','sid-learn-fit-cuda-v1':'cuda-preview','sid-learn-fit-v1':'fit','sid-learn-fit-verify-v1':'verify'}

def validate(req):
 job=req['job'];a=job['argv'];rid=job['run_id'];root=Path(req['deployment'])
 if rid not in STAGES or type(a) is not list or len(a)<5 or any(type(x) is not str for x in a) or not Path(a[0]).is_absolute() or Path(a[0]).resolve()!=Path(sys.executable).resolve() or a[1:5]!=['-B','-m','vkgr.sid_learn_run',STAGES[rid]] or a.count('-m')!=1 or '--engineering' in a or any(x.startswith('--engineering-') for x in a):raise ValueError('fixed formal S2 entry/stage required')
 if job['cpu_cores']!=1 or job['no_admission'] is not False or job['cwd']!=str(root) or job['config']!=str(root/'configs/sid_learn/fit-s42.json') or job['wall_hours']!=4.:raise ValueError('fixed fit-stage resources/config required')
 required={'CUBLAS_WORKSPACE_CONFIG=:4096:8','OMP_NUM_THREADS=1','MKL_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1'}
 if not required<=set(job['env']) or any(sum(e.startswith(k+'=') for e in job['env'])!=1 for k in ('CUBLAS_WORKSPACE_CONFIG','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')):raise ValueError('deterministic single-thread environment required')
 if ops.flag(a,'--config')!=job['config'] or ops.flag(a,'--output')!=str(Path(req['runs'])/rid/'out') or Path(req['root'])/'runs'!=Path(req['runs']):raise ValueError('own config/output scope required')
 if STAGES[rid]=='preflight':
  if job['vram_mib']!=256:raise ValueError('readonly preflight reservation')
 elif STAGES[rid]=='cuda-preview':
  if req.get('memory_status')!='not_measured' or type(job['vram_mib']) is not int or not 256<=job['vram_mib']<=39168:raise ValueError('initial CUDA reservation must be explicitly not_measured')
 else:
  proof=req.get('cuda_proof',{})
  if req.get('memory_status')!='measured' or proof.get('status')!='cuda_engineering_verified' or job['vram_mib']!=ops.validate_vram(proof):raise ValueError('context-inclusive measured fit reservation required')
 return job

def check_fresh(req):
 job=validate(req);rid=job['run_id'];root=Path(req['root'])
 if ops.exists(root/'queue',root/'runs',rid) or any(root.glob('queue/*/*-'+rid+'.reason')) or (Path(req['ops'])/('intent-'+rid+'.json')).exists():raise FileExistsError('never resubmit '+rid)

def submit(req,*,prepare=None,publisher=None):
 check_fresh(req);job=req['job']
 (prepare or ops.prepare_envelope)(req,job)
 ops.new(Path(req['ops'])/('intent-'+job['run_id']+'.json'),{'run_id':job['run_id'],'job_sha256':ops.digest(job)})
 (publisher or ops.publish)(Path(req['root'])/'queue',Path(req['runs']),job)
 return {'status':'written_unconfirmed','run_id':job['run_id']}

def derive_verify(req,manifest_sha,producer_sha):
 job=copy.deepcopy(req['job']);runs=Path(req['runs']);rid='sid-learn-fit-verify-v1';a=job['argv']
 if job['run_id']!='sid-learn-fit-v1' or a[4]!='fit':raise ValueError('derive only from unique original fit')
 job.update(run_id=rid,wall_hours=4.,disk_gib=1)
 a[4]='verify';ops.replace(a,'--output',runs/rid/'out')
 bundle=str(runs/'sid-learn-fit-v1/out/payload/run_manifest.json');producer=str(runs/'sid-learn-fit-v1/run.json')
 a.extend(['--bundle',bundle,'--sha',manifest_sha,'--producer-run',producer,'--producer-run-sha',producer_sha])
 job['inputs']=list(dict.fromkeys([*job.get('inputs',[]),bundle,producer]))
 return job

def main():
 req=json.loads(sys.stdin.read(2*1024*1024+1));runtime(req).identity(req)
 if 'scripts/sid_learn_fit_queue.py' not in req['files']:raise ValueError('queue bridge must be deployment-pinned')
 if req['action']=='launch-driver':
  import sid_learn_fit_driver
  print(json.dumps(sid_learn_fit_driver.launch(req),sort_keys=True));return
 check_fresh(req);pin=ops.digest(req['job']);dry=Path(req['ops'])/('dry-'+req['job']['run_id']+'.json')
 if req['action']=='dry':
  proof={'status':'dry_ok','job_sha256':pin,'run_id':req['job']['run_id'],'envelope':ops.prepare_envelope(req,req['job'])};ops.new(dry,proof);result=proof
 elif req['action']=='submit':
  if ops.load(dry).get('job_sha256')!=pin:raise ValueError('original dry job binding differs')
  result=submit(req)
 else:raise ValueError('dry/submit metadata only')
 print(json.dumps(result,sort_keys=True))
if __name__=='__main__':main()
