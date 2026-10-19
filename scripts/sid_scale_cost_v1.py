"""Approved synthetic-only SID cost model and bounded own-child calibration.
No log-log fit, scientific checkpoint, complete mapping or raw-vector export.
"""
import os
THREAD_ENV=('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS')
for key in THREAD_ENV:os.environ[key]='1'
import argparse,collections,hashlib,json,math,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT/'src') not in sys.path:sys.path.insert(0,str(ROOT/'src'))
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
RUN_ID='sid-scale-cost-v1'
MIB=1024**2
TARGETS=(1250000,12470000)

def positive(*values):
 if any(type(v) is not int or v<1 for v in values):raise ValueError('positive plain integers required')

def bhkm_work(n,k=256,depth=3,dim=64,iters=10):
 positive(n,k,depth,dim,iters)
 if k<2 or depth>8:raise ValueError('finite branching/depth required')
 sizes={n:1};levels=[]
 for level in range(depth):
  edges=sum(count*size*min(k,size) for size,count in sizes.items())
  levels.append({'level':level+1,'nodes':sum(sizes.values()),'items':n,'edges':edges,'units':edges*dim*iters})
  nxt=collections.Counter()
  for size,count in sizes.items():
   children=min(k,size);q,r=divmod(size,children)
   if children-r:nxt[q]+=count*(children-r)
   if r:nxt[q+1]+=count*r
  sizes=dict(nxt)
 return {'n':n,'k':k,'depth':depth,'dimension':dim,'max_iter':iters,'levels':levels,'units':sum(row['units'] for row in levels),'data_bytes':n*64*4,'root_distance_bytes':n*min(k,n)*8,'measured':False}

def extrapolate(calibration,targets=TARGETS):
 bhkm=[r for r in calibration if r.get('method')=='bhkm' and r.get('status')=='COMPLETE']
 rq=[r for r in calibration if r.get('method')=='rqvae' and r.get('status')=='COMPLETE']
 for r in bhkm:
  if type(r.get('model_units')) is not int or r['model_units']<=0 or not math.isfinite(r['fit_seconds']) or r['fit_seconds']<=0:raise ValueError('completed measured BHKM units/time required')
 for r in rq:
  if type(r.get('steps')) is not int or r['steps']<=0 or not math.isfinite(r['fit_seconds']) or r['fit_seconds']<=0:raise ValueError('completed measured RQ steps/time required')
 bc=sum(r['fit_seconds'] for r in bhkm)/sum(r['model_units'] for r in bhkm) if bhkm else None
 rc=sum(r['fit_seconds'] for r in rq)/sum(r['steps'] for r in rq) if rq else None
 memory=[]
 for row in bhkm:
  values=[row.get('worker_peak_rss_bytes',0),row.get('sampled_peak_own_tree_rss_bytes',0)]
  if any(type(v) is not int or v<0 for v in values):raise ValueError('plain measured RSS byte counters required')
  observed=max(values)
  if observed:
   distance=bhkm_work(row['n'])['root_distance_bytes'];memory.append({'n':row['n'],'observed_peak_rss_bytes':observed,'root_distance_bytes':distance,'rss_per_root_distance_byte':observed/distance})
 rows=[]
 for n in targets:
  model=bhkm_work(n);steps=math.ceil(300000*n/120415)
  rows.append({'n':n,'data_bytes':model['data_bytes'],'bhkm_model_units':model['units'],'bhkm_seconds':bc*model['units'] if bc is not None else 'UNKNOWN','bhkm_per_calibration_seconds':[r['fit_seconds']/r['model_units']*model['units'] for r in bhkm],'bhkm_root_distance_bytes':model['root_distance_bytes'],'bhkm_peak_rss_bytes_scenarios':[math.ceil(r['rss_per_root_distance_byte']*model['root_distance_bytes']) for r in memory] if memory else 'UNKNOWN',
               'rq_fixed_steps':300000,'rq_fixed_steps_seconds':rc*300000 if rc is not None else 'UNKNOWN','rq_fixed_visits_steps':steps,'rq_fixed_visits_seconds':rc*steps if rc is not None else 'UNKNOWN'})
 return {'kind':'cost_model_projection_not_measurement','bhkm_memory_calibration':memory,'bhkm_seconds_per_unit':bc if bc is not None else 'UNKNOWN','rq_seconds_per_step':rc if rc is not None else 'UNKNOWN','targets':rows,
         'assumptions':['BHKM balanced effective min(K,node_size) branching and max_iter=10 units; calibration absorbs actual early convergence, distance, sorting and Python costs.',
                        'Projection assumes the calibrated coefficient, convergence, distribution and sorting/memory constants remain applicable; per-point estimates are scenarios, not confidence intervals.',
                        'RQ fixed steps holds 300000 updates; fixed visits scales steps by N/120415 (ceil), batch1024, with the same original L2/reset cadence.',
                        'RQ measured core seconds excludes initialization; initialization/generation and finalization are reported separately. No RSS slope fitted over the Torch baseline.',
                        'Data memory is analytically N*64*4 bytes; root distance is N*min(256,N)*8 bytes, not peak RSS.','BHKM memory scenarios multiply each measured peak/root-distance ratio by target root-distance bytes; allocator, tree and sorting composition are assumed transferable, not a bound. RQ Torch baseline is not used for a memory slope.']}

def terminal_status(*,wall_hit,rss_hit,returncode):
 if wall_hit:return 'CENSORED_WALL'
 if rss_hit:return 'CENSORED_RSS'
 return 'COMPLETE' if returncode==0 else 'FAILED'

def peak_rss_bytes(value,platform):return int(value if platform=='darwin' else value*1024)

def own_tree_rss(pid):
 pending=[pid];seen=set();total=0
 while pending:
  current=pending.pop()
  if current in seen:continue
  seen.add(current)
  if sys.platform=='linux':
   base=Path('/proc')/str(current)
   try:
    status=(base/'status').read_text();children=(base/'task'/str(current)/'children').read_text().split()
   except (FileNotFoundError,ProcessLookupError):continue
   rss=next((int(line.split()[1])*1024 for line in status.splitlines() if line.startswith('VmRSS:')),0);pending.extend(map(int,children))
  else:
   result=subprocess.run(['ps','-o','rss=','-p',str(current)],capture_output=True,text=True)
   rss=int(result.stdout.strip() or '0')*1024
   children=subprocess.run(['pgrep','-P',str(current)],capture_output=True,text=True).stdout.split();pending.extend(map(int,children))
  total+=rss
 return total,seen

def supervise(argv,*,wall_seconds,rss_mib):
 for value in (wall_seconds,rss_mib):
  if type(value) not in (int,float) or not math.isfinite(value) or value<=0:raise ValueError('positive finite child budgets required')
 start=time.monotonic();process=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
 try:
  wall_hit=rss_hit=False;peak=0;seen=set();terminated=False
  while process.poll() is None:
   elapsed=time.monotonic()-start
   if elapsed>=wall_seconds:wall_hit=True
   rss,pids=own_tree_rss(process.pid);peak=max(peak,rss);seen.update(pids)
   if rss>=rss_mib*MIB:rss_hit=True
   if wall_hit or rss_hit:
    if process.poll() is None:
     if os.getpgid(process.pid)!=process.pid:raise ValueError('only own new child session may be terminated')
     os.killpg(process.pid,signal.SIGTERM);terminated=True
     try:process.wait(timeout=2)
     except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=2)
    break
   time.sleep(min(.02,max(0,wall_seconds-elapsed)))
  stdout,stderr=process.communicate(timeout=3);elapsed=time.monotonic()-start
  wall_hit=wall_hit or elapsed>=wall_seconds
  result={'status':terminal_status(wall_hit=wall_hit,rss_hit=rss_hit,returncode=process.returncode),'pid':process.pid,'returncode':process.returncode,'wall_seconds':elapsed,'sampled_peak_own_tree_rss_bytes':peak,'rss_scope':'sum_RSS_own_child_tree_samples','observed_pids':sorted(seen),'terminated_own_group':terminated,'wall_cap_seconds':wall_seconds,'rss_cap_bytes':rss_mib*MIB}
  if len(stdout)>65536 or len(stderr)>65536:raise ValueError('bounded aggregate worker output required')
  if result['status']=='COMPLETE':
   try:payload=json.loads(stdout)
   except ValueError:result.update(status='FAILED',reason='worker aggregate JSON absent')
   else:
    if type(payload) is not dict or payload.get('status')!='COMPLETE':result.update(status='FAILED',reason='worker incomplete aggregate')
    else:
     reported=payload.get('worker_peak_rss_bytes')
     if type(reported) is not int or reported<0:result.update(status='FAILED',reason='worker peak RSS counter absent')
     elif reported>=rss_mib*MIB:result.update(status='CENSORED_RSS',reported_peak_rss_bytes=reported)
     else:result['result']=payload
  if stderr:result['stderr_tail']=stderr.decode(errors='replace')[-4000:]
  return result
 finally:
  if process.poll() is None:
   try:
    if os.getpgid(process.pid)!=process.pid:raise ValueError('only own new session may be cleaned up')
    os.killpg(process.pid,signal.SIGTERM)
    try:process.wait(timeout=2)
    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=2)
   except ProcessLookupError:process.wait(timeout=2)
  for stream in (process.stdout,process.stderr):stream.close()

def synthetic(n):
 from scripts.sid_scale_benchmark import synthetic_blocks
 import numpy as np
 return np.concatenate(list(synthetic_blocks(n,block=2048,seed=42)))

def producer():
 from importlib.metadata import version
 names=('scripts/sid_scale_cost_v1.py','scripts/sid_scale_benchmark.py','src/vkgr/sid.py','src/vkgr/sid_rqvae.py','src/vkgr/sid_rqvae_r.py','src/vkgr/sid_tok_audit.py','src/vkgr/generative_data.py','src/vkgr/data_audit.py','src/vkgr/retrieval_data.py','src/vkgr/manifest.py','src/vkgr/config.py','uv.lock')
 return {'files':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names},'python':sys.version,'dependencies':{name:version(name) for name in ('numpy','torch','PyYAML')},'thread_env':{key:os.environ.get(key) for key in THREAD_ENV}}

def worker(method,n,*,k,steps,engineering):
 if type(engineering) is not bool or method not in ('bhkm','rqvae'):raise ValueError('explicit synthetic worker kind required')
 positive(n,k,steps)
 if engineering:
  if n>1024 or k>16 or steps>8:raise ValueError('tiny local selftests only')
 elif sys.platform!='linux' or os.environ.get('VKGR_RUN_ID')!=RUN_ID or k!=256 or steps!=3000 or n not in ((250000,300000) if method=='bhkm' else (120415,)):raise ValueError('large calibration only in formal Linux own queue job')
 import resource,numpy as np
 from scripts.sid_scale_benchmark import _numpy_threads
 blas_threads=_numpy_threads()
 start=time.monotonic();values=synthetic(n);generation=time.monotonic()-start;hash_start=time.monotonic();data_sha=hashlib.sha256(values.astype('<f4',copy=False).tobytes(order='C')).hexdigest();hash_seconds=time.monotonic()-hash_start;fit_start=time.monotonic()
 if method=='bhkm':
  from vkgr import sid
  fitted=sid.fit(np.arange(n,dtype='uint32'),values,branching=k,depth=3,seed=42,max_iter=10)
  fit_seconds=time.monotonic()-fit_start;layers=[]
  for layer in range(3):
   nodes=[row for row in fitted.nodes if len(row['prefix'])==layer]
   layers.append({'layer':layer+1,'nodes':len(nodes),'iterations_sum':sum(row['iterations'] for row in nodes),'converged_nodes':sum(row['converged'] for row in nodes),'actual_units':sum(sum(row['sizes'])*len(row['sizes'])*64*row['iterations'] for row in nodes)})
  result={'method':method,'status':'COMPLETE','n':n,'model_units':bhkm_work(n,k=k)['units'],'actual_units':sum(row['actual_units'] for row in layers),'actual_layers':layers,'fit_seconds':fit_seconds,'suffix_vocabulary':fitted.suffix_vocabulary}
 else:
  import torch
  from vkgr import sid_rqvae_r as rq
  torch.set_num_threads(1);torch.set_num_interop_threads(1)
  fitted=rq.fit_r(values,K=k,arm='l2',seed=42,max_steps=steps,batch_size=min(1024,n),device='cpu',wall_cap_seconds=600,project=False,diagnostics=False)
  if fitted['status']!='complete':raise ValueError('RQ calibration failed: '+fitted['status'])
  result={'method':method,'status':'COMPLETE','n':n,'steps':fitted['step'],'fit_seconds':fitted['elapsed'],'initialization_and_finalization_seconds':time.monotonic()-fit_start-fitted['elapsed'],'optimizer':'original_L2_Adagrad_reset','projection_stop':False}
 return {**result,'synthetic_features_sha256':data_sha,'input_hash_seconds':hash_seconds,'generation_seconds':generation,'worker_wall_seconds':time.monotonic()-start,'worker_peak_rss_bytes':peak_rss_bytes(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,sys.platform),'blas_threads':blas_threads,'producer':producer(),'engineering':engineering}

def check_l2_source(root=ROOT):
 expected={'src/vkgr/sid_rqvae.py':'52173bea9b80ffeb40e25943e3d7069ec53f9bd34a4509c97278194714065067','src/vkgr/sid_rqvae_r.py':'c786848ab85c6ce2a593ccdfc626688484ae58d2ee0406c889a5ad48c5f54392'}
 for name,pin in expected.items():
  if hashlib.sha256((Path(root)/name).read_bytes()).hexdigest()!=pin:raise ValueError('original fixed L2 reference bytes differ')
 return {'origin_commit':'80f7eed179981c65094c628744099a181a81d907','files':expected}

def run_benchmark(request,*,clock=time.monotonic,supervisor=supervise):
 req=validate_request(request)
 if not req['engineering'] and (sys.platform!='linux' or os.environ.get('VKGR_RUN_ID')!=RUN_ID):raise ValueError('formal benchmark only under own Linux queue job')
 fixed=check_l2_source();start=clock();records=[];failed=False
 stages=[('bhkm',n) for n in req['bhkm_sizes']]+[('rqvae',req['rq_n'])]
 for method,n in stages:
  remaining=req['overall_seconds']-(clock()-start)
  if failed or remaining<=0:
   records.append({'method':method,'n':n,'status':'NOT_RUN_FAILURE' if failed else 'NOT_RUN_BUDGET'});continue
  argv=[sys.executable,'-B',str(Path(__file__).resolve()),'--worker',method,'--n',str(n),'--k',str(req['branching']),'--steps',str(req['rq_steps'])]
  if req['engineering']:argv.append('--engineering')
  row=supervisor(argv,wall_seconds=min(remaining,req['point_seconds'] if method=='bhkm' else 600),rss_mib=req['rss_mib'])
  record={**row,'method':method,'n':n}
  if row['status']=='COMPLETE':
   payload=row['result']
   if payload['method']!=method or payload['n']!=n or payload['engineering'] is not req['engineering'] or payload['producer']!=producer():raise ValueError('worker source/request identity differs')
   record.update(payload)
  records.append(record);failed=row['status']=='FAILED'
 status='FAILED' if failed else 'COMPLETE' if all(r['status']=='COMPLETE' for r in records) else 'INCOMPLETE'
 return {'schema':'sid-scale-cost-report-v1','status':status,'engineering':req['engineering'],'request':req,'records':records,'projection':extrapolate(records),'elapsed_seconds':clock()-start,'fixed_l2':fixed,'producer':producer(),'synthetic_only':True,'test_read':False,'checkpoint_written':False,'mapping_exported':False}

def default_request():return {'schema':'sid-scale-cost-request-v1','engineering':False,'bhkm_sizes':[250000,300000],'rq_n':120415,'rq_steps':3000,'branching':256,'dimension':64,'depth':3,'seed':42,'point_seconds':3600,'rss_mib':16384,'overall_seconds':8970}

def validate_request(req):
 base=default_request()
 if type(req) is not dict or set(req)!=set(base) or type(req['engineering']) is not bool:raise ValueError('exact benchmark request fields required')
 if req['engineering']:
  if any(type(v) is not int or not 1<=v<=1024 for v in req['bhkm_sizes']) or type(req['rq_n']) is not int or not 1<=req['rq_n']<=1024 or type(req['rq_steps']) is not int or not 1<=req['rq_steps']<=8 or not 2<=req['branching']<=16:raise ValueError('tiny local benchmark only')
 else:
  for key in set(base)-{'overall_seconds','point_seconds','rss_mib'}:
   if type(req[key]) is not type(base[key]) or req[key]!=base[key]:raise ValueError('fixed formal calibration points and recipe')
 for key,cap in (('point_seconds',3600),('rss_mib',16384),('overall_seconds',9000)):
  if type(req[key]) not in (int,float) or not math.isfinite(req[key]) or not 0<req[key]<=cap:raise ValueError('approved finite benchmark cap')
 return req

def hash_file(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def source_identity(engineering):
 identity=producer();commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip();dirty=bool(subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain','--untracked-files=all'],text=True))
 if not engineering:
  if dirty:raise ValueError('formal benchmark requires clean committed source')
  for name,pin in identity['files'].items():
   data=subprocess.check_output(['git','-C',str(ROOT),'show',commit+':'+name])
   if hashlib.sha256(data).hexdigest()!=pin:raise ValueError('fixed source Git bytes differ')
 return {'commit':commit,'dirty':dirty,**identity}

def bind_parent(output,request,pin,argv,source,engineering):
 binding={'run_id':None,'run_manifest_sha256':None,'request_file_sha256':pin,'source_commit':source['commit']}
 if engineering:return binding
 out=Path(output);record=out.parent/'run.json'
 if os.environ.get('VKGR_RUN_ID')!=RUN_ID or out.parent.name!=RUN_ID or out.name!='out':raise ValueError('formal original queue run required')
 from vkgr import generative_data as gd
 meta=gd._json(gd._no_symlink(record));a=meta.get('argv',[])
 if meta.get('run_id')!=RUN_ID or meta.get('git',{}).get('commit')!=source['commit'] or meta['git'].get('dirty') is not False or Path(meta.get('cwd','')).resolve()!=ROOT or meta.get('config',{}).get('path')!=str(request) or meta['config'].get('sha256')!=pin:raise ValueError('original benchmark source/config/cwd differs')
 if len(a)<3 or not Path(a[0]).is_absolute() or Path(a[0]).resolve()!=Path(sys.executable).resolve() or a[1:3]!=['-B',str(ROOT/'scripts/sid_scale_cost_v1.py')] or a[3:]!=list(argv):raise ValueError('original benchmark actual argv differs')
 binding.update(run_id=RUN_ID,run_manifest_sha256=hash_file(record));return binding

def main(argv=None):
 raw=list(sys.argv[1:] if argv is None else argv);p=argparse.ArgumentParser(allow_abbrev=False)
 p.add_argument('--worker',choices=('bhkm','rqvae'));p.add_argument('--n',type=int);p.add_argument('--k',type=int);p.add_argument('--steps',type=int);p.add_argument('--engineering',action='store_true')
 p.add_argument('--request');p.add_argument('--request-sha');p.add_argument('--output');a=p.parse_args(raw)
 try:
  if a.worker:
   if any(x is not None for x in (a.request,a.request_sha,a.output)) or any(x is None for x in (a.n,a.k,a.steps)):raise ValueError('exact worker-only fields required')
   print(json.dumps(worker(a.worker,a.n,k=a.k,steps=a.steps,engineering=a.engineering),sort_keys=True,allow_nan=False));return 0
  if any(x is None for x in (a.request,a.request_sha,a.output)) or any(x is not None for x in (a.n,a.k,a.steps)) or a.engineering:raise ValueError('exact parent request fields required; engineering is bound in request')
  from vkgr import generative_data as gd
  path=gd._checked_file(a.request,a.request_sha);req=validate_request(gd._json(path));out=Path(a.output)
  if not out.is_absolute() or '..' in out.parts or not out.parent.is_dir() or out.exists():raise ValueError('exclusive absolute benchmark output required')
  gd._no_symlink(out);source=source_identity(req['engineering']);binding=bind_parent(out,path,a.request_sha,raw,source,req['engineering'])
  report=run_benchmark(req);report.update(binding=binding,source=source)
  out.mkdir()
  with (out/'report.json').open('x') as stream:json.dump(report,stream,sort_keys=True,allow_nan=False)
  result={'status':('engineering_benchmark_reported' if req['engineering'] else 'benchmark_reported') if report['status']!='FAILED' else 'ALERT','report_status':report['status'],'report_sha256':hash_file(out/'report.json'),'synthetic_only':True,**binding}
  print(json.dumps(result,sort_keys=True,allow_nan=False));return 3 if report['status']=='FAILED' else 0
 except (ValueError,KeyError,TypeError,OSError,RuntimeError) as exc:
  print('SID scale cost gate failed: '+str(exc),file=sys.stderr);return 1
if __name__=='__main__':raise SystemExit(main())
