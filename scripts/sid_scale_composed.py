"""Formal two-component synthetic cost wiring; old cost_v1 and L2 stay unchanged."""
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
for _path in (ROOT,ROOT/'src'):
    if str(_path) not in sys.path:sys.path.insert(0,str(_path))
from scripts import sid_scale_cost_v1 as base
from scripts import sid_scale_rq_components as components
import argparse,hashlib,json,math,platform,subprocess,time
THREAD_ENV=base.THREAD_ENV
RUN_ID=base.RUN_ID
PF_ID='sid-scale-components-preflight-v1'
TARGETS=base.TARGETS
HISTORICAL_REFS={'g-tok-t1r-fit': {'run': {'path': '/workspace/vkgr-infra01-root/runs/g-tok-t1r-fit/run.json', 'sha256': 'f7272ac3df3333c9466b59517737deb93da1241ebbec744fd6f8032bede0918a'}, 'terminal': {'path': '/workspace/vkgr-infra01-root/runs/g-tok-t1r-fit/terminal.json', 'sha256': 'ae82b35c15226ad8961efe2c6b496654680cac36997bb900786a4137c30653be'}, 'verify_terminal': {'path': '/workspace/vkgr-infra01-root/runs/g-tok-t1r-fit-verify/terminal.json', 'sha256': 'f166c9fdeba4869982db79d12a6437f8030a122e542926dc4356796c4353954d'}, 'verify_stdout': {'path': '/workspace/vkgr-infra01-root/runs/g-tok-t1r-fit-verify/logs/stdout.log', 'sha256': '5331049d6f03f56c39c006b11d96c2bcbe5b141cf194e7e948be0a1c879d0424'}, 'run_manifest': {'path': '/workspace/vkgr-infra01-root/runs/g-tok-t1r-fit/out/run_manifest.json', 'sha256': 'ca4f357ce469c519a0cd4836d79c4714d3707d3317f050108b43b43995615a9f'}, 'tokenizer': {'path': '/workspace/vkgr-infra01-root/runs/g-tok-t1r-fit/out/tokenizer.json', 'sha256': '36150f5bd929fdf26d1cf60930ab62a5661c03486fa79c24bd836c9943b1fc6b'}}, 'g-tok-t2r-fit': {'run': {'path': '/workspace/vkgr-infra01-root/runs/g-tok-t2r-fit/run.json', 'sha256': '86c50b876d4ee13c036b29c54a03bb9a0b63b4c97d234cf8f8399d035451934f'}, 'terminal': {'path': '/workspace/vkgr-infra01-root/runs/g-tok-t2r-fit/terminal.json', 'sha256': '5aa446855b59550544a7b182bea8a6f7a81a5bbb767de14511dd25e5fddbc7f6'}, 'verify_terminal': {'path': '/workspace/vkgr-infra01-root/runs/g-tok-t2r-fit-verify/terminal.json', 'sha256': 'a3da1f8f9b29a88404e9099708e1945c76efaea9b9de55ea8903a6eaf5882b82'}, 'verify_stdout': {'path': '/workspace/vkgr-infra01-root/runs/g-tok-t2r-fit-verify/logs/stdout.log', 'sha256': '1363505f03f9e759454f2cacc031af286f13349870ea20b9b4e7a06638b71263'}, 'run_manifest': {'path': '/workspace/vkgr-infra01-root/runs/g-tok-t2r-fit/out/run_manifest.json', 'sha256': '34d852e2e680ca8741ed5724157c535f5d7580be117950a34a4f6fc9f530cbea'}, 'tokenizer': {'path': '/workspace/vkgr-infra01-root/runs/g-tok-t2r-fit/out/tokenizer.json', 'sha256': 'd4256dfc437e9c6d4234feb0d74e0150f051f89af833ccef96a845554a517b87'}}}

def require(ok,message):
    if not ok:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def default_request():
    return {**base.default_request(),'schema':'sid-scale-components-request-v1','usage_sizes':[120415,250000,300000],'history_refs':HISTORICAL_REFS}
def validate_request(req):
    fixed=default_request()
    require(type(req) is dict and set(req)==set(fixed) and type(req['engineering']) is bool,'exact composed request required')
    old={k:req[k] for k in base.default_request()};old['schema']='sid-scale-cost-request-v1';base.validate_request(old)
    require(req['schema']==fixed['schema'] and req['history_refs']==HISTORICAL_REFS,'pinned historical sources required')
    if req['engineering']:
        require(type(req['usage_sizes']) is list and len(req['usage_sizes']) in (2,3) and len(set(req['usage_sizes']))==len(req['usage_sizes']) and all(type(n) is int and 1<=n<=1024 for n in req['usage_sizes']),'tiny distinct engineering usage sizes only')
    else:
        require(type(req['bhkm_sizes']) is list and all(type(n) is int for n in req['bhkm_sizes']) and req['bhkm_sizes']==fixed['bhkm_sizes'],'fixed plain-integer BHKM points required')
        require(type(req['usage_sizes']) is list and all(type(n) is int for n in req['usage_sizes']) and req['usage_sizes']==fixed['usage_sizes'],'fixed plain-integer independent usage points required')
    return req

def projection(records,histories,source):
    bhkm=base.extrapolate([r for r in records if r['method']=='bhkm'])
    core=[r for r in records if r['method']=='core' and r['status']=='COMPLETE']
    usage=[r for r in records if r['method']=='usage' and r['status']=='COMPLETE']
    result={'kind':'cost_model_projection_not_measurement','bhkm':bhkm,'rq_targets':'UNKNOWN','rq_selfcheck_passed':False,'historical_selfchecks':[],
            'assumptions':['Synthetic calibrated distribution/convergence/constants are assumed transferable.',
                           'Core calibration K256 reused for historical K16; this approximation is subject to the unchanged factor-two rejection.',
                           'Historical CUDA/diagnostics/project and end-to-end job wall differ from CPU core+usage excluding setup/export; differences do not relax factor-two rejection.',
                           'Item forward counts are not wall time. Separately measured constants determine wall shares.']}
    if len(core)!=1 or core[0].get('usage_calls')!=0 or core[0].get('steps',source['usage_interval'])>=source['usage_interval'] or len(usage)!=3 or {r.get('n') for r in usage}!={120415,250000,300000} or any(r.get('usage_calls')!=1 for r in usage):
        result['alert']='incomplete_or_contaminated_components';return result
    c=core[0]['core_seconds_per_step'];components.positive_seconds(c)
    fitted=components.usage_constant([{'n':r['n'],'seconds':r['seconds']} for r in usage],k=256,d=32,levels=3)
    result.update(core_seconds_per_step=c,usage_regression=fitted)
    for h in histories:
        composition=components.compose(n=h['n'],steps=h['steps'],usage_interval=h['usage_interval'],k=h['k'],d=h['d'],levels=h['levels'],core_seconds_per_step=c,assign_seconds_per_unit=fitted['assign_seconds_per_unit'])
        check=components.historical_check(composition['total_fit_seconds'],h['wall_seconds'])
        result['historical_selfchecks'].append({'run_id':h['run_id'],'composition':composition,'check':check})
    if len(histories)!=2 or any(h['check']['status']!='passed' for h in result['historical_selfchecks']):
        result['alert']='historical_factor_exceeds_two';return result
    result['rq_selfcheck_passed']=True;result['rq_targets']=[]
    for n in TARGETS:
        row={'n':n,'extrapolation':True}
        for name,steps in [('fixed_steps',300000),('fixed_visits',(300000*n+120415-1)//120415)]:
            row[name]=components.compose(n=n,steps=steps,usage_interval=source['usage_interval'],k=256,d=32,levels=3,core_seconds_per_step=c,assign_seconds_per_unit=fitted['assign_seconds_per_unit'])
        result['rq_targets'].append(row)
    return result

def safe_metadata(path):
    p=Path(path)
    require(p.is_absolute() and '..' not in p.parts and not any(x.is_symlink() for x in (p,*p.parents)),'absolute nonsymlink metadata required')
    require(p.is_file() and p.stat().st_size<=2*1024*1024 and not any(x in p.name.lower() for x in ('checkpoint','embedding','weights','mapping')),'bounded nonmodel metadata required')
    return p

def pinned_read(ref,text=False):
    p=safe_metadata(ref['path']);data=p.read_bytes();require(hashlib.sha256(data).hexdigest()==ref['sha256'],'historical metadata pin differs')
    return data.decode() if text else json.loads(data)

def history_entry(rid,manifest,tokenizer,meta,terminal):
    require(rid in HISTORICAL_REFS and meta['run_id']==terminal['run_id']==rid,'historical identity differs')
    require(manifest['status']=='complete' and manifest['engineering'] is False and manifest['catalog_items']==120415 and manifest['branching']==(256 if rid=='g-tok-t1r-fit' else 16),'historical fit scope differs')
    require(all(manifest['producer'].get(k)==v for k,v in components.PINS.items()),'historical original L2 bytes differ')
    require(terminal['status']=='succeeded' and terminal['exit_code']==0 and terminal['wall_cap_exceeded'] is False,'historical terminal failed')
    components.positive_seconds(terminal['wall_seconds'])
    cfg=tokenizer['config'];fit=cfg['fit'];effective=cfg['effective_prereg']
    require(cfg['arm']=='l2' and fit['max_steps']==300000 and effective['latent_dim']==32 and effective['levels']==3,'historical L2 parameters differ')
    components.positive_ints(fit['eval_every'])
    a=meta['argv'];device=a[a.index('--device')+1] if '--device' in a else 'UNKNOWN'
    return {'run_id':rid,'n':manifest['catalog_items'],'steps':fit['max_steps'],'k':manifest['branching'],'d':effective['latent_dim'],'levels':effective['levels'],
            'usage_interval':fit['eval_every'],'effective_prereg_default_interval':effective['eval_every'],'wall_seconds':terminal['wall_seconds'],
            'device':device,'observed_scope':'original_scheduler_job_wall_includes_setup_diagnostics_projection_export',
            'historical_run_git_dirty':meta.get('git',{}).get('dirty'),'machine_identity':'UNKNOWN; path alone does not establish same hardware'}

def historical_receipts(req):
    records=[]
    for rid,refs in req['history_refs'].items():
        manifest=pinned_read(refs['run_manifest']);tok=pinned_read(refs['tokenizer']);meta=pinned_read(refs['run']);term=pinned_read(refs['terminal'])
        require(manifest['artifacts']['tokenizer.json']==refs['tokenizer']['sha256'],'historical tokenizer not manifest member')
        vt=pinned_read(refs['verify_terminal']);proof=json.loads(pinned_read(refs['verify_stdout'],text=True))
        require(vt['status']=='succeeded' and vt['exit_code']==0 and proof['manifest_sha256']==refs['run_manifest']['sha256'] and proof['replay_prefix'] is True,'historical registered replay metadata differs')
        caller='src/vkgr/sid_tok_run.py'
        blob=subprocess.check_output(['git','-C',str(ROOT),'show',components.l2_source(ROOT)['origin_commit']+':'+caller])
        require(hashlib.sha256(blob).hexdigest()==manifest['producer'][caller],'historical caller source differs')
        lines=[i for i,line in enumerate(blob.decode().splitlines(),1) if "'eval_every': r['usage_diag_every']" in line]
        require(len(lines)==1,'historical caller interval source ambiguous')
        row=history_entry(rid,manifest,tok,meta,term);row.update(caller_source={'commit':components.l2_source(ROOT)['origin_commit'],'path':caller,'sha256':hashlib.sha256(blob).hexdigest(),'line':lines[0]},input_refs=refs,legacy_verify_limitation='Original legacy stdout lacks explicit status=verified; existing historical acceptance is not reopened or upgraded here.')
        records.append(row)
    return records

def core_result(fitted,source):
    c=components.core_constant(fitted,usage_interval=source['usage_interval'])
    return {'steps':fitted['step'],'fit_seconds':fitted['elapsed'],'core_seconds_per_step':c,'usage_calls':len(fitted['usage_log']),
            'core_interval_uncontaminated':True,'usage_interval':source['usage_interval'],'usage_log':fitted['usage_log']}

def producer():
    p=base.producer();p['files'].update({n:sha(ROOT/n) for n in ('scripts/sid_scale_composed.py','scripts/sid_scale_composed_queue.py','scripts/sid_scale_rq_components.py')})
    return p

def source_identity(engineering):
    source=producer();commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip();dirty=bool(subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain','--untracked-files=all'],text=True))
    if not engineering:
        require(not dirty,'formal composed source must be clean')
        for name,pin in source['files'].items():
            blob=subprocess.check_output(['git','-C',str(ROOT),'show',commit+':'+name]);require(hashlib.sha256(blob).hexdigest()==pin,'formal source differs from committed bytes')
    return {**source,'commit':commit,'dirty':dirty}

def worker(method,n,*,k,steps,engineering):
    require(type(engineering) is bool and method in ('bhkm','core','usage'),'explicit worker scope required')
    base.positive(n,k,steps)
    if engineering:require(n<=1024 and k<=16 and steps<=8,'tiny engineering worker only')
    else:require(sys.platform=='linux' and os.environ.get('VKGR_RUN_ID')==RUN_ID and k==256 and steps==3000 and n in ((250000,300000) if method=='bhkm' else (120415,) if method=='core' else (120415,250000,300000)),'formal original queue worker scope required')
    source=components.l2_source(ROOT)
    if method=='bhkm':
        row=base.worker('bhkm',n,k=k,steps=steps,engineering=engineering);row['producer']=producer();return row
    import resource,numpy as np,torch
    from vkgr import sid_rqvae_r as repair
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    start=time.monotonic();features=base.synthetic(n);generated=time.monotonic()-start
    input_sha=hashlib.sha256(features.astype('<f4',copy=False).tobytes(order='C')).hexdigest()
    if method=='core':
        require(steps<source['usage_interval'],'calibration would trigger periodic usage')
        fitted=repair.fit_r(features,K=k,arm='l2',seed=42,max_steps=steps,batch_size=min(1024,n),device='cpu',wall_cap_seconds=600,project=False,diagnostics=False)
        result=core_result(fitted,source)
    else:result=components.measure_usage_once(features,k=k,d=32,levels=3,engineering=engineering,root=ROOT)
    return {**result,'method':method,'status':'COMPLETE','n':n,'engineering':engineering,'source':source,'diagnostics':False,'projection_stop':False,'device':'cpu',
            'synthetic_features_sha256':input_sha,'generation_seconds':generated,'worker_wall_seconds':time.monotonic()-start,
            'worker_peak_rss_bytes':base.peak_rss_bytes(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,sys.platform),'producer':producer()}

def run_benchmark(req,histories,source,*,clock=time.monotonic,supervisor=base.supervise):
    validate_request(req)
    require(req['engineering'] or sys.platform=='linux' and os.environ.get('VKGR_RUN_ID')==RUN_ID,'formal own Linux queue required')
    start=clock();records=[];failed=False
    stages=[('bhkm',n) for n in req['bhkm_sizes']]+[('core',req['rq_n'])]+[('usage',n) for n in req['usage_sizes']]
    for method,n in stages:
        remaining=req['overall_seconds']-(clock()-start)
        if failed or remaining<=0:
            records.append({'method':method,'n':n,'status':'NOT_RUN_FAILURE' if failed else 'NOT_RUN_BUDGET'});continue
        a=[sys.executable,'-B',str(Path(__file__).resolve()),'--worker',method,'--n',str(n),'--k',str(req['branching']),'--steps',str(req['rq_steps'])]
        if req['engineering']:a.append('--engineering')
        row=supervisor(a,wall_seconds=min(remaining,req['point_seconds'] if method=='bhkm' else 600),rss_mib=req['rss_mib']);record={**row,'method':method,'n':n}
        if row['status']=='COMPLETE':
            payload=row['result'];require(payload['method']==method and payload['n']==n and payload['engineering'] is req['engineering'] and payload['producer']==producer(),'worker source/request differs')
            record.update(payload)
        records.append(record);failed=row['status']=='FAILED'
    status='FAILED' if failed else 'COMPLETE' if all(r['status']=='COMPLETE' for r in records) else 'INCOMPLETE'
    return {'schema':'sid-scale-components-report-v1','status':status,'request':req,'engineering':req['engineering'],'records':records,
            'projection':projection(records,histories,source),'elapsed_seconds':clock()-start,'fixed_l2':source,'producer':producer(),'historical_inputs':histories,
            'synthetic_only':True,'test_read':False,'checkpoint_written':False,'mapping_exported':False,
            'scaling_law':{'fixed_steps':'periodic usage linear in N; core fixed','fixed_visits':'periodic usage quadratic in N; core linear in N; usage fraction increases with N'},
            'frozen_interpretation':'At fixed original usage interval, constant visits scales total periodic usage quadratically with N; item-forward share rises to the same order as core. Absolute wall time and feasibility require measured components and a passing historical selfcheck; counts are not wall time.'}

def bind_job(reqpath,pin,raw,source,preflight):
    rid=PF_ID if preflight else RUN_ID;root=Path('/workspace/vkgr-infra01-root/runs')/rid;meta_path=safe_metadata(root/'run.json');meta=json.loads(meta_path.read_bytes())
    require(os.environ.get('VKGR_RUN_ID')==rid and meta['run_id']==rid and meta['git']['commit']==source['commit'] and meta['git']['dirty'] is False and Path(meta['cwd']).resolve()==ROOT,'original cost job/source differs')
    require(meta['config']['path']==str(reqpath) and meta['config']['sha256']==pin and meta['argv'][1:]==['-B',str(Path(__file__).resolve()),*raw] and Path(meta['argv'][0]).resolve()==Path(sys.executable).resolve(),'original request/argv differs')
    return {'run_id':rid,'run_manifest_sha256':sha(meta_path),'request_sha256':pin,'source_commit':source['commit']}

def main():
    raw=sys.argv[1:];p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--worker',choices=('bhkm','core','usage'));p.add_argument('--n',type=int);p.add_argument('--k',type=int);p.add_argument('--steps',type=int);p.add_argument('--engineering',action='store_true');p.add_argument('--request');p.add_argument('--request-sha');p.add_argument('--output');p.add_argument('--preflight',action='store_true');a=p.parse_args(raw)
    if a.worker:
        require(all(x is None for x in (a.request,a.request_sha,a.output)) and not a.preflight,'worker-only arguments required')
        print(json.dumps(worker(a.worker,a.n,k=a.k,steps=a.steps,engineering=a.engineering),sort_keys=True,allow_nan=False));return
    require(not a.engineering and all(x is None for x in (a.n,a.k,a.steps)) and all(x is not None for x in (a.request,a.request_sha,a.output)),'exact formal parent arguments required')
    path=safe_metadata(a.request);require(sha(path)==a.request_sha,'physical request SHA differs');req=validate_request(json.loads(path.read_bytes()));require(req['engineering'] is False,'formal request required')
    source=source_identity(False);fixed=components.l2_source(ROOT);require(req['rq_steps']<fixed['usage_interval'],'core interval would include usage');histories=historical_receipts(req);binding=bind_job(path,a.request_sha,raw,source,a.preflight)
    if a.preflight:
        print(json.dumps({'status':'preflight_ok','engineering':False,'test_read':False,'checkpoint_read':False,'model_run':False,'source':source,'fixed_l2':fixed,'historical_inputs':histories,'synthetic_calibration_recipe':req,**binding},sort_keys=True,allow_nan=False));return
    out=Path(a.output);require(out==Path('/workspace/vkgr-infra01-root/runs')/RUN_ID/'out' and not out.exists() and out.parent.is_dir() and not any(x.is_symlink() for x in (out,*out.parents)),'exclusive original parent output required')
    report=run_benchmark(req,histories,fixed);report.update(binding=binding,source=source);out.mkdir()
    with (out/'report.json').open('x') as f:json.dump(report,f,sort_keys=True,allow_nan=False)
    print(json.dumps({'status':'benchmark_reported' if report['status']!='FAILED' else 'ALERT','calibration_status':report['status'],'report_sha256':sha(out/'report.json'),'model_alert':report['projection'].get('alert'),**binding},sort_keys=True,allow_nan=False))
    if report['status']=='FAILED':raise SystemExit(3)

if __name__=='__main__':main()
