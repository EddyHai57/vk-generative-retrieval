"""Isolated bootstrap for pinned original evidence; never calls fit/run/verify."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from types import ModuleType

# -I omits both cwd and PYTHONPATH. The only local bootstrap is this sibling,
# already part of the consumer's clean, exact full deployment source ledger.
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('_four_gate_transport',ROOT/'analysis/sgns_subset_acceptance.py')
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)

API={'analysis.sgns_subset_a1':['read_request','identity','job_binding','predecessor','train_output_binding','measured_resources'],
     'vkgr.sgns_subset_a1':['engine_for','registered_config','validation_hits'],
     'analysis.sgns_subset_sid':['load_request','successful_stage'],
     'vkgr.sgns_subset_sid':['load_sgns_stage','load_bundle','collision_gate']}


def file_sha(path):
    h=hashlib.sha256()
    with s.absolute(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def transport_job(run,*,runroot,code,source,argv,config,inputs,resources,data_root=None,cublas=False,openblas=True,extra_env=None):
    root=s.absolute(code);rid=run['run_id']
    runroot=s.absolute(runroot)
    if runroot.name!=rid:raise ValueError('original run directory differs')
    expected_env={'PYTHONPATH':str(root/'src'),'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
        'VKGR_RUN_ID':rid,'VKGR_PROGRESS_PATH':str(runroot/'progress.json')}
    if openblas:expected_env['OPENBLAS_NUM_THREADS']='1'
    if extra_env:expected_env.update(extra_env)
    if data_root is not None:expected_env['VKGR_DATA_ROOT']=data_root
    if cublas:expected_env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    expected_inputs=[{'path':r['path'],'sha256':r['sha256'],'bytes':len(s.record_bytes(r))} for r in inputs]
    expected={'schema':'vkgr-run-v1','run_id':rid,'kind':'ops','cwd':str(root),'argv':argv,'seed':42,
        'git':{'commit':source['commit'],'dirty':False,'dirty_files':[]},'config':config,'inputs':expected_inputs,
        'env':expected_env,'resources':resources,'ops_source':{'commit':'5840e6b686ec9651314a9e28a1a2c0befcc05158',
        'sha256':'7a6e105d77f5105b66c39f28fe8849589bc8624b65c95102a53a3964707d4040'},
        'overlay':None,'pinned_files':{},'stop_mode':'signal'}
    if set(run)-set(expected)-{'created_at','admission'} or any(s.encoded(run.get(k))!=s.encoded(val) for k,val in expected.items()):
        raise ValueError('complete original launcher metadata contract differs')


def check_input_producer(accepted,source,env,code):
    """Complete original derive transport, independent of its older checker."""
    records=accepted['input_producer']
    run,receipt,_=s.producer_records(records)
    argv=run.get('argv',[]);rid='sgns-gpu-subset-a1-inputs-retry1-v1'
    if (len(argv)!=9 or argv[:6]!=[sys.executable,'-B','-m','analysis.sgns_subset_a1_inputs','derive','--request']
            or argv[7]!='--request-sha256' or run.get('run_id')!=rid):
        raise ValueError('original input producer derive argv differs')
    request={'path':argv[6],'sha256':argv[8]};value=json.loads(s.record_bytes(request))
    keys={'schema','source_commit','lock','raw_data','ur001','content','gpu_sid_manifest_sha256','output_root','ops_root'}
    if type(value) is not dict or set(value)!=keys or value.get('schema')!='gpu-sgns-subset-a1-input-request-v1':
        raise ValueError('complete original derive request required')
    jobroot=s.absolute(value['ops_root'])/'runs'/rid
    output=s.absolute(value['output_root']);family=json.loads(s.record_bytes(accepted['prepared']))
    expected_paths={'run':str(jobroot/'run.json'),'terminal':str(jobroot/'terminal.json'),
        'stdout':str(jobroot/'logs/stdout.log'),'receipt':str(output/'stage-receipt.json')}
    if (any(records[k]['path']!=path for k,path in expected_paths.items())
            or value['source_commit']!=source['commit'] or value['lock']!=accepted['lock']
            or value['content']!=accepted['content'] or output!=Path(accepted['prepared']['path']).parent
            or family.get('source')!=source or family.get('runtime')!=env
            or family.get('history')!=accepted['history'] or family.get('content')!=value['content']
            or family.get('ur001')!=value['ur001'] or family.get('raw_data')!=value['raw_data']
            or family.get('gpu_sid_manifest_sha256')!=value['gpu_sid_manifest_sha256']
            or value['gpu_sid_manifest_sha256']!=accepted['gpu_sid']['manifest']['sha256']):
        raise ValueError('original derive request/family/source paths differ')
    expected={'schema':'gpu-sgns-subset-a1-input-stage-v1','authority':'ad1c428','status':'verified','verified':True,
        'run_id':rid,'run_manifest_sha256':records['run']['sha256'],'request_sha256':request['sha256'],
        'source':source,'runtime':env,'output':{**accepted['prepared'],'history':accepted['history'],'subset_four_gate_acceptance':False},
        'science_training':False,'test_read':False,'subset_four_gate_acceptance':False}
    if s.encoded(receipt)!=s.encoded(expected):raise ValueError('original derive receipt semantics differ')
    declared=[request,value['lock'],value['ur001'],value['content']]
    transport_job(run,runroot=jobroot,code=code,source=source,argv=argv,config=request,inputs=declared,
        resources={'cpu_cores':1,'ram_mib':98304,'vram_mib':0,'disk_gib':4,'wall_seconds':7200.0},
        openblas=False,extra_env={'PYTHONDONTWRITEBYTECODE':'1'})
    evidence={rec['path']:rec['sha256'] for rec in [*records.values(),*declared,accepted['prepared'],accepted['history']]}
    for path,digest in evidence.items():s.record_bytes({'path':path,'sha256':digest})
    return {'receipt':receipt,'evidence':evidence}


def bootstrap(code,pins):
    s.source_tree(code,pins['a1_source'])
    # All first-party imports resolve only to the independent, fully pinned A1
    # deployment. No caller-selected import string or executable is accepted.
    analysis=ModuleType('analysis');analysis.__path__=[str(code/'analysis')];sys.modules['analysis']=analysis
    sys.path.insert(0,str(code/'src'))
    vk_spec=importlib.util.spec_from_file_location('vkgr',code/'src/vkgr/__init__.py',submodule_search_locations=[str(code/'src/vkgr')])
    vk=importlib.util.module_from_spec(vk_spec);sys.modules['vkgr']=vk;vk_spec.loader.exec_module(vk)
    from analysis import sgns_subset_a1 as a1
    from analysis import sgns_subset_sid as sid
    from vkgr import sgns_subset_a1 as core
    from vkgr import sgns_subset_sid as science
    from vkgr.sid_depth import SIDTokensLD
    s.module_origins(sys.modules,code,pins['a1_source'])
    for name,functions in API.items():
        module=sys.modules[name]
        if any(not callable(getattr(module,fn,None)) for fn in functions):raise ValueError('fixed original read-only API missing')
    return a1,sid,core,science,SIDTokensLD


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request',type=Path,required=True);p.add_argument('--request-sha256',required=True);p.add_argument('--scratch',type=Path,required=True)
    args=p.parse_args(argv)
    pins,_=s.code_pins()  # No original imports or temporary writes before pins.
    if pins['a1_api']!=API:raise ValueError('fixed original read-only API differs')
    value=s.read_request(args.request,args.request_sha256);code=s.absolute(value['a1_code_root'])
    if Path.cwd()!=code:raise ValueError('original A1 cwd required')
    scratch=s.absolute(args.scratch)
    if scratch.exists():raise FileExistsError('new exclusive evidence scratch required')
    a1,sid,core,science,SIDTokensLD=bootstrap(code,pins)
    evidence={}
    def remember(path,digest=None):
        path=s.absolute(path);actual=file_sha(path)
        if digest is not None and actual!=digest:raise ValueError('original evidence hash differs')
        key=str(path)
        if key in evidence and evidence[key]!=actual:raise ValueError('original evidence changed')
        evidence[key]=actual
        return actual
    def tree(root):
        root=s.absolute(root)
        for parent,dirs,files in os.walk(root,followlinks=False):
            if any((Path(parent)/d).is_symlink() for d in dirs):raise ValueError('original evidence symlink')
            for name in files:remember(Path(parent)/name)
    def records(record):
        for rec in record.values():remember(rec['path'],rec['sha256'])
        run,receipt,_=s.producer_records(record)
        if run.get('git',{}).get('commit')!=receipt.get('source',{}).get('commit'):
            raise ValueError('original job/source commit differs')
        return run,receipt
    records(value['a1_verify'])
    request=value['a1_request'];remember(request['path'],request['sha256'])
    accepted=a1.read_request(Path(request['path']),request['sha256'])
    for key in ('data_root','output_root','scratch_root'):
        original=s.absolute(accepted[key])
        if scratch==original or scratch in original.parents or original in scratch.parents:
            raise ValueError('scratch must be separate from original inputs and outputs')
    if accepted['lock']!=value['lock'] or accepted['source_commit']!=pins['a1_source']['commit']:
        raise ValueError('original A1 request/source/lock differs')
    source,env=a1.identity(accepted)
    if source!=pins['a1_source']:raise ValueError('original A1 complete source ledger differs')
    checked_input=check_input_producer(accepted,source,env,code)
    for path,digest in checked_input['evidence'].items():remember(path,digest)
    for record in a1.stage_inputs(accepted,Path(request['path']),request['sha256']):remember(record['path'],record['sha256'])
    def sgns_input_jobs(fit_record,fit_run,fit_receipt):
        from analysis import sgns_gpu_subset as subset
        run_parent=Path(fit_record['run']['path']).parent.parent
        prepared=fit_receipt['input_producer'];jobs={};previous=None
        bundle=Path(fit_run['argv'][6]);bundle_sha=fit_run['argv'][8]
        remember(bundle/'manifest.json',bundle_sha);tree(bundle)
        for stage,command in (('input-preflight','preflight'),('input-prepare','prepare')):
            jobroot=run_parent/('sgns-gpu-subset-'+stage+'-v1')
            raw={n:(jobroot/path).read_bytes() for n,path in
                (('run','run.json'),('terminal','terminal.json'),('stdout','logs/stdout.log'))}
            for name,path in (('run','run.json'),('terminal','terminal.json'),('stdout','logs/stdout.log')):
                remember(jobroot/path,s.sha(raw[name]))
            rec=s.last_receipt(raw['stdout']);raw['receipt']=s.encoded(rec)
            job,receipt,_=s.producer_bytes(raw,run_path=jobroot/'run.json',physical=True)
            argv=job.get('argv',[])
            if (len(argv)!=13 or argv[:5]!=[sys.executable,'-B','-m','analysis.sgns_gpu_subset',command]
                    or argv[5:13:2]!=['--data','--embedding-root','--lock','--output']
                    or argv[10]!=accepted['lock']['path'] or Path(argv[12])!=bundle
                    or receipt.get('source')!=fit_receipt['source'] or receipt.get('runtime')!=env
                    or receipt.get('records')!=subset.fixed_records() or receipt.get('engineering') is not False
                    or receipt.get('training') is not False or receipt.get('status')!=('verified' if stage=='input-preflight' else 'prepared')
                    or receipt.get('payload_written')!=(stage=='input-prepare')):
                raise ValueError('original SGNS physical input predecessor differs')
            expected_receipt={k:val for k,val in prepared.items() if k!='artifacts'}
            expected_receipt.update(status=receipt['status'],payload_written=receipt['payload_written'])
            if {k:val for k,val in receipt.items() if k!='artifacts'}!=expected_receipt:
                raise ValueError('original SGNS preflight/prepare physical records differ')
            if stage=='input-prepare' and receipt!=prepared:raise ValueError('original SGNS prepare stdout/bundle differs')
            if previous is not None and argv[5:]!=previous[5:]:raise ValueError('original SGNS input argv path drift')
            cfg=Path(job['cwd'])/'configs/a40-subset/hf-files.json'
            declared=[accepted['lock']]
            if stage=='input-prepare':
                prior=run_parent/'sgns-gpu-subset-input-preflight-v1'
                declared.extend({'path':str(prior/n),'sha256':remember(prior/n)} for n in ('run.json','terminal.json','logs/stdout.log'))
            transport_job(job,runroot=jobroot,code=fit_run['cwd'],source=fit_receipt['source'],argv=argv,
                config={'path':str(cfg),'sha256':remember(cfg)},inputs=declared,
                resources={'cpu_cores':1,'ram_mib':24576,'vram_mib':0,'disk_gib':2,'wall_seconds':7200.0},cublas=True)
            previous=argv;jobs[stage]=job
        prior=run_parent/'sgns-gpu-subset-input-prepare-v1'
        declared=[accepted['lock']]+[{'path':str(prior/n),'sha256':remember(prior/n)} for n in ('run.json','terminal.json','logs/stdout.log')]
        declared.append({'path':str(bundle/'manifest.json'),'sha256':bundle_sha})
        cfg=Path(fit_run['cwd'])/'configs/a40-subset/hf-files.json'
        transport_job(fit_run,runroot=Path(fit_record['run']['path']).parent,code=fit_run['cwd'],source=fit_receipt['source'],argv=fit_run['argv'],
            config={'path':str(cfg),'sha256':remember(cfg)},inputs=declared,
            resources={'cpu_cores':1,'ram_mib':24576,'vram_mib':8192 if fit_receipt['device']=='cuda' else 0,
                'disk_gib':2,'wall_seconds':86400.0},cublas=True)
    # Keep all original outputs stable while re-reading manifests and rows.
    tree(accepted['output_root'])
    run,verified=records(value['a1_verify'])
    verify_root=s.absolute(accepted['ops_root'])/'runs'/core.RUN_IDS['verify']
    expected_record={name:str(verify_root/path) for name,path in
        (('run','run.json'),('terminal','terminal.json'),('stdout','logs/stdout.log'))}
    expected_record['receipt']=str(s.absolute(accepted['output_root'])/'verify/stage-receipt.json')
    if any(value['a1_verify'][k]['path']!=v for k,v in expected_record.items()):raise ValueError('original A1 verify record paths differ')
    run_sha=a1.job_binding(accepted,Path(request['path']),request['sha256'],'verify')
    trained=a1.predecessor(accepted,'verify',request['sha256'],source,request=Path(request['path']))
    a1.predecessor(accepted,'train',request['sha256'],source,request=Path(request['path']))
    for stage in ('preflight','train'):
        stage_root=s.absolute(accepted['ops_root'])/'runs'/core.RUN_IDS[stage]
        record={k:{'path':str(stage_root/n),'sha256':remember(stage_root/n)} for k,n in
            (('run','run.json'),('terminal','terminal.json'),('stdout','logs/stdout.log'))}
        art=s.absolute(accepted['output_root'])/stage/'stage-receipt.json'
        record['receipt']={'path':str(art),'sha256':remember(art)};stage_run,_=records(record)
        a1.job_binding(accepted,Path(request['path']),request['sha256'],stage)
        transport_job(stage_run,runroot=stage_root,code=code,source=source,argv=a1.invocation(stage,Path(request['path']),request['sha256']),
            config={'path':request['path'],'sha256':request['sha256']},
            inputs=a1.stage_inputs(accepted,Path(request['path']),request['sha256']),
            resources={'cpu_cores':1,'ram_mib':98304,'vram_mib':32768,'disk_gib':4,'wall_seconds':7200.0 if stage=='preflight' else 172800.0},
            data_root=accepted['data_root'],cublas=True)
    measured=a1.measured_resources(trained)
    transport_job(run,runroot=verify_root,code=code,source=source,argv=a1.invocation('verify',Path(request['path']),request['sha256']),
        config={'path':request['path'],'sha256':request['sha256']},
        inputs=a1.stage_inputs(accepted,Path(request['path']),request['sha256']),
        resources={'cpu_cores':1,'ram_mib':measured['ram_mib'],'vram_mib':measured['vram_mib'],'disk_gib':4,'wall_seconds':172800.0},
        data_root=accepted['data_root'],cublas=True)
    engine,common,extra,producers=core.engine_for(accepted,request['sha256'],source)
    config=core.registered_config(accepted['gpu_sid']['manifest']['sha256'])
    expected={'schema':'gpu-sgns-subset-a1-stage-v1','authority':'ad1c428','engineering':False,
        'stage':'verify','status':'verified','verified':True,'run_id':core.RUN_IDS['verify'],
        'run_manifest_sha256':run_sha,'request_sha256':request['sha256'],'source':source,'runtime':env,
        'config':config,'input_producers':producers,'test_read':False,'subset_four_gate_acceptance':False,
        'hits_gate':378,'training':False,'resource_scope':a1.RESOURCE_SCOPE}
    details={'native_verification','validation_hits','context_inclusive_peak_mib','context_samples','rss_tree_upper_bound_mib'}
    if set(verified)!=set(expected)|details or any(s.encoded(verified.get(k))!=s.encoded(v) for k,v in expected.items()):
        raise ValueError('original A1 verify semantic receipt differs')
    a1.measured_resources(verified)
    native_runtime=engine['_runtime'](False)
    native=a1.train_output_binding(accepted,trained,engine,config,native_runtime)
    # Original CPU/GPU producer sets, SGNS audit health and original SID gates
    # are consumed independently again. No BHKM/SGNS fitting is invoked.
    inputs={};sid_bundles={};sid_requests={}
    for key,device in (('cpu_sid','cpu'),('gpu_sid','cuda')):
        record=accepted[key]['producer'];sid_run,sid_receipt=records(record)
        argv=sid_run.get('argv',[])
        if len(argv)!=9 or argv[1:6]!=['-B','-m','analysis.sgns_subset_sid','verify','--request'] or argv[7]!='--request-sha':
            raise ValueError('original SID verify argv differs')
        req_path=Path(argv[6]);req_sha=argv[8];remember(req_path,req_sha)
        sr=sid.load_request(req_path,req_sha);sid_requests[device]=(sr,str(req_path),req_sha)
        if sr['source']!=pins['sid_source'] or sr['device']!=device or sr['lock']!=accepted['lock']['path']:
            raise ValueError('frozen original SID source/device/lock differs')
        s.source_tree(Path(sid_run['cwd']),pins['sid_source'])
        for stage in ('preflight','fit','verify'):
            rid=sid.run_id(sr,stage);jobroot=Path(sr['ops_root'])/'runs'/rid
            rec={k:{'path':str(jobroot/n),'sha256':remember(jobroot/n)} for k,n in
                (('run','run.json'),('terminal','terminal.json'),('stdout','logs/stdout.log'))}
            art=Path(sr[stage+'_output'])/'stage-receipt.json';rec['receipt']={'path':str(art),'sha256':remember(art)}
            stage_run,stage_receipt=records(rec)
            if stage_run.get('cwd')!=sid_run['cwd'] or stage_run.get('argv')!=sid.worker_argv(stage,req_path,req_sha):
                raise ValueError('original SID stage cwd/argv differs')
            sid.successful_stage(sr,stage,req_path,req_sha)
            ram=24576
            if stage=='verify':
                fit_receipt=json.loads((Path(sr['fit_output'])/'stage-receipt.json').read_bytes())
                rss=fit_receipt.get('rss_tree_upper_bound_mib')
                if type(rss) not in (int,float) or not math.isfinite(rss) or rss<=0:raise ValueError('original SID fit RSS required')
                ram=max(256,math.ceil(rss*1.15/256)*256)
            cfg=Path(sid_run['cwd'])/'configs/a40-subset/hf-files.json'
            transport_job(stage_run,runroot=jobroot,code=sid_run['cwd'],source=sr['source'],argv=sid.worker_argv(stage,req_path,req_sha),
                config={'path':str(cfg),'sha256':remember(cfg)},
                inputs=[{'path':str(req_path),'sha256':req_sha},accepted['lock']],
                resources={'cpu_cores':1,'ram_mib':ram,'vram_mib':0,'disk_gib':2,'wall_seconds':86400.0})
            if stage_receipt.get('runtime')!=env:raise ValueError('original SID runtime differs')
        if str(Path(sr['verify_output'])/'stage-receipt.json')!=record['receipt']['path']:
            raise ValueError('original SID verify output differs')
        tree(sr['fit_output']);tree(sr['verify_output'])
        sgns_record=accepted['cpu_sgns' if device=='cpu' else 'gpu_sgns'];sgns_run,sgns_receipt=records(sgns_record)
        s.source_tree(Path(sgns_run['cwd']),sgns_receipt['source'])
        sgns_input_jobs(sgns_record,sgns_run,sgns_receipt)
        if (sr['sgns_receipt_sha']!=sgns_record['receipt']['sha256']
                or sr['sgns_run']!=str(Path(sgns_record['run']['path']).parent)
                or sr['sgns_output']!=str(Path(sgns_record['receipt']['path']).parent)):
            raise ValueError('SID request/SGNS producer path/hash differs')
        tree(sr['sgns_output'])
        inputs[device]=science.load_sgns_stage(Path(sr['bundle']),sr['bundle_sha'],Path(sr['sgns_output']),
            sr['sgns_receipt_sha'],Path(sr['sgns_run']),device,Path(sr['lock']),sr['source'],env)
        if inputs[device]['sgns_receipt']!=sgns_receipt:raise ValueError('original SGNS receipt differs')
        manifest=accepted[key]['manifest'];remember(manifest['path'],manifest['sha256'])
        bundle=science.load_bundle(Path(manifest['path']),manifest['sha256'],catalog_ids=inputs[device]['ids'],
            vectors=inputs[device]['vectors'],input_manifest=inputs[device]['input_manifest'],engineering=False)
        if (sid_receipt.get('gates')!=bundle.manifest['gates'] or sid_receipt.get('manifest_sha256')!=manifest['sha256']
                or Path(manifest['path'])!=Path(sr['fit_output'])/'run_manifest.json'):
            raise ValueError('SID loaded gate/manifest identity differs')
        sid_bundles[device]=bundle
    gpu_sr=sid_requests['cuda'][0]
    if (gpu_sr['cpu_request']!=sid_requests['cpu'][1] or gpu_sr['cpu_request_sha']!=sid_requests['cpu'][2]):
        raise ValueError('original CUDA SID CPU reference differs')
    collision=science.collision_gate(sid_bundles['cpu'].manifest['audit'],sid_bundles['cuda'].manifest['audit'])
    if collision!=sid_bundles['cuda'].manifest['collision_gate'] or collision!=producers['gpu_sid']['collision_gate']:
        raise ValueError('original integer collision comparison differs')
    # _contexts materializes engineering scratch, never original output data.
    # The original validation readers and EarlyStop selection replay are reused.
    scratch.mkdir(parents=True,exist_ok=False)
    context,engineering,_=engine['_contexts'](*common,config,scratch/'contexts',**extra)
    if engineering is not False:raise ValueError('formal original validation context required')
    tokens=SIDTokensLD(context.codec,50)
    original=core.check_native();stop=original.EarlyStop(config['max_epochs'],config['min_epochs'],config['patience'])
    selected=None;selected_rows=None;selected_metrics=None
    trained_path=Path(accepted['output_root'])/'train/native/manifest.json'
    for row in native['epochs']:
        aggregate,computation,rows=engine['_read_evaluation'](row['evaluation']['kind'],trained_path.parent,row,context,tokens)
        if aggregate!=row['metrics'] or computation!=row['evaluation']['computation_sha256']:
            raise ValueError('original epoch validation rows differ')
        metrics=aggregate['metrics']['all_eligible']
        if stop.observe(row['epoch'],metrics['recall@50'],metrics['ndcg@50']):selected=row;selected_rows=rows;selected_metrics=aggregate
    if (not stop.stopped or stop.state()!=native['selection'] or selected is None
            or native['best']['epoch']!=selected['epoch'] or native['best']['sha256']!=selected['checkpoint_sha256']
            or native['best']['metric']!=list(stop.best_metric)):
        raise ValueError('original selected epoch replay differs')
    hits=core.validation_hits(selected_rows,context.queries,selected_metrics)
    if type(verified['validation_hits']) is not int or hits!=verified['validation_hits']:
        raise ValueError('original integer validation hit numerator differs')
    expected_native={'status':'verified','replayed_epochs':[selected['epoch']],'best_epoch':selected['epoch'],
        'epochs':len(native['epochs']),'run_id':trained_path.parent.parent.name,
        'manifest_sha256':trained['native_manifest_sha256'],'verify_run_id':core.RUN_IDS['verify']}
    if verified['native_verification']!=expected_native:raise ValueError('original native verify binding differs')
    replay_path=Path(accepted['output_root'])/'verify/native'/('replay-best-%03d'%selected['epoch'])
    replay_record={**selected,'evaluation':{**selected['evaluation'],'path':replay_path.name,
        'manifest_sha256':remember(replay_path/'manifest.json')}}
    aggregate,computation,_=engine['_read_evaluation'](selected['evaluation']['kind'],replay_path.parent,replay_record,context,tokens)
    if computation!=selected['evaluation']['computation_sha256'] or aggregate!=selected_metrics:
        raise ValueError('original native verify published replay differs')
    if a1.identity(accepted)!=(source,env):raise ValueError('original A1 source/runtime changed')
    s.source_tree(code,pins['a1_source']);s.module_origins(sys.modules,code,pins['a1_source'])
    for path,digest in evidence.items():
        if file_sha(path)!=digest:raise ValueError('original evidence changed during aggregation')
    proof={'health':inputs['cuda']['sgns_receipt']['original_audit_verification']['health'],
        'sid_gates':{d:sid_bundles[d].manifest['gates'] for d in ('cpu','cuda')},'collision':collision,
        'validation_hits':hits,'hits_gate':378,'a1_native':expected_native,'evidence':evidence}
    s.check_proof(proof)
    print(json.dumps(proof,sort_keys=True,allow_nan=False));return 0


if __name__=='__main__':raise SystemExit(main())
