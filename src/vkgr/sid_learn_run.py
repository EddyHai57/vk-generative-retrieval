"""S2 readonly preflight, exclusive fit, and independent full same-seed verify.
No validation selection, test data, old checkpoints, or target-vector export.
"""
import argparse,hashlib,io,json,math,os,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from . import sid_learn_v1 as contract,sid_learn_input as inputs,sid_learn_bundle as bundle,sid_learn_fit as fit,sid_rqvae as rq,generative_data as gd
from .data_audit import canonical_bytes,file_sha256
ROOT=Path(__file__).resolve().parents[2]
from scripts import coll_sid_a1_ops as profile_ops
MODULE='vkgr.sid_learn_run'
RUN_IDS={'preflight':'sid-learn-input-preflight-v1','cuda-preview':'sid-learn-fit-cuda-v1','fit':'sid-learn-fit-v1','verify':'sid-learn-fit-verify-v1'}

def source_identity(engineering=False):
    identity=bundle.producer_identity();git=gd._git(ROOT)
    if not engineering:
        if git['git_dirty']:raise ValueError('formal source must be committed and clean')
        for name,pin in identity['files'].items():
            blob=subprocess.check_output(['git','-C',str(ROOT),'show',git['git_base']+':'+name])
            if hashlib.sha256(blob).hexdigest()!=pin:raise ValueError('source bytes differ from fixed Git blob')
    return {'commit':git['git_base'],'dirty':git['git_dirty'],'producer':identity['files'],'producer_runtime':identity['runtime']}

def bind_job(command,output,config_sha,input_spec,input_sha,argv,source,engineering):
    result={'run_id':None,'run_manifest_sha256':None,'source_commit':source['commit'],'fit_config_file_sha256':config_sha,'input_spec_file_sha256':input_sha}
    if engineering:return result
    rid=RUN_IDS[command];out=Path(output);record=out.parent/'run.json'
    if os.environ.get('VKGR_RUN_ID')!=rid or out.parent.name!=rid or out.name!='out' or not record.is_file():raise ValueError('formal own job/run identity required')
    meta=gd._json(gd._no_symlink(record));args=meta.get('argv',[])
    if meta.get('run_id')!=rid or meta.get('git',{}).get('commit')!=source['commit'] or meta['git'].get('dirty') is not False or Path(meta.get('cwd','')).resolve()!=ROOT or meta.get('config',{}).get('sha256')!=config_sha or Path(meta['config'].get('path','')).resolve()!=ROOT/bundle.CONFIG_MEMBER:
        raise ValueError('own job source/cwd/config differs')
    if len(args)<5 or not Path(args[0]).is_absolute() or Path(args[0]).resolve()!=Path(sys.executable).resolve() or args[1:4]!=['-B','-m',MODULE] or args[4:]!=list(argv) or not argv or argv[0]!=command:raise ValueError('own actual argv/entry differs')
    if not any(row.get('path')==str(Path(input_spec).resolve()) and row.get('sha256')==input_sha for row in meta.get('inputs',[]) if type(row) is dict):raise ValueError('input spec differs from own run manifest')
    result.update(run_id=rid,run_manifest_sha256=file_sha256(record));return result

def _safe(value):
    if type(value) is float and not math.isfinite(value):return str(value)
    if isinstance(value,dict):return {k:_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_safe(v) for v in value]
    return value

def _new(path,value):
    with Path(path).open('xb') as stream:stream.write(canonical_bytes(_safe(value)))

def _config(path,pin):
    expected=ROOT/bundle.CONFIG_MEMBER
    if not Path(path).is_absolute() or Path(path).resolve()!=expected or pin!=file_sha256(expected):raise ValueError('fixed committed S2 config member required')
    return contract.validate_config(gd._json(gd._checked_file(path,pin)))

def cuda_preview(data,config):
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for actual measurement')
    contract.validate_config(config);torch.set_num_threads(1);torch.cuda.init();torch.cuda.empty_cache();torch.cuda.synchronize()
    memory={'pid':os.getpid(),'context_process_mib':profile_ops.own_gpu_mib(),'context_reserved_mib':torch.cuda.memory_reserved()/2**20,'samples':[]}
    torch.cuda.reset_peak_memory_stats()
    def sample(stage):
        torch.cuda.synchronize()
        memory['samples'].append({'stage':stage,'own_process_mib':profile_ops.own_gpu_mib(),'reserved_mib':torch.cuda.memory_reserved()/2**20,'max_reserved_mib':torch.cuda.max_memory_reserved()/2**20,'max_allocated_mib':torch.cuda.max_memory_allocated()/2**20})
    # Real formal input was checked above. The two updates use synthetic values
    # of the full real shape, not scientific fit observations or selection.
    rng=np.random.default_rng(42);content=rng.normal(size=data['content'].shape).astype('float32');content/=np.linalg.norm(content,axis=1,keepdims=True)
    target=rng.normal(size=data['target'].shape).astype('float32');target/=np.linalg.norm(target,axis=1,keepdims=True)
    calls=[0]
    def boundary():
        sample('fit-boundary-'+str(calls[0]));calls[0]+=1
    result=fit.fit_learned(content,target,data['fallback'],config=config,device='cuda',engineering=True,options={'max_steps':2,'check_every':1},guard=boundary)
    if result['status']!='complete':raise ValueError('engineering updates failed: '+result['status'])
    sample('after-full-shape-diagnostic')
    state=result['state'];before=rq.export_codes(state,content[:8]);buffer=io.BytesIO();bundle.sid_v2._write_npz(buffer,bundle.sid_v2._weights_arrays(state,256));buffer.seek(0)
    restored=bundle.sid_v2._check_state_arrays(bundle.sid_v2._load_npz(buffer),256)
    after=rq.export_codes(restored,content[:8])
    if rq.state_digest(restored['state_dict'])!=rq.state_digest(state['state_dict']) or not np.array_equal(before,after):raise ValueError('own new in-memory weights/export reload differs')
    sample('memory-reload-export')
    measured=profile_ops.measured_vram(memory);props=torch.cuda.get_device_properties(0)
    return {'status':'cuda_engineering_verified','engineering_only':True,'data_engineering':data['engineering'],'synthetic_grad_steps':2,'full_real_shape':{'items':len(content),'content_dim':64,'target_dim':32,'K':256,'batch_size':1024},
            'memory_measurement':memory,'measured_mib':measured,'declared_vram_mib':profile_ops.declared_vram(measured),'measurement_basis':'context_inclusive_own_pid','reload_decode_equal':True,
            'memory_checkpoint_replay':{'own_new_only':True,'storage':'memory','status':'passed'},'server_full_runner_train_verify':False,
            'runtime':{'environment':gd._environment(),'device':'cuda:0','cuda_version':torch.version.cuda,'gpu_name':props.name,'gpu_uuid':str(getattr(props,'uuid','UNKNOWN')),'threads':torch.get_num_threads(),'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),'allow_tf32':torch.backends.cuda.matmul.allow_tf32},'test_read':False}

def verify_replay(loaded,bundle_path,data,config,*,device,engineering,guard=None):
    original=gd._json(Path(bundle_path).parent/'train-log.json')
    options={'K':original['K'],'batch_size':original['batch_size'],'max_steps':original['max_steps']} if engineering else None
    replay=fit.fit_learned(data['content'],data['target'],data['fallback'],config=config,device=device,engineering=engineering,options=options,guard=guard)
    if replay['status']!='complete' or replay.get('final_state_digest')!=original['final_state_digest'] or replay['init_digest']!=original['init_digest']:raise ValueError('full same-seed state replay failed')
    sids=rq.assign_fourth_code(data['ids'],rq.export_codes(replay['state'],data['content']))
    if not np.array_equal(sids,loaded.codec.catalog.sids):raise ValueError('full same-seed mapping replay failed')
    return {'state_replay':True,'mapping_replay':True,'pca_replay':True,'fallback_recomputed':True,'replay_steps':replay['step'],'replay_elapsed_seconds':replay['elapsed']}

def _producer_binding(args,output,loaded,config_sha,engineering):
    path=Path(args.producer_run)
    if not engineering:
        expected=Path(output).parent.parent/'sid-learn-fit-v1'/'run.json'
        expected_bundle=expected.parent/'out/payload/run_manifest.json'
        if path!=expected or Path(args.bundle)!=expected_bundle:raise ValueError('verify only owns its one original fit')
    producer=gd._json(gd._checked_file(path,args.producer_run_sha))
    if producer.get('run_id')!='sid-learn-fit-v1' or producer.get('git',{}).get('commit')!=loaded.manifest['source_commit'] or producer.get('config',{}).get('sha256')!=config_sha or not engineering and producer['git'].get('dirty') is not False:raise ValueError('original fit job/source/config differs')
    proof={'producer_run_manifest_sha256':args.producer_run_sha,'producer_run_id':'sid-learn-fit-v1','producer_commit':loaded.manifest['source_commit']}
    if not engineering:
        terminal_path=path.parent/'terminal.json';terminal=gd._json(gd._no_symlink(terminal_path))
        if terminal.get('run_id')!='sid-learn-fit-v1' or terminal.get('status')!='succeeded' or type(terminal.get('exit_code')) is not int or terminal['exit_code']!=0 or terminal.get('wall_cap_exceeded') is not False:raise ValueError('original fit must have succeeded within cap')
        queue=path.parent.parent.parent/'queue';states={key:list((queue/key).glob('*-sid-learn-fit-v1.json')) for key in ('pending','claimed','rejected')}
        if len(states['claimed'])!=1 or states['pending'] or states['rejected']:raise ValueError('original fit requires unique claimed without rejection')
        claimed=gd._json(gd._no_symlink(states['claimed'][0]))
        if claimed.get('run_id')!='sid-learn-fit-v1' or claimed.get('argv')!=producer.get('argv') or claimed.get('cwd')!=producer.get('cwd'):raise ValueError('original fit queue/job binding differs')
        log=gd._no_symlink(path.parent/'logs/stdout.log')
        if log.stat().st_size>65536:raise ValueError('bounded original fit stdout required')
        expected={'status':'complete','engineering':False,'run_id':'sid-learn-fit-v1','run_manifest_sha256':args.producer_run_sha,'manifest_sha256':args.sha,'fit_config_file_sha256':config_sha,'input_spec_file_sha256':args.inputs_sha,'source_commit':loaded.manifest['source_commit']}
        records=[]
        for line in log.read_text().splitlines():
            try:records.append(json.loads(line))
            except ValueError:pass
        if not any(type(row) is dict and all(type(row.get(k)) is type(v) and row[k]==v for k,v in expected.items()) for row in records):raise ValueError('original fit raw complete/job/manifest absent')
        proof.update(producer_terminal_sha256=file_sha256(terminal_path),producer_stdout_sha256=file_sha256(log),producer_claimed_sha256=file_sha256(states['claimed'][0]))
    return proof

def main(argv=None):
    raw=list(sys.argv[1:] if argv is None else argv)
    parser=argparse.ArgumentParser(allow_abbrev=False);sub=parser.add_subparsers(dest='command',required=True)
    for name in ('preflight','cuda-preview','fit','verify'):
        q=sub.add_parser(name,allow_abbrev=False)
        for role in ('config','inputs'):
            q.add_argument('--'+role,required=True);q.add_argument('--'+role+'-sha',required=True)
        q.add_argument('--output',required=True);q.add_argument('--engineering',action='store_true');q.add_argument('--device',choices=('cpu','cuda'))
        if name in ('fit','verify'):
            q.add_argument('--engineering-steps',type=int);q.add_argument('--engineering-batch',type=int);q.add_argument('--engineering-k',type=int)
        if name=='verify':
            q.add_argument('--bundle',required=True);q.add_argument('--sha',required=True);q.add_argument('--producer-run',required=True);q.add_argument('--producer-run-sha',required=True)
    args=parser.parse_args(raw)
    try:
        options={k:v for k,v in (('max_steps',getattr(args,'engineering_steps',None)),('batch_size',getattr(args,'engineering_batch',None)),('K',getattr(args,'engineering_k',None))) if v is not None}
        if options and not args.engineering:raise ValueError('formal schedule overrides prohibited')
        if args.command=='cuda-preview' and args.device not in (None,'cuda'):raise ValueError('CUDA measurement cannot use CPU')
        import pyarrow as pa
        pa.set_cpu_count(1);pa.set_io_thread_count(1);torch.set_num_threads(1)
        out=Path(args.output)
        if not out.is_absolute() or '..' in out.parts or not out.parent.is_dir() or out.exists():raise ValueError('absolute exclusive output in existing run required')
        gd._no_symlink(out)
        config=_config(args.config,args.config_sha);source=source_identity(args.engineering)
        binding=bind_job(args.command,out,args.config_sha,args.inputs,args.inputs_sha,raw,source,args.engineering)
        if not Path(args.inputs).is_absolute():raise ValueError('absolute input spec required')
        spec=gd._json(gd._checked_file(args.inputs,args.inputs_sha))
        if type(spec) is not dict or set(spec)!={'schema','refs'} or spec['schema']!='sid-learn-input-spec-v1':raise ValueError('fixed input spec schema required')
        data=inputs.load_inputs(spec['refs'],engineering=args.engineering);summary=inputs.summary(data)
        common={**summary,**binding,'producer':source['producer'],'producer_runtime':source['producer_runtime'],'vectors_file_sha256':data['physical']['c1_vectors'],'input_manifest_file_sha256':data['physical']['c1_manifest'],'catalog_file_sha256':data['physical']['catalog'],'content_file_sha256':data['physical']['content']}
        if args.command=='preflight':print(json.dumps(common,sort_keys=True));return 0
        if args.command=='cuda-preview':
            print(json.dumps({**common,**cuda_preview(data,config)},sort_keys=True));return 0
        device=args.device or ('cpu' if args.engineering else 'cuda')
        if not args.engineering and device!='cuda':raise ValueError('formal S2 backend is CUDA')
        from .generative_b1_run import _guard
        guard=_guard(args.engineering,out.parent,device)
        if args.command=='fit':
            out.mkdir();_new(out/'started.json',{**binding,'started_epoch':time.time(),'wall_cap_seconds':config['wall_cap_seconds']})
            result=fit.fit_learned(data['content'],data['target'],data['fallback'],config=config,device=device,engineering=args.engineering,options=options or None,guard=guard)
            report={k:v for k,v in result.items() if k!='state'};_new(out/'fit-result.json',report)
            if result['status']!='complete':
                _new(out/'failure.json',{'status':'ALERT','reason':result['status'],'fit':report,**binding});print(json.dumps({'status':'ALERT','reason':result['status'],**binding},sort_keys=True));return 3
            try:
                manifest=bundle.write_bundle(out/'payload',ids=data['ids'],content=data['content'],c1_vectors=data['c1_vectors'],c1_manifest=data['c1_manifest'],physical=data['physical'],pca_state=data['pca_state'],fallback=data['fallback'],fit=result,config=config,engineering=args.engineering)
            except bundle.FitGateFailure as exc:
                _new(out/'failure.json',{'status':'ALERT','reason':'FIT_FAILED','gates':exc.gates,'audit':exc.audit,**binding})
                sid_arrays=bundle.sid_v2._weights_arrays(result['state'],result['K']);bundle.sid_v2._write_npz(out/'failed-weights.npz',sid_arrays)
                print(json.dumps({'status':'ALERT','reason':'FIT_FAILED',**binding},sort_keys=True));return 3
            print(json.dumps({'status':'engineering_complete' if args.engineering else 'complete','engineering':args.engineering,'manifest_sha256':file_sha256(out/'payload/run_manifest.json'),'pca_components_sha256':manifest['pca']['components_sha256'],'fallback_mask_sha256':manifest['fallback_mask_sha256'],**binding},sort_keys=True));return 0
        loaded=bundle.load_bundle(args.bundle,args.sha,catalog_ids=data['ids'],content=data['content'],c1_vectors=data['c1_vectors'],c1_manifest=data['c1_manifest'],physical=data['physical'],fallback=data['fallback'],engineering=args.engineering)
        producer=_producer_binding(args,out,loaded,args.config_sha,args.engineering)
        replay=verify_replay(loaded,args.bundle,data,config,device=device,engineering=args.engineering,guard=guard)
        proof={**common,**producer,**replay,'schema':'learn-fit-verify-v1','status':'engineering_verified' if args.engineering else 'verified','written':True,'manifest_sha256':args.sha,'sid_version':loaded.info['sid_version'],'verifier_commit':source['commit']}
        out.mkdir();_new(out/'verification.json',proof);print(json.dumps(proof,sort_keys=True));return 0
    except (ValueError,KeyError,TypeError,OSError,RuntimeError) as exc:
        print('SID-LEARN gate failed: '+str(exc),file=sys.stderr);return 1

if __name__=='__main__':raise SystemExit(main())
