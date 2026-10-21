"""Fixed original BHKM48 subset SID stages; no G training or holdout reader."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys

from analysis.ur001_derive_cli import encoded,secure,digest,source_identity,guards
from vkgr import sgns_subset_sid as science

REQUEST_KEYS={'schema','device','source','lock','ops_root','ops_code','bundle','bundle_sha',
              'sgns_output','sgns_receipt_sha','sgns_run','preflight_output','fit_output',
              'verify_output','cpu_request','cpu_request_sha'}
PATH_KEYS=('lock','ops_root','ops_code','bundle','sgns_output','sgns_run',
           'preflight_output','fit_output','verify_output')
STAGES=('preflight','fit','verify')
ROOT=Path(__file__).resolve().parents[1]


def load_request(path,sha):
    value=json.loads(science.read_pinned(path,sha))
    if set(value)!=REQUEST_KEYS or value['schema']!='gpu-sgns-subset-sid-request-v1' or value['device'] not in ('cpu','cuda'):
        raise ValueError('fixed SID request fields required')
    for key in PATH_KEYS:secure(value[key])
    for key in ('bundle_sha','sgns_receipt_sha'):science._hex(value[key])
    if value['device']=='cpu':
        if value['cpu_request'] is not None or value['cpu_request_sha'] is not None:raise ValueError('CPU reference must not recurse')
    else:secure(value['cpu_request']);science._hex(value['cpu_request_sha'])
    return value


def run_id(request,stage):return 'sgns-gpu-subset-'+request['device']+'-sid-'+stage+'-v1'


def worker_argv(stage,path,sha):
    if stage not in STAGES:raise ValueError('fixed SID stage required')
    return [sys.executable,'-B','-m','analysis.sgns_subset_sid',stage,'--request',str(secure(path)),'--request-sha',science._hex(sha)]


def own_job(request,stage,path,sha):
    rid=run_id(request,stage);run=secure(Path(request['ops_root'])/'runs'/rid)
    raw=secure(run/'run.json').read_bytes();job=json.loads(raw)
    if (os.environ.get('VKGR_RUN_ID')!=rid or job.get('run_id')!=rid or job.get('kind')!='ops'
            or job.get('git',{}).get('commit')!=request['source']['commit']
            or job.get('argv')!=worker_argv(stage,path,sha)):
        raise ValueError('own formal SID job identity differs')
    return {'run_id':rid,'run_manifest_sha256':hashlib.sha256(raw).hexdigest()}


def successful_stage(request,stage,path,sha):
    rid=run_id(request,stage);run=secure(Path(request['ops_root'])/'runs'/rid)
    snapshots={n:secure(run/n).read_bytes() for n in ('run.json','terminal.json','logs/stdout.log')}
    job,terminal=json.loads(snapshots['run.json']),json.loads(snapshots['terminal.json'])
    stdout=json.loads(next((x for x in reversed(snapshots['logs/stdout.log'].splitlines()) if x.strip()),b'{}'))
    receipt=secure(Path(request[stage+'_output'])/'stage-receipt.json').read_bytes();value=json.loads(receipt)
    if (run.name!=rid or job.get('run_id')!=rid or job.get('kind')!='ops'
            or job.get('git',{}).get('commit')!=request['source']['commit']
            or job.get('argv')!=worker_argv(stage,path,sha) or terminal.get('run_id')!=rid
            or terminal.get('status')!='succeeded' or terminal.get('exit_code')!=0 or stdout!=value
            or value.get('schema')!='gpu-sgns-subset-sid-stage-v1' or value.get('status')!='verified'
            or value.get('verified') is not True or value.get('engineering') is not False
            or value.get('run_id')!=rid or value.get('run_manifest_sha256')!=hashlib.sha256(snapshots['run.json']).hexdigest()
            or value.get('source')!=request['source'] or value.get('device')!=request['device']
            or value.get('stage')!='sid-'+stage or value.get('request_sha256')!=sha
            or value.get('subset_four_gate_acceptance') is not False):
        raise ValueError('original SID predecessor binding differs')
    return value,{'receipt_sha256':hashlib.sha256(receipt).hexdigest(),
                  **{n:hashlib.sha256(b).hexdigest() for n,b in snapshots.items()}}


def input_stage(request,env):
    return science.load_sgns_stage(Path(request['bundle']),request['bundle_sha'],Path(request['sgns_output']),
           request['sgns_receipt_sha'],Path(request['sgns_run']),request['device'],Path(request['lock']),request['source'],env)


def cpu_reference(request,env):
    if request['device']=='cpu':return None
    cp=secure(request['cpu_request']);cs=request['cpu_request_sha'];cpu=load_request(cp,cs)
    if cpu['device']!='cpu' or any(cpu[k]!=request[k] for k in ('source','lock','bundle','bundle_sha','ops_root','ops_code')):
        raise ValueError('same source/runtime/input CPU reference required')
    proof,_=successful_stage(cpu,'verify',cp,cs)
    if proof.get('runtime')!=env:raise ValueError('CPU verification runtime differs')
    inputs=input_stage(cpu,env);path=Path(cpu['fit_output'])/'run_manifest.json'
    loaded=science.load_bundle(path,proof['manifest_sha256'],catalog_ids=inputs['ids'],vectors=inputs['vectors'],input_manifest=inputs['input_manifest'])
    if proof.get('mapping_replay') is not True or proof.get('gates')!=loaded.manifest['gates']:
        raise ValueError('CPU full replay/gate proof differs')
    return loaded.manifest['audit'],{'request_sha256':cs,'verification_receipt_sha256':digest(Path(cpu['verify_output'])/'stage-receipt.json'),
                                    'manifest_sha256':proof['manifest_sha256']}


def execute(stage,path,sha):
    import numpy as np
    from analysis.new_instance_prepare import runtime
    request=load_request(path,sha);source=source_identity()
    if source!=request['source']:raise ValueError('fixed current source differs')
    env=runtime(secure(request['lock']));own=own_job(request,stage,path,sha)
    output=secure(request[stage+'_output'])
    if output.exists() or ROOT in output.parents or output in ROOT.parents:raise ValueError('exclusive external stage output required')
    guards(output,projected_memory=16*1024**3)
    inputs=input_stage(request,env);cpu=cpu_reference(request,env)
    predecessor={}
    if stage!='preflight':
        prior='preflight' if stage=='fit' else 'fit'
        prior_receipt,predecessor=successful_stage(request,prior,path,sha)
        if prior_receipt.get('runtime')!=env or prior_receipt.get('sgns_stage_receipt_sha256')!=request['sgns_receipt_sha']:
            raise ValueError('SID predecessor runtime/input differs')
    common={'schema':'gpu-sgns-subset-sid-stage-v1','status':'verified','verified':True,'authority':'ad1c428',
            'engineering':False,'stage':'sid-'+stage,'device':request['device'],'source':source,'runtime':env,**own,
            'request_sha256':sha,'sgns_stage_receipt_sha256':request['sgns_receipt_sha'],
            'input_manifest_sha256':hashlib.sha256(science.canonical_bytes(inputs['input_manifest'])).hexdigest(),
            'predecessor':predecessor,'subset_four_gate_acceptance':False,'test_read':False,'validation_read':False}
    if stage=='preflight':
        value={**common,'payload_written':False,'mapping_replay':False,'manifest_sha256':None,'gates':None,'collision_gate':None}
    else:
        if stage=='fit':
            model=science.fit_bhkm_coll(inputs['ids'],inputs['vectors'],48,3,42,10)
            audit,gates=science.audit_gates(model.ids,model.sids,inputs['vectors'],inputs['input_manifest'],engineering=False)
            if not gates['passed']:raise ValueError('FIT_FAILED original SID gates')
            comparison=science.collision_gate(cpu[0],audit) if cpu else None
            if comparison and not comparison['passed']:raise ValueError('FIT_FAILED GPU collision gate')
            # Replay before signing; comparison is of original mapping bytes, not metrics.
            replay=science.fit_bhkm_coll(inputs['ids'],inputs['vectors'],48,3,42,10)
            if model.ids.tobytes()!=replay.ids.tobytes() or model.sids.tobytes()!=replay.sids.tobytes():
                raise ValueError('same-seed full mapping replay differs')
            codec=science.sid_v2.build_codec(model.ids,model.sids,48,catalog_ids=inputs['ids'])
            output.mkdir(exist_ok=False)
            buffer=io.BytesIO();science.sid_v2._write_npz(buffer,{'ids':model.ids,'sids':model.sids})
            artifacts={'mapping.npz':buffer.getvalue(),'tokenizer.json':science.canonical_bytes(science.CONFIG),
                       'audit.json':science.canonical_bytes(audit),'input-manifest.json':science.canonical_bytes(inputs['input_manifest'])}
            for name,raw in artifacts.items():science.write_new(output/name,raw)
            manifest={'schema':'gpu-sgns-subset-sid-v1','status':'verified','authority':'ad1c428','engineering':False,
                      'branching':48,'depth':3,'config':science.CONFIG,'source':source,'runtime':env,
                      'input_dimension':64,'content_feature_sha256':inputs['input_manifest']['content_feature_sha256'],
                      'input_manifest_sha256':common['input_manifest_sha256'],'catalog_sha256':science.sid_v2.catalog_digest(inputs['ids']),
                      'input_feature_sha256':science.au.feature_digest(inputs['vectors']),'mapping_replay':True,
                      'subset_four_gate_acceptance':False,'gates':gates,'audit':audit,'collision_gate':comparison,
                      'cpu_reference':None if cpu is None else cpu[1],
                      'suffix_vocabulary':codec.catalog.suffix_vocabulary,'vocab_size':codec.vocab_size,
                      'artifacts':{name:hashlib.sha256(raw).hexdigest() for name,raw in artifacts.items()}}
            manifest_raw=science.canonical_bytes(manifest);science.write_new(output/'run_manifest.json',manifest_raw)
            manifest_sha=hashlib.sha256(manifest_raw).hexdigest()
            loaded=science.load_bundle(output/'run_manifest.json',manifest_sha,catalog_ids=inputs['ids'],vectors=inputs['vectors'],input_manifest=inputs['input_manifest'])
        else:
            manifest_sha=prior_receipt['manifest_sha256'];fit=secure(request['fit_output'])
            loaded=science.load_bundle(fit/'run_manifest.json',manifest_sha,catalog_ids=inputs['ids'],vectors=inputs['vectors'],input_manifest=inputs['input_manifest'])
            replay=science.fit_bhkm_coll(inputs['ids'],inputs['vectors'],48,3,42,10)
            if loaded.codec.catalog.ids.tobytes()!=replay.ids.tobytes() or loaded.codec.catalog.sids.tobytes()!=replay.sids.tobytes():
                raise ValueError('independent same-seed mapping replay differs')
            audit,gates=loaded.manifest['audit'],loaded.manifest['gates']
            comparison=science.collision_gate(cpu[0],audit) if cpu else None
            if comparison!=loaded.manifest['collision_gate'] or comparison and not comparison['passed']:
                raise ValueError('GPU collision proof differs')
            output.mkdir(exist_ok=False)
        value={**common,'payload_written':True,'mapping_replay':True,'manifest_sha256':manifest_sha,
               'gates':loaded.manifest['gates'],'collision_gate':loaded.manifest['collision_gate']}
    # Recheck actual consumed inputs/source/runtime before a successful stage receipt.
    again=input_stage(request,env)
    if again['input_manifest']!=inputs['input_manifest'] or source_identity()!=source or runtime(secure(request['lock']))!=env:
        raise ValueError('source/runtime/input changed during SID stage')
    if stage=='preflight':output.mkdir(exist_ok=False)
    value={**value,'rss_tree_upper_bound_mib':rss_bound_mib(),'rss_scope':'self peak + largest serial child peak; conservative upper bound, not simultaneous tree peak'}
    if stage=='verify':science.write_new(output/'verification.json',encoded(value))
    science.write_new(output/'stage-receipt.json',encoded(value))
    return value


def rss_bound_mib():
    import resource
    unit=1 if sys.platform=='darwin' else 1024
    # This worker uses one serial subprocess at a time (Git identity reads).
    return (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss+resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)*unit/1024**2


def queue_job(command,stage,path,sha):
    import math
    from analysis.a40_ops import load_ops
    request=load_request(path,sha)
    if source_identity()!=request['source']:raise ValueError('fixed queue source differs')
    _,launcher,scheduler,_=load_ops(secure(request['ops_code']))
    ram=24576;scope='not_measured conservative first SID stage allocation'
    if stage!='preflight':
        prior='preflight' if stage=='fit' else 'fit';receipt,_=successful_stage(request,prior,path,sha)
        if stage=='verify':
            measured=receipt.get('rss_tree_upper_bound_mib')
            if type(measured) not in (int,float) or not math.isfinite(measured) or measured<=0:raise ValueError('measured fit RSS required')
            ram=max(256,math.ceil(measured*1.15/256)*256);scope='fit self peak + largest serial child peak; conservative tree upper bound *1.15 ceil256'
    rid=run_id(request,stage)
    flags=['--root',request['ops_root'],'--run-id',rid,'--kind','ops','--cwd',str(ROOT),
           '--config',str(ROOT/'configs/a40-subset/hf-files.json'),'--seed','42',
           '--input',str(secure(path)),'--input',request['lock'],'--env','PYTHONPATH='+str(ROOT/'src'),
           '--env','OMP_NUM_THREADS=1','--env','OPENBLAS_NUM_THREADS=1','--env','MKL_NUM_THREADS=1',
           '--cpu-cores','1','--ram-mib',str(ram),'--vram-mib','0','--disk-gib','2','--wall-hours','24']
    spec=launcher.spec_from_args(launcher.build_parser().parse_args([*flags,'--',*worker_argv(stage,path,sha)]))
    metadata=launcher.prepare(spec)
    result={'status':'prepared_dry','spec':spec,'metadata':metadata,'resource_scope':scope,'subset_four_gate_acceptance':False}
    if command=='queue-dry':return result
    root=launcher.resolve_root(request['ops_root'])
    if (root/'runs'/rid).exists():raise FileExistsError('original SID run exists')
    for state in ('pending','claimed','rejected'):
        if list((root/'queue'/state).glob('*-'+rid+'.json')):raise FileExistsError('SID run already queued')
    path=scheduler.submit(root,spec)
    return {**result,'status':'written','queue_file':path.name}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False);sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('dry')
    for stage in (*STAGES,'queue-dry','queue-submit'):
        q=sub.add_parser(stage,allow_abbrev=False);q.add_argument('--request',type=Path,required=True);q.add_argument('--request-sha',required=True)
        if stage.startswith('queue-'):q.add_argument('--stage',choices=STAGES,required=True)
    a=p.parse_args(argv)
    if a.command=='dry':value={'status':'dry','config':science.CONFIG,'subset_four_gate_acceptance':False,'training':False,'payload_read':False}
    elif a.command.startswith('queue-'):value=queue_job(a.command,a.stage,a.request,a.request_sha)
    else:value=execute(a.command,a.request,a.request_sha)
    print(json.dumps(value,sort_keys=True));return 0


if __name__=='__main__':raise SystemExit(main())
