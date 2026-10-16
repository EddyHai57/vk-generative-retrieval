"""Assemble a finite S1 queue plan from pinned metadata; no target data/checkpoint I/O.

CLI --deployment JSON --metadata JSON --s0-plan JSON --output NEWDIR.
Output documents.json is basename->JSON for control.seal, whose writer uses
json.dump(sort_keys=True,allow_nan=False) with default separators and no newline.
Each stage gets only its own refs. Exact @stage.field tokens are resolved by ops
from previously verified receipts, never shell-interpolated or embedded in text.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
OPS='/workspace/vkgr-sid-coll-ops-v1'
DEP='/workspace/vkgr-sid-coll-g-v1'
DATA='/workspace/vkgr-b1-g'
R='/workspace/vkgr-infra01-root'
PY='/workspace/vkgr-p6-20261006t2050/code-v1/.venv/bin/python'
TEACHER='/workspace/vkgr-seqm1-p-v1'
STAGES=('input_preflight','input','input_verify','sid_fit','sid_verify','preflight','cuda','dry',
        'train42','verify42','train2024','verify2024','train2025','verify2025')
LIMIT=48*1024


def wire_bytes(value):return json.dumps(value,sort_keys=True,allow_nan=False).encode()
def wire_sha(value):return hashlib.sha256(wire_bytes(value)).hexdigest()
def file_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _hex(value,n=64):
    if type(value) is not str or not re.fullmatch('[0-9a-f]{'+str(n)+'}',value):raise ValueError('exact SHA required')
    return value

def ref(value):
    if type(value) is not dict or set(value)!={'path','sha256'}:raise ValueError('exact metadata path/SHA reference required')
    p=Path(value['path'])
    if not p.is_absolute() or '..' in p.parts or p.suffix.lower() in ('.pt','.pth','.ckpt') or re.search(r'week[_-]?26|/test/',str(p).lower()):
        raise ValueError('safe absolute metadata/artifact ref required')
    _hex(value['sha256']);return copy.deepcopy(value)


def build_documents(deployment,metadata,s0_plan,*,root=ROOT):
    """Produce all JSON documents in memory; never opens a server ref or teacher CP."""
    commit=_hex(deployment.get('commit'),40);sources=deployment.get('files')
    if type(sources) is not dict or 'scripts/sid_coll_job_entry.py' not in sources:raise ValueError('deployed source ledger required')
    for name,pin in sources.items():
        p=Path(name)
        if p.is_absolute() or '..' in p.parts:raise ValueError('relative source member required')
        _hex(pin)
    created=metadata.get('created_epoch')
    if type(created) not in (int,float) or not math.isfinite(created):raise ValueError('actual seal created_epoch required')
    common=metadata.get('common',{})
    def fixed(name,fallback=None):
        value=common.get(name,fallback)
        if value is None:raise ValueError('missing actual common reference '+name)
        return ref(value)
    catalog=fixed('catalog',s0_plan.get('catalog'));content=fixed('content')
    history=fixed('history',s0_plan.get('history'));prepared=fixed('g_prepared')
    slices=fixed('slices');s0_result=fixed('s0_result')
    support=ref(common.get('support',metadata.get('support_ref')))
    if support['path']!=OPS+'/supported.npy':raise ValueError('fixed derived supported output required')
    deployment_ref=ref(metadata.get('deployment_ref'))
    if deployment_ref['path']!=OPS+'/deployment.json':raise ValueError('fixed actual deployment ref required')
    teacher_manifest=ref(metadata.get('manifest'));lock=metadata.get('lock')
    if type(lock) is not dict or lock.get('schema')!='coll-c2-source-lock-v1' or lock.get('train_sequence_sha256')!='PENDING_TRAIN_SEQUENCE':
        raise ValueError('frozen pending-sequence teacher lock template required')
    teacher_lock={'path':OPS+'/teacher-lock.json','sha256':wire_sha(lock)}
    if 'lock_template_ref' in metadata and ref(metadata['lock_template_ref'])!=teacher_lock:raise ValueError('teacher lock physical SHA differs from seal writer')
    if type(s0_plan) is not dict or 'train_audit' not in s0_plan or not Path(s0_plan.get('raw_root','')).is_absolute():raise ValueError('frozen raw train source plan required')
    ref(s0_plan['train_audit'])
    docs={'s0-plan.json':copy.deepcopy(s0_plan)}
    s0_ref={'path':OPS+'/s0-plan.json','sha256':wire_sha(s0_plan)}
    plan={'schema':'vkgr-sid-coll-plan-v1','root':R,'control_dir':OPS+'/controller','created_epoch':created,
          'deadline_epoch':created+48*3600,'common_refs':[deployment_ref,teacher_lock],'branches':{}}
    for variant,label in (('C1','C1p'),('C2','C2')):
        stem='sid-coll-'+variant.lower()+'-';specs={};spec_shas={};proof_origins={};stages={}
        def rid(name):return stem+name+'-v1'
        def stage_path(name):return OPS+'/'+rid(name)+'.json'
        def out(name):return R+'/runs/'+rid(name)+'/out'
        def cfg(name):
            rel='configs/sid_coll/'+name;path=Path(root)/rel
            pin=file_sha(path)
            if sources.get(rel)!=pin:raise ValueError('config bytes differ from deployment '+rel)
            return {'path':DEP+'/'+rel,'sha256':pin}
        fit_config=cfg('fit-coll-'+variant.lower()+'.json')
        configs={seed:cfg(f'g-coll-{variant.lower()}-s{seed}.json') for seed in (42,2024,2025)}
        def dynamic(name,proof,path):
            if name not in spec_shas:raise ValueError('future stage ref forbidden')
            generic='train' if name.startswith('train') else 'verify' if name.startswith('verify') else name
            return {'path':path,'sha256':'DYNAMIC','from':{'receipt_path':out(name)+'/receipt.json',
                 'stage':generic,'run_id':rid(name),'spec_sha256':spec_shas[name],'proof':proof}}
        def token(reference):
            origin=reference['from'];name=next(n for n in spec_shas if rid(n)==origin['run_id'])
            return '@'+name+('.receipt_sha256' if origin['proof']=='__receipt__' else '.proofs.'+origin['proof']+'.sha256')
        for name in STAGES:
            generic='train' if name.startswith('train') else 'verify' if name.startswith('verify') else name
            params={'root':DATA,'data_root':DATA};refs={}
            if name in ('input_preflight','input','input_verify'):
                refs={'catalog':catalog,'content':content,'s0_plan':s0_ref}
                if variant=='C2':refs.update(teacher_manifest=teacher_manifest,teacher_lock=teacher_lock,support=support);params['teacher_root']=TEACHER
                if name=='input':params['device']='cpu'
                if name=='input_verify':
                    refs['producer_manifest']=dynamic('input','producer_manifest',out('input')+'/payload/producer-manifest.json')
                    if variant=='C2':
                        refs['source_lock']=dynamic('input','source_lock',out('input')+'/source-lock.json')
                        refs['input_sources']=dynamic('input','input_sources',out('input')+'/input-sources.json')
            elif name in ('sid_fit','sid_verify'):
                refs={'fit_config':fit_config,'catalog':catalog,
                      'vectors':dynamic('input','vectors',out('input')+'/payload/vectors.npy'),
                      'input_manifest':dynamic('input_verify','input_manifest',out('input_verify')+'/input-manifest.json')}
                if name=='sid_verify':
                    refs.update(content=content,s0_plan=s0_ref)
                    refs['sid_manifest']=dynamic('sid_fit','sid_manifest',out('sid_fit')+'/payload/run_manifest.json')
            elif name in ('preflight','cuda') or generic in ('train','verify'):
                refs={'prepared':prepared,'history':history,'content':content,
                      'vectors':dynamic('input','vectors',out('input')+'/payload/vectors.npy'),
                      'input_manifest':dynamic('input_verify','input_manifest',out('input_verify')+'/input-manifest.json'),
                      'sid_manifest':dynamic('sid_fit','sid_manifest',out('sid_fit')+'/payload/run_manifest.json')}
                if generic in ('train','verify'):
                    seed=int(name.removeprefix(generic));params['seed']=seed;refs['config'+str(seed)]=configs[seed]
                    if generic=='verify':
                        refs.update(catalog=catalog,slices=slices,s0_result=s0_result)
                        previous='train'+str(seed)
                        refs['train_receipt']=dynamic(previous,'__receipt__',out(previous)+'/receipt.json')
                        refs['train_manifest']=dynamic(previous,'manifest',out(previous)+'/payload/manifest.json')
                else:refs.update({'config'+str(seed):c for seed,c in configs.items()})
            elif name=='dry':
                refs={'cuda':dynamic('cuda','__receipt__',out('cuda')+'/receipt.json'),**{'config'+str(seed):c for seed,c in configs.items()}}
            else:raise ValueError('unregistered stage')
            spec={'schema':'coll-stage-spec-v1','variant':variant,'engineering':False,'commit':commit,
                  'source_files':copy.deepcopy(sources),'refs':copy.deepcopy(refs),'stages':{generic:{'run_id':rid(name),'parameters':params}}}
            pin=wire_sha(spec);docs[rid(name)+'.json']=spec;specs[name]=spec;spec_shas[name]=pin
            argv=[PY,'-B',DEP+'/scripts/sid_coll_job_entry.py','--stage',generic,'--spec',stage_path(name),'--spec-sha',pin,'--output',out(name)]
            for role,reference in refs.items():
                if reference['sha256']=='DYNAMIC':argv+=['--ref',role,token(reference)]
            gpu=2048 if name=='preflight' else 39312 if name=='cuda' else '@cuda.raw_proof.declared_vram_mib' if generic in ('train','verify') else 0
            env=['PYTHONPATH='+DEP+'/src:'+DEP,'OMP_NUM_THREADS=1','MKL_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1',
                 'VKGR_DATA_ROOT='+DATA,'CUDA_VISIBLE_DEVICES='+('0' if gpu!=0 else ''),
                 'CUBLAS_WORKSPACE_CONFIG=:4096:8','PYTHONDONTWRITEBYTECODE=1']
            job={'schema':'vkgr-queue-job-v1','run_id':rid(name),'kind':'ops','cwd':DEP,'config':stage_path(name),
                 'seed':params.get('seed',42),'inputs':[stage_path(name),OPS+'/deployment.json'],
                 'pinned_manifest':None,'pinned_files':[],'env':env,'cpu_cores':1,'ram_mib':8192,'vram_mib':gpu,
                 'disk_gib':4,'wall_hours':24.,'stop_mode':'signal','no_admission':False,'argv':argv}
            status='produced' if name=='input' else 'complete' if name=='sid_fit' or generic=='train' else 'verified'
            expected={'schema':'coll-stage-receipt-v1','status':status,'stage':generic,'variant':variant,'run_id':rid(name),
                      'binding.commit':commit,'binding.spec_sha256':pin,
                      'binding.config_sha256':configs[params['seed']]['sha256'] if generic in ('train','verify') else refs.get('fit_config',{}).get('sha256'),
                      'binding.data':{role:token(reference) if reference['sha256']=='DYNAMIC' else reference['sha256'] for role,reference in refs.items()}}
            stages[name]={'job':job,'proof':{'path':out(name)+'/receipt.json','expected':expected}}
        plan['branches'][label]={'stages':stages}
    docs['pipeline-plan.json']=plan
    for basename,value in docs.items():
        bound=2*1024*1024 if basename=='pipeline-plan.json' else LIMIT
        if len(wire_bytes(value))>=bound:raise ValueError('sealed document exceeds metadata bound: '+basename+' '+str(len(wire_bytes(value))))
    return docs


def main(argv=None):
    p=argparse.ArgumentParser(allow_abbrev=False)
    for name in ('deployment','metadata','s0-plan','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args(argv)
    try:
        values=[json.loads(Path(x).read_bytes()) for x in (a.deployment,a.metadata,a.s0_plan)]
        docs=build_documents(*values)
        out=Path(a.output).absolute()
        if out.exists() or not out.parent.is_dir() or any(x.is_symlink() for x in (out,*out.parents)):raise ValueError('exclusive new nonsymlink output directory required')
        out.mkdir()
        (out/'documents.json').write_bytes(wire_bytes(docs))
        for name,value in docs.items():(out/name).write_bytes(wire_bytes(value))
        (out/'documents-sha256.json').write_bytes(wire_bytes({name:wire_sha(value) for name,value in docs.items()}))
        print(json.dumps({'status':'assembled','documents':len(docs),'plan_sha256':wire_sha(docs['pipeline-plan.json'])}));return 0
    except (ValueError,KeyError,TypeError,OSError) as exc:
        print('S1 plan STOP: '+str(exc),file=sys.stderr);return 1

if __name__=='__main__':raise SystemExit(main())
