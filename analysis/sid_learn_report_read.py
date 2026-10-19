"""Queued, aggregate-only structure report using frozen S0 math and archives.
No model/checkpoint/embedding imports, training, new thresholds or cross-radix score.
"""
import argparse
import collections
import datetime
import hashlib
import importlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
SEEDS=(42,2024,2025)
RUNS=tuple(f'g-learn-a1-s{s}' for s in SEEDS)+tuple(f'g-coll-a1-c1-s{s}' for s in SEEDS)
PF_ID='sid-learn-report-preflight-v1'
REPORT_ID='sid-learn-report-v1'
FROZEN={'sid_diag_prime.py':'e2693679630ac4bcae09b474d923a5cb507b692bd12397efa77988204c57813b','sid_diag.py':'e7db6e37cb00f7c165cfd373da53a4538cfa4e54b84113fbbae3bde6d2523161','sid_diag_metrics.py':'ad966940d682aa0f4e576eeacfece8cef92b2f9597fec4a41522b7e535e55dc4','phase2_validation_read.py':'548d8373f1a5008c2a83f29f729930892b1793b01e46e4dbf21942c76adb38d8'}
REPORT_FILES=('uv.lock','analysis/sid_learn_report_read.py','scripts/sid_learn_report_queue.py',*(f'analysis/{n}' for n in FROZEN))
SID_PINS={'learn':'1ecb6619cba1154e0ab48ea00648817745a933bac3d2926d4ad5a17874de272b','control':'c53b81ddfcf8d45fd55903b30f8538372110c8e1df7376156719509994625187'}
SOURCES={'learn':'52b016a623f7d274fa1110a80fa532cb99bdece3','control':'a5da4d6d0746ef41ca1ed93028971360dcb89547'}
ACCEPTED_MANIFESTS=dict(zip(RUNS,('e91198ee5c885fd905b806906d281b529affb5808ee231752e1bc2bc4cb48131','39ec974b5ecf7989cfcb4eff57febb83c68b0f88c00be04636ba1c9c0a5e9ca7','bb1044b3c2f48d04f5ed3a82a543ad37cc4b6ee096f560f64d3d58fb84b550f8','a431a5b1734930671cf7ca092e469a6da16e3c130ffb30836b9aa57aaca8b36b','09a0b790c6b027613ef15bb6c9855d7b1adee79d9bf6ea093c16d09a1a5b8d36','86767e4d30ababc8087898f6f960197021ce54fd09cc6d775da4c373b242c712')))
HITS=dict(zip(RUNS,(382,373,375,388,376,384)))
ACCEPTED_STDOUTS=dict(zip(RUNS,('145ef07cec6b19d92e8ad9852185f2bc272d031586141f27b851b64788c2c513','4d792073e7a5a34a95e134b25f3d8a5b98a4b2f2e65d71f3aa50beb4eea9de21','9c5d0bd80c6602bf3828ccf33d8a695e7a090b7cbe59a271af5d7812479cbb09','9a04887ff44ee514fffe569a51360184e9c4dab314247493e49585634fb66aac','48a22f75d1b915c056edb775e8d9b0fab745d26ffbf16302979870037036ce67','48a22f75d1b915c056edb775e8d9b0fab745d26ffbf16302979870037036ce67')))

def require(ok,message):
    if not ok:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value):return hashlib.sha256((json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()).hexdigest()
def hexpin(value):return type(value) is str and re.fullmatch('[a-f0-9]{64}',value) is not None

def safe(path,roots):
    p=Path(path)
    require(p.is_absolute() and '..' not in p.parts,'absolute nontraversing path required')
    require(not any(x.is_symlink() for x in (p,*p.parents)),'symlink forbidden')
    require(not any(re.search(r'(^|[-_])test([._-]|$)',x.lower()) or any(w in x.lower() for w in ('checkpoint','embedding')) for x in p.parts),'forbidden artifact ancestry')
    require(p.suffix not in ('.pt','.pth','.ckpt') and p.is_file(),'regular nonmodel artifact required')
    require(any(p.is_relative_to(Path(r)) for r in roots),'outside explicit readonly scope')
    return p

def frozen(root):
    root=Path(root)/'analysis'
    for name,pin in FROZEN.items():require(sha(safe(root/name,[root]))==pin,'fixed S0 source bytes differ')
    sys.path.insert(0,str(root))
    for name in ('phase2_validation_read','sid_diag_metrics','sid_diag','sid_diag_prime'):
        old=sys.modules.get(name)
        require(old is None or Path(old.__file__).resolve()==(root/(name+'.py')).resolve(),'foreign S0 module already loaded')
    return importlib.import_module('sid_diag'),importlib.import_module('sid_diag_metrics')

RAW_ROOT='/workspace/vkgr-p2-20261006t1755/data/raw/vk-lsvd-084a183'
TRAIN_AUDIT={'path':'/workspace/vkgr-p4b-20261007t-pipeline-03/prepared-v3/prepare-manifest.json','sha256':'7f3ccd70c02157542e94f41f39e62988fbd650b4be8976455ff20cfe1e4394d2'}

def validate_run_ids(ids):require(type(ids) is list and ids==list(RUNS),'exact six authorized sources required')

def validate_request(req):
    keys={'schema','frozen_root','source_commit','scheduler_root','run_ids','manifest_pins','stdout_pins','history','catalog','sid_manifests','source_delta','train_audit','raw_root'}
    require(type(req) is dict and set(req)==keys and req['schema']=='sid-learn-report-request-v1','exact request schema required')
    validate_run_ids(req['run_ids'])
    require(req['train_audit']==TRAIN_AUDIT and req['raw_root']==RAW_ROOT,'frozen original train input required')
    require(req['manifest_pins']==ACCEPTED_MANIFESTS,'accepted manifests cannot change')
    require(type(req['stdout_pins']) is dict and set(req['stdout_pins'])==set(RUNS) and all(hexpin(x) for x in req['stdout_pins'].values()),'six original stdout pins required')
    require(req['stdout_pins']==ACCEPTED_STDOUTS,'accepted original stdout pins cannot change')
    require(re.fullmatch('[a-f0-9]{40}',req['source_commit']) is not None,'fixed source commit required')
    for name in ('frozen_root','scheduler_root'):require(Path(req[name]).is_absolute() and '..' not in Path(req[name]).parts,'fixed absolute roots required')
    require(set(req['sid_manifests'])==set(SID_PINS) and set(req['source_delta'])==set(SID_PINS),'two original SID families required')
    for ref in (req['history'],req['catalog'],*req['sid_manifests'].values()):require(type(ref) is dict and set(ref)=={'path','sha256'} and type(ref['path']) is str and hexpin(ref['sha256']),'exact artifact path and SHA required')
    require(all(req['sid_manifests'][label]['sha256']==pin for label,pin in SID_PINS.items()),'frozen SID identity differs')
    for delta in req['source_delta'].values():require(type(delta) is dict and all(type(k) is str and hexpin(v) for k,v in delta.items()),'explicit pinned delta required')
    return req

def check_execution_ledger(root,training,delta):
    require(type(training) is dict and bool(training) and all(hexpin(v) for v in training.values()),'original nonempty training ledger required')
    extras=set(REPORT_FILES)-set(training)
    require(set(delta)==extras,'delta paths must exactly equal fixed execution additions')
    expected={**training,**delta};actual={name:sha(safe(Path(root)/name,[root])) for name in set(training)|set(REPORT_FILES)}
    require(actual==expected,'execution ledger differs from training union frozen delta')
    return actual

def original_lock_sha(root,role,training):
    if 'uv.lock' in training:return training['uv.lock']
    # Legacy A1 ledgers omit the lock member. Anchor to the recorded immutable
    # training Git commit, never infer the old lock from current working bytes.
    blob=subprocess.check_output(['git','-C',str(root),'show',SOURCES[role]+':uv.lock'])
    return hashlib.sha256(blob).hexdigest()

def audit_source(root,role,binding,delta,own):
    source=binding['source']
    require(binding['sid']['manifest_sha256']==SID_PINS[role],'training SID and mapping source differ')
    require(source['git']=={'git_base':SOURCES[role],'git_dirty':False},'original training source identity differs')
    ledger=check_execution_ledger(root,source['files'],delta)
    require(binding['runtime']['environment']==own['runtime'] and original_lock_sha(root,role,source['files'])==own['lock_sha256'],'Python/dependencies/lock differ from original training')
    return ledger

def runtime():return {'python':platform.python_version(),'exact_dependencies':{n:version(n) for n in ('numpy','torch','pyarrow','PyYAML')}}

def source_identity(req):
    root=Path(req['frozen_root'])
    require(root.resolve()==ROOT,'own committed report deployment required')
    commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    require(commit==req['source_commit'],'deployed report commit differs')
    require(not subprocess.check_output(['git','-C',str(root),'status','--porcelain','--untracked-files=all'],text=True),'clean source required')
    return {'commit':commit,'files':{p:sha(safe(root/p,[root])) for p in REPORT_FILES},'runtime':runtime(),'lock_sha256':sha(root/'uv.lock')}

def guard_verifier(root,rid,manifest_pin,stdout_pin):
    require(rid in RUNS and hexpin(manifest_pin) and hexpin(stdout_pin),'authorized raw verifier required')
    root=Path(root);vr=root/'runs'/(rid+'-verify')
    for name in (rid,rid+'-verify',rid+'-v1-verify'):
        require(not list((root/'queue/rejected').glob('*-'+name+'.json')),'rejected input source')
    for suffix in ('-verify','-v1-verify'):
        d=root/'runs'/(rid+suffix)
        if d.exists() or d.is_symlink():
            require(not any(p.is_symlink() for p in (d,*d.parents)),'verifier directory symlink')
            for p in (d/'logs/stdout.log',d/'out/verify.json',d/'out/manifest.json'):
                if p.exists() or p.is_symlink():safe(p,[d])
    log=safe(vr/'logs/stdout.log',[vr]);require(log.stat().st_size<=1024*1024 and sha(log)==stdout_pin,'original stdout physical pin differs')
    records=[]
    for line in log.read_text().splitlines():
        try:records.append(json.loads(line))
        except ValueError:pass
    verified=[x for x in records if type(x) is dict and x.get('status')=='verified']
    if rid.startswith('g-learn-a1-'):
        verified=[x for x in verified if x.get('run_id')==rid and x.get('verify_run_id')==rid+'-verify' and x.get('manifest_sha256')==manifest_pin]
    require(bool(verified),'original verified stdout identity absent')
    return {'status':'raw_stdout_verified','path':str(log),'sha256':stdout_pin}

def guard_archive_paths(manifest_path,manifest,reader):
    # The legacy archive reader resolves paths. Guard original lexical ancestors
    # first, so an internal symlink cannot disappear from subsequent checks.
    best=manifest['selection']['best_epoch']
    require(type(best) is int and best>0 and manifest['best']['epoch']==best,'original best selection differs')
    rows=[row for row in manifest['epochs'] if row['epoch']==best]
    require(len(rows)==1,'one original best evaluation required')
    entry=rows[0]['evaluation'];require(entry['path']==f'evaluation-{best:03d}','original evaluation role differs')
    parent=Path(manifest_path).parent;path=safe(parent/entry['path']/'manifest.json',[parent])
    ev,pin=reader(path,entry['manifest_sha256'])
    require(ev['complete'] is True and ev['completed_queries']==2520,'original evaluation incomplete')
    for part in ev['shards']:
        sp=safe(part['path'],[path.parent])
        require(sp.parent==path.parent and hexpin(part['sha256']),'original shard path/pin differs')
    return {'manifest_sha256':pin,'shards':len(ev['shards']),'lexical_paths_checked':True}

def population(b,expected_hits,total=2520,in_catalog=1897,oov=623):
    require((b['n_total'],b['n_in_catalog'],b['n_oov'])==(total,in_catalog,oov),'population differs')
    require(b['exact_item']['top50_hits']==expected_hits,'accepted item hits differ')

def catalog_structure(mapping,branching,metrics):
    a=metrics.structure_metrics(mapping,branching,3)
    leaves=collections.Counter(tuple(code[:3]) for code in mapping.values())
    histogram=collections.Counter(leaves.values())
    return {'S0':a,'leaf_size_histogram':[[size,count] for size,count in sorted(histogram.items())],
            'semantic_collision_items':sum(size for size in leaves.values() if size>1),
            'surplus_collision_items':a['leaves']['collision_items'],'collision_pairs':a['leaves']['collision_pairs']}

def summarize_pair(targets,new_predictions,control_predictions,new_mapping,control_mapping,counts,thresholds,metrics):
    result={}
    for role,predictions,mapping in (('new',new_predictions,new_mapping),('control',control_predictions,control_mapping)):
        result[role]={'B':metrics.prediction_metrics(targets,predictions,mapping),
                      'C':metrics.popularity_metrics(targets,predictions,counts,thresholds['median'],thresholds['p90'],mapping)}
    delta={key:result['new']['C']['bins'][key]['hits']-result['control']['C']['bins'][key]['hits'] for key in result['new']['C']['bins']}
    require(all(result['new']['C']['bins'][key]['n']==result['control']['C']['bins'][key]['n'] for key in delta),'paired bin denominators differ')
    difference=result['new']['B']['exact_item']['top50_hits']-result['control']['B']['exact_item']['top50_hits']
    require(sum(delta.values())==difference and delta['oov']==0,'bin deltas do not sum to item delta')
    return {**result,'delta_hits':difference,'popularity_delta_hits':delta,'prefix_descriptive_only':True}

def cooccurrence_reports(mapping,histories,metrics):
    prime=importlib.import_module('sid_diag_prime')
    return {'D':metrics.cooccurrence_metrics(mapping,histories,seed=0),
            'Dprime':prime.cooccurrence_prime(mapping,histories,seed=0)}

def train_receipts(raw_root,records):
    diag=importlib.import_module('sid_diag')
    out=[]
    for rec in diag.validate_train_records(records):
        p=safe(Path(raw_root)/rec['path'],[raw_root])
        require(p.stat().st_size==rec['size_bytes'],'original train file size differs')
        out.append({'path':str(p),'sha256':rec['sha256'],'bytes':rec['size_bytes']})
    return out

def execute(req,preflight=False):
    validate_request(req);own=source_identity(req);diag,metrics=frozen(req['frozen_root']);archive=diag.archive
    root=Path(req['scheduler_root']);require(root==archive.ROOT,'original archive root required')
    allowed=[root/'runs'/r for rid in RUNS for r in (rid,rid+'-verify',rid+'-v1-verify')]
    allowed+=[root/'queue/claimed',Path(req['history']['path']).parent,Path(req['catalog']['path']).parent,*[Path(ref['path']).parent for ref in req['sid_manifests'].values()]]
    original=archive.read
    def guarded(path,pin=None):safe(path,allowed);return original(path,pin)
    archive.READS.clear();diag.READS.clear();archive.read=guarded
    try:
        hist=diag.read_json(req['history']);catalog=diag.read_json(req['catalog'])
        require(len(catalog)==120415 and len(set(catalog))==120415,'catalog population differs')
        require(req['history']['sha256']==archive.HISTORY_PIN and archive.plain_digest(catalog)==hist['catalog_sha256'] and req['catalog']['sha256']==hist['binding']['catalog_sha'],'history/catalog identity differs')
        hp=Path(req['history']['path']).parent
        queries=diag.read_json({'path':str(hp/'validation-l50.json'),'sha256':hist['artifacts']['validation-l50.json']})
        slices=diag.read_json({'path':str(hp/'train-slices.json'),'sha256':hist['artifacts']['train-slices.json']})
        require(len(queries)==2520 and sum(q['oov'] for q in queries)==623,'query population differs')
        hist['_g_catalog_digest']=archive.digest(catalog);hist['_base_queries_canonical_digest']=hist['binding']['query_sha']
        audit=diag.read_json(req['train_audit']);records=diag.train_records(audit)
        train_files=train_receipts(req['raw_root'],records)
        refs={};states={};source_checks={};ledgers={}
        for role,ref in req['sid_manifests'].items():
            sid=diag.read_json(ref);require(sid['sid_version']=={'learn':'LEARN_LETTER_B256D3_S42','control':'COLL_C1_B256D3_S42'}[role],'SID family differs')
            refs[role]={'path':str(Path(ref['path']).parent/'mapping.npz'),'sha256':sid['artifacts']['mapping.npz']};safe(refs[role]['path'],allowed);diag.stat_ref(refs[role])
        for rid in RUNS:
            role='learn' if rid.startswith('g-learn') else 'control'
            raw=guard_verifier(root,rid,req['manifest_pins'][rid],req['stdout_pins'][rid]);archive.READS[raw['path']]=raw['sha256']
            state=archive.inspect_run(rid)
            require(state['verified'] and state['manifest_sha256']==req['manifest_pins'][rid] and state['terminal']['exit_code']==0,'unverified or changed source manifest')
            require(len(state['queue']['claimed'])==1 and not state['queue']['pending'] and not state['queue']['rejected'],'source must be unique claimed without rejection')
            manifest,_=archive.read(state['manifest_path'],state['manifest_sha256']);binding=manifest['binding'];source=binding['source']
            ledger=audit_source(req['frozen_root'],role,binding,req['source_delta'][role],own)
            archive_paths=guard_archive_paths(state['manifest_path'],manifest,archive.read)
            source_checks[rid]={'manifest_sha256':state['manifest_sha256'],'training_source_commit':SOURCES[role],'execution_ledger_sha256':digest(ledger),'training_lock_sha256':original_lock_sha(req['frozen_root'],role,source['files']),'lock_basis':'manifest_member' if 'uv.lock' in source['files'] else 'original_training_Git_blob','archive_paths':archive_paths,'raw_verified':raw};ledgers[role]=ledger;states[rid]=state
        result={'status':'preflight_ok' if preflight else 'learn_report_complete','read_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source':own,'source_checks':source_checks,'execution_ledgers':ledgers,'population':{'all':2520,'in_catalog':1897,'oov':623},'threshold_source':{'sha256':hist['artifacts']['train-slices.json'],'thresholds':slices['popularity_thresholds']},'engineering':False,'test_read':False,'checkpoint_read':False,'model_run':False,'prefix_cross_branch_comparison':False,'train_files':train_files}
        if not preflight:
            mappings={label:diag.load_mapping(ref,catalog) for label,ref in refs.items()};counts={int(k):v for k,v in slices['item_like_counts'].items()};targets=[q['target'] for q in queries]
            result['learn']={label:catalog_structure(mapping,{'learn':256,'control':256}[label],metrics) for label,mapping in mappings.items()}
            histories=diag.raw_histories({'raw_root':req['raw_root']},records,catalog)
            result['cooccurrence']={label:cooccurrence_reports(mapping,histories,metrics) for label,mapping in mappings.items()}
            require(all(row[part]['n_pairs']==3630957 for row in result['cooccurrence'].values() for part in ('D','Dprime')),'frozen original adjacency count differs')
            reports={}
            for seed in SEEDS:
                nr=f'g-learn-a1-s{seed}';cr=f'g-coll-a1-c1-s{seed}'
                npred,ns=diag.load_predictions(nr,queries,hist,states[nr],catalog);cpred,cs=diag.load_predictions(cr,queries,hist,states[cr],catalog)
                pair=summarize_pair(targets,npred,cpred,mappings['learn'],mappings['control'],counts,slices['popularity_thresholds'],metrics)
                for role,rid,summary in (('new',nr,ns),('control',cr,cs)):
                    population(pair[role]['B'],HITS[rid]);require(summary['hits']==HITS[rid],'archive differs from accepted hits')
                    pair[role]['archive_selfcheck']=summary
                reports[str(seed)]=pair
            result['restrictions']={'training':False,'test_read':False,'checkpoint_read':False,'prefix_descriptive_only':True,'raw_history':'original weeks 00..24 physical order; repeats retained','D_Dprime_not_directly_comparable':'different random algorithms; reported separately; no bootstrap'}
            deltas=[reports[str(seed)]['delta_hits'] for seed in SEEDS]
            result.update(seeds=reports,mechanical_conditions={'three_train_and_raw_verify':True,'mean_delta_hits':sum(deltas)/3,'mean_delta_ge_6':sum(deltas)>=18,'seeds_delta_ge_6':sum(d>=6 for d in deltas),'at_least_two_delta_ge_6':sum(d>=6 for d in deltas)>=2})
        result['input_sha256']={**archive.READS,**diag.READS}
        return result
    finally:archive.read=original

def bind_job(req,path,pin,raw,preflight):
    rid=PF_ID if preflight else REPORT_ID
    require(os.environ.get('VKGR_RUN_ID')==rid,'original queued report identity required')
    root=Path(req['scheduler_root']);run=root/'runs'/rid;meta_path=safe(run/'run.json',[run]);meta=json.loads(meta_path.read_bytes());a=meta.get('argv',[])
    require(meta.get('run_id')==rid and meta.get('git',{}).get('commit')==req['source_commit'] and meta['git'].get('dirty') is False and Path(meta.get('cwd','')).resolve()==ROOT,'run source/cwd differs')
    require(meta.get('config',{}).get('path')==str(path) and meta['config'].get('sha256')==pin,'original request identity differs')
    require(len(a)>=3 and Path(a[0]).resolve()==Path(sys.executable).resolve() and a[1:3]==['-B',str(Path(__file__).resolve())] and a[3:]==raw,'original report argv differs')
    return {'run_id':rid,'run_manifest_sha256':sha(meta_path),'request_sha256':pin,'source_commit':req['source_commit']}

def main():
    raw=sys.argv[1:];p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--request',required=True);p.add_argument('--request-sha',required=True);p.add_argument('--preflight',action='store_true');a=p.parse_args(raw)
    path=Path(a.request);require(path.suffix=='.json' and not path.is_symlink() and path.stat().st_size<=512*1024 and sha(path)==a.request_sha,'bounded pinned request required');req=json.loads(path.read_bytes());validate_request(req)
    binding=bind_job(req,path,a.request_sha,raw,a.preflight);result=execute(req,a.preflight)
    print(json.dumps({**result,**binding},sort_keys=True,allow_nan=False))

if __name__=='__main__':main()
