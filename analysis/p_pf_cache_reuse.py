"""Semantic sealing and read-only reuse of one successful P PF compact cache.

No train/submit/signal command. Existing backend, compact module and driver bytes
are unchanged. A seal proves current cache semantics against fixed input bytes;
it cannot attest cache bytes at the earlier PF completion timestamp.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
from analysis import p_full_compact_inputs as compact
from analysis import p_full_training_backend as backend

ROOT=Path(__file__).resolve().parents[1]
BASE_COMMIT='18f98f56fc7211d549eaf296015f256d81691c5b'
BASE_FILES={'analysis/p_full_training_backend.py':'d9c9f038e9d87e07a41c42e818db4483e0b4415187c874df09ca810a9ba836a1',
            'analysis/p_full_compact_inputs.py':'4953efea67bb59bc5b9903530e0736c90fe5cc1b202a70ea7628b555d432216a'}
ADDITIONS=frozenset({'analysis/p_pf_cache_reuse.py','tests/test_p_pf_cache_reuse.py',
                    'docs/superpowers/reviews/2026-10-22-p-pf-cache-reuse-implementation.md'})
SCHEMA='p-pf-cache-semantic-seal-v1'


def exact(a,b):return backend.encoded(a)==backend.encoded(b)


def fixed_delta(producer,reader,delta,producer_runtime,reader_runtime):
    from analysis.producer_dag_binding import _source,_runtime
    _source(producer);_source(reader);_runtime(producer_runtime);_runtime(reader_runtime)
    if (type(delta)is not dict or set(delta)!=ADDITIONS or set(delta)&set(producer['files'])
        or reader['files']!={**producer['files'],**delta} or not exact(producer_runtime,reader_runtime)):
        raise ValueError('exact additive source delta and identical runtime required')
    return {'producer_commit':producer['commit'],'reader_commit':reader['commit'],'added':delta}


def inventory(root):
    root=compact.checked_root(root)
    if not root.is_dir():raise ValueError('existing PF cache required')
    if {p.name for p in root.iterdir()}!={'train-offsets','validation-offsets','side'}:
        raise ValueError('closed PF cache root file set required')
    if (root/'side').is_symlink()or not(root/'side').is_dir():raise ValueError('real side directory required')
    files={}
    for path in root.rglob('*'):
        compact.checked_root(path)
        relative=path.relative_to(root).as_posix()
        if path.is_dir():
            if relative!='side':raise ValueError('extra cache directory')
            continue
        if not path.is_file():raise ValueError('ordinary cache file required')
        if relative.endswith('-offsets')and relative in ('train-offsets','validation-offsets'):
            dtype='<u8';width=8
        elif re.fullmatch(r'side/week-(?:[0-9]|1[0-9]|2[0-4])\.bin',relative):
            dtype=compact.DenseSide.DTYPE.descr;width=compact.DenseSide.DTYPE.itemsize
        else:raise ValueError('extra cache file')
        size=path.stat().st_size
        if not size or size%width:raise ValueError('cache dtype/shape byte size differs')
        before=compact.file_identity(path);digest=backend.file_sha(path)
        if compact.file_identity(path)!=before:raise ValueError('cache changed while hashing')
        files[relative]={'sha256':digest,'bytes':size,'dtype':dtype,'shape':[size//width]}
    if not any(name.startswith('side/')for name in files):raise ValueError('nonempty history side required')
    return json.loads(backend.encoded(files))


def _stat_identity(stat):
    return stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns


def mapped_file(path,dtype,shape,expected_sha):
    """Hash and map one pinned open fd; never map a separately reopened path."""
    import os
    path=compact.checked_root(path);before=compact.file_identity(path)
    with path.open('rb')as stream:
        if _stat_identity(os.fstat(stream.fileno()))!=before:raise ValueError('mmap inode identity changed before open')
        digest=hashlib.sha256()
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
        if digest.hexdigest()!=expected_sha:raise ValueError('mapped fd bytes differ from sealed hash')
        if (_stat_identity(os.fstat(stream.fileno()))!=before
            or compact.file_identity(path)!=before):raise ValueError('mmap inode identity changed while hashing')
        stream.seek(0)
        array=np.memmap(stream,dtype=dtype,mode='r',shape=shape)
        if (_stat_identity(os.fstat(stream.fileno()))!=before
            or compact.file_identity(path)!=before):raise ValueError('mmap inode identity changed during mapping')
    array._sealed_identity=before
    return array


def mapped_closure(array):
    if compact.file_identity(array.filename)!=array._sealed_identity:
        raise ValueError('mapped inode identity changed during closure')


def _offsets(root,split,record,sealed):
    if type(record)is not dict or set(record)!={'path','sha256','count'} or type(record['count'])is not int or record['count']<1:
        raise ValueError('closed positive input record required')
    path=compact.checked_root(root)/(split+'-offsets')
    if sys.byteorder!='little'or path.stat().st_size!=(record['count']+1)*8:raise ValueError('original little-endian offset dtype/count differs')
    array=mapped_file(path,'<u8',(record['count']+1,),sealed[split+'-offsets']['sha256'])
    source=backend.checked_path(record['path'])
    if int(array[0])!=0 or int(array[-1])!=source.stat().st_size:raise ValueError('offset boundary differs')
    return source,array


def verify_semantics(root,records,partitions,durations,encode,guard):
    """One new read-only revalidation, no claim about old PF output timestamp."""
    if set(records)!={'train','validation'}:raise ValueError('train/validation only')
    guard();root=compact.checked_root(root);before=inventory(root);guard()
    arrays={int(name.split('-')[-1].split('.')[0]):mapped_file(root/name,compact.DenseSide.DTYPE,tuple(before[name]['shape']),before[name]['sha256'])for name in before if name.startswith('side/')}
    wanted={week:np.zeros(len(array),dtype=np.uint8)for week,array in arrays.items()}
    maxima={};counts={}
    for split in ('train','validation'):
        source,offsets=_offsets(root,split,records[split],before);digest=hashlib.sha256();position=0;count=0
        with source.open('rb')as stream:
            for ordinal,raw in enumerate(stream):
                guard()
                if ordinal>=records[split]['count']or int(offsets[ordinal])!=position:raise ValueError('offset line order differs')
                digest.update(raw);position+=len(raw);count+=1
                if int(offsets[ordinal+1])!=position:raise ValueError('offset line boundary differs')
                row=backend.strict_json(raw);items=row['history'];positions=row['history_positions']
                if not items or len(items)!=len(positions):raise ValueError('original history/position alignment required')
                for item,loc in zip(items,positions):
                    if (type(item)is not int or not 0<=item<2**32 or type(loc)is not list or len(loc)!=2
                        or any(type(v)is not int or v<0 for v in loc)or loc[0]>24):raise ValueError('train-only physical history required')
                    week,index=loc;array=arrays.get(week)
                    if array is None or index>=len(array):raise ValueError('history missing from side cache')
                    record=array[index]
                    if record['state']!=2 or int(record['item'])!=item:raise ValueError('side wanted/item/state differs')
                    wanted[week][index]=1;maxima[week]=max(maxima.get(week,-1),index)
        if count!=records[split]['count']or digest.hexdigest()!=records[split]['sha256']:raise ValueError('fixed original JSONL bytes/count differ')
        counts[split]=count
    if set(maxima)!=set(arrays)or any(len(arrays[w])!=maximum+1 for w,maximum in maxima.items()):raise ValueError('exact physical side maxima/shape differs')
    previous=-1;raw_rows=0
    for week,events in partitions:
        if type(week)is not int or not previous<week<=24:raise ValueError('ordered original train partitions required')
        previous=week;array=arrays.get(week)
        for ordinal,row in enumerate(events):
            raw_rows+=1
            if ordinal%65536==0:guard()
            if array is None or ordinal>=len(array)or wanted[week][ordinal]!=1:continue
            record=array[ordinal];item=int(record['item'])
            expected=encode(row,durations.get(item))
            if (row.get('item_id')!=item or type(expected)is not list or len(expected)!=3
                or any(type(v)is not int or not 0<=v<=255 for v in expected)
                or record['side'].tolist()!=expected):raise ValueError('side differs from original raw event encoding')
            wanted[week][ordinal]=2
    positions=0
    for week,array in arrays.items():
        for start in range(0,len(array),65536):
            guard();mark=wanted[week][start:start+65536];block=array[start:start+65536]
            if np.any(mark==1)or np.any(block['state']!=np.where(mark==2,2,0)):raise ValueError('missing or extra wanted side record')
            unused=block[mark==0]
            if np.any(unused.view('u1')):raise ValueError('unused side slots must retain original zero bytes')
            positions+=int(np.count_nonzero(mark==2))
    if inventory(root)!=before:raise ValueError('cache changed during semantic verification')
    guard()
    return {'rows':counts,'wanted_positions':positions,'raw_rows_scanned':raw_rows,'files':before,
            'scope':'current_cache_revalidated_not_historical_PF_bytes','scientific_training':False}


def readonly_views(root,records,sealed,guard):
    """Original objects, unchanged methods; mode=r and exact current byte closure."""
    root=compact.checked_root(root)
    if not exact(inventory(root),sealed):raise ValueError('sealed cache bytes/set/dtype/shape differ')
    side=compact.DenseSide.__new__(compact.DenseSide)
    side.arrays={int(name.split('-')[-1].split('.')[0]):mapped_file(root/name,compact.DenseSide.DTYPE,tuple(record['shape']),record['sha256'])for name,record in sealed.items()if name.startswith('side/')}
    side.guard=guard;side.identities={week:array._sealed_identity for week,array in side.arrays.items()}
    result={}
    for split in ('train','validation'):
        source,offsets=_offsets(root,split,records[split],sealed)
        source_identity=compact.file_identity(source)
        if backend.file_sha(source)!=records[split]['sha256']:raise ValueError('original JSONL hash differs')
        rows=compact.JsonRows.__new__(compact.JsonRows);rows.source=source;rows.offsets=offsets;rows.guard=guard
        if compact.file_identity(source)!=source_identity:raise ValueError('JSONL inode identity changed during hash')
        rows._identity=source_identity;rows._offset_identity=offsets._sealed_identity
        result[split]=compact.SideRows(rows,side)
    views_closure(result)
    guard();return result


def views_closure(views):
    for array in views['train'].side.arrays.values():mapped_closure(array)
    for view in views.values():
        mapped_closure(view.rows.offsets)
        if view.rows.identity()!=view.rows._identity:raise ValueError('JSONL inode identity changed during closure')



# Executed in the original clean consumer deployment, not imported as a new
# consumer claiming its old identity. No context rebuild or vector load occurs.
ORIGINAL_PROOF=r"""
import json,sys
from pathlib import Path
from analysis import p_full_training_backend as b,producer_dag_binding as d
x=json.loads(sys.argv[1]);i=b.record_path(x['inputs']);r=b.record_path(x['request'])
v=b.read_transport(i,r,x['request']['sha256']);run=Path(x['run_dir'])
path=b.stage_receipt_path(Path(v['output_root']),'preflight')
if path!=b.record_path(x['manifest']) or run!=Path(v['ops_root'])/'runs'/v['run_ids']['preflight']:raise ValueError('original PF paths differ')
proof=d.verify_producer(path,run,x['pin']);receipt=proof['manifest']
records=b.stage_inputs(v,i,r,x['request']['sha256'],'preflight');expected=b.job_contract(v,records,r,x['request']['sha256'],'preflight')
actual=b.strict_json((run/'run.json').read_bytes())
if set(actual)-set(expected)-{'created_at','admission'} or any(b.encoded(actual.get(k))!=b.encoded(value)for k,value in expected.items()):raise ValueError('original complete PF job differs')
for record in records:
 p=b.record_path({'path':record['path'],'sha256':record['sha256']})
 if p.stat().st_size!=record['bytes']:raise ValueError('original PF input size differs')
semantics={'schema':'p-full-backend-receipt-v1','stage':'preflight','run_id':v['run_ids']['preflight'],'source':v['source'],'runtime':v['runtime'],'request_sha256':x['request']['sha256'],'run_manifest_sha256':b.file_sha(run/'run.json'),'verified':True,'engineering':False,'test_read':False,'training':False,'scientific_verified':False,'status':'verified','engineering_cache_written':True,'payload_written':False}
if any(b.encoded(receipt.get(k))!=b.encoded(value)for k,value in semantics.items()):raise ValueError('original PF receipt semantics differ')
b.close_context(v,receipt['binding'])
if d.verify_producer(path,run,x['pin'])!=proof:raise ValueError('original PF evidence changed')
print(json.dumps({'value':v,'receipt':receipt,'proof':proof,'records':records},sort_keys=True))
"""


def original_pf(reference):
    import os,subprocess
    from analysis.sgns_subset_acceptance import source_tree
    fields={'root','source','inputs','request','manifest','run_dir','pin'}
    if type(reference)is not dict or set(reference)!=fields:raise ValueError('closed original successful PF reference required')
    root=backend.checked_path(reference['root'])
    if (reference['source'].get('commit')!=BASE_COMMIT or any(reference['source'].get('files',{}).get(k)!=v for k,v in BASE_FILES.items())):
        raise ValueError('fixed original PF source required')
    source_tree(root,reference['source'])
    if reference['pin']['source']!=reference['source']:raise ValueError('original PF pin/source differs')
    for role in ('inputs','request','manifest'):backend.record_path(reference[role])
    env={k:v for k,v in os.environ.items()if k not in ('PYTHONPATH','PYTHONHOME')}
    env.update(PYTHONPATH=str(root/'src'),PYTHONDONTWRITEBYTECODE='1')
    proc=subprocess.run([sys.executable,'-B','-c',ORIGINAL_PROOF,backend.encoded(reference).decode()],cwd=root,env=env,capture_output=True)
    if proc.returncode:raise ValueError('original successful PF closure rejected')
    source_tree(root,reference['source'])
    result=backend.strict_json(proc.stdout)
    if (type(result)is not dict or set(result)!={'value','receipt','proof','records'}
        or result['value']['source']!=reference['source']or result['receipt']!=result['proof']['manifest']):raise ValueError('original PF proof differs')
    return result


def read_request(path,digest):
    value=backend.strict_json(backend.record_path({'path':str(path),'sha256':digest}).read_bytes())
    fields={'schema','original','reader','ops_root','run_id','job_contract','output'}
    if type(value)is not dict or set(value)!=fields or value['schema']!='p-pf-cache-reuse-request-v1':raise ValueError('closed cache reuse request required')
    if type(value['reader'])is not dict or set(value['reader'])!={'source','runtime','delta'}:raise ValueError('closed reader identity required')
    for name in ('ops_root','output'):backend.checked_path(value[name])
    if type(value['run_id'])is not str or re.fullmatch('[a-z0-9][a-z0-9._-]{0,127}',value['run_id'])is None:raise ValueError('fixed sealing run ID required')
    return value


def context(path,digest):
    from analysis.sgns_subset_acceptance import source_tree
    from analysis.new_instance_prepare import runtime
    value=read_request(path,digest);proof=original_pf(value['original']);old=proof['value'];reader=value['reader']
    delta=fixed_delta(value['original']['source'],reader['source'],reader['delta'],old['runtime'],reader['runtime'])
    source_tree(ROOT,reader['source'])
    if runtime(backend.record_path(old['lock']))!=reader['runtime']:raise ValueError('actual reader runtime differs')
    root=compact.cache_root(old['output_root'])/old['run_ids']['preflight']
    output=backend.checked_path(value['output'])
    for protected in (ROOT,Path(value['original']['root']),root,Path(old['stage2']),Path(old['output_root']),Path(old['native_root'])):
        if output==protected or output in protected.parents or protected in output.parents:raise ValueError('new seal separate from all original artifacts/source required')
    manifest=backend.strict_json((Path(old['stage2'])/'UR001-manifest.json').read_bytes())
    records={split:{'path':str(Path(old['stage2'])/('UR001-'+split+'.jsonl')),
                   'sha256':manifest['artifacts']['UR001-'+split+'.jsonl']['sha256'],
                   'count':manifest['counts']['train_examples'if split=='train'else'eligible']}for split in ('train','validation')}
    if records['train']['count']!=proof['receipt']['training_rows']or records['validation']['count']!=proof['receipt']['validation_queries']:raise ValueError('original PF and DAG counts differ')
    return value,proof,root,records,delta


def input_records(value,proof,root,request,request_sha):
    records=[];seen={}
    def add(path,digest):
        path=backend.checked_path(str(path))
        if str(path)in seen:
            if digest!=seen[str(path)]:raise ValueError('conflicting cache input pin')
            return
        seen[str(path)]=digest;records.append({'path':str(path),'sha256':digest,'bytes':path.stat().st_size})
    add(request,request_sha)
    original=value['original']
    for role in ('inputs','request','manifest'):add(original[role]['path'],original[role]['sha256'])
    for name,digest in original['pin']['job_sha256'].items():add(Path(original['run_dir'])/name,digest)
    for record in proof['records']:add(record['path'],record['sha256'])
    for name,digest in original['source']['files'].items():add(Path(original['root'])/name,digest)
    for name,record in inventory(root).items():add(root/name,record['sha256'])
    return records


def invocation(stage,inputs,request,request_sha):
    return [sys.executable,'-B','-m','analysis.p_pf_cache_reuse','seal','--request',str(request),'--request-sha256',request_sha,'--execute-readonly']


def transport(value,proof):
    from types import FunctionType
    namespace={**vars(backend),'ROOT':ROOT,'invocation':invocation}
    for name in ('job_contract','check_job'):
        f=getattr(backend,name);namespace[name]=FunctionType(f.__code__,namespace,f.__name__,f.__defaults__,f.__closure__)
    old=proof['value']
    own={'source':value['reader']['source'],'config':old['config'],'ops_root':value['ops_root'],
         'run_ids':{'seal':value['run_id']},'job_contract':{'seal':value['job_contract']}}
    return own,namespace


def expected_job(value,proof,root,path,digest):
    records=input_records(value,proof,root,path,digest);own,functions=transport(value,proof)
    # Original job_contract requires its first two ordered records; the second
    # is the original admission file. The G/P scientific schema is not reused.
    return functions['job_contract'](own,records,path,digest,'seal'),records,own,functions


def seal(path,digest):
    value,proof,root,records,delta=context(path,digest)
    job,inputs,own,functions=expected_job(value,proof,root,path,digest)
    before=functions['check_job'](own,inputs,path,digest,'seal');old=proof['value']
    native=backend.native_for(Path(old['native_root']));guard=compact.make_guard(native,old)
    import importlib
    side_module=importlib.import_module(native.__package__+'.history_side')
    catalog=backend.strict_json((Path(old['stage2'])/'UR001-catalog.json').read_bytes())
    durations=side_module.load_durations(backend.record_path(old['items_metadata']),catalog)
    admitted=backend.strict_json(Path(value['original']['inputs']['path']).read_bytes())
    import pyarrow.parquet as pq
    columns=['user_id','item_id',*side_module.FEEDBACK_BITS,'timespent']
    def events(file):
        backend.stage_progress('cache_seal_raw_week_begin',week=Path(file).stem)
        for batch in pq.ParquetFile(file).iter_batches(batch_size=65536,columns=columns,use_threads=False):
            guard();yield from batch.to_pylist()
        backend.stage_progress('cache_seal_raw_week_complete',week=Path(file).stem)
    semantic=verify_semantics(root,records,((week,events(file))for week,file in enumerate(admitted['train'])),durations,side_module._encode,guard)
    after=context(path,digest)
    if not exact(after[:2],(value,proof))or not exact(after[3:],(records,delta))or after[2]!=root:raise ValueError('seal source/input/DAG closure changed')
    if functions['check_job'](own,inputs,path,digest,'seal')!=before:raise ValueError('own sealing job changed')
    receipt={'schema':SCHEMA,'status':'verified','verified':True,'training':False,'scientific_verified':False,
             'engineering':False,'test_read':False,'source':value['reader']['source'],'runtime':value['reader']['runtime'],
             'run_id':value['run_id'],'run_manifest_sha256':before,'request':{'path':str(path),'sha256':digest},
             'original_pf':value['original'],'original_binding':proof['receipt']['binding'],'reader_delta':delta,
             'cache_root':str(root),'inputs':records,'semantic':semantic}
    output=Path(value['output']);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb')as stream:stream.write(backend.encoded(receipt))
    return receipt


def consume(reference,guard):
    """Original scientific binding and new reader identity are distinct outputs."""
    from analysis.producer_dag_binding import verify_producer
    if type(reference)is not dict or set(reference)!={'manifest','run_dir','pin'}:raise ValueError('original successful seal producer required')
    path=backend.record_path(reference['manifest']);proof=verify_producer(path,Path(reference['run_dir']),reference['pin']);receipt=proof['manifest']
    if (receipt.get('schema')!=SCHEMA or receipt.get('training')is not False or receipt.get('scientific_verified')is not False
        or receipt.get('engineering')is not False or receipt.get('test_read')is not False):raise ValueError('readonly seal semantics required')
    request=receipt['request'];value,original,root,records,delta=context(Path(request['path']),request['sha256'])
    if (not exact(receipt.get('source'),value['reader']['source'])
        or not exact(receipt.get('runtime'),value['reader']['runtime'])
        or receipt.get('run_id')!=value['run_id']or receipt.get('status')!='verified'
        or receipt.get('verified')is not True):raise ValueError('seal reader source/runtime/run identity differs')
    if Path(value['output'])!=path or Path(reference['run_dir'])!=Path(value['ops_root'])/'runs'/value['run_id']:raise ValueError('fixed sealing producer paths differ')
    expected,inputs,own,functions=expected_job(value,original,root,Path(request['path']),request['sha256'])
    actual=backend.strict_json((Path(reference['run_dir'])/'run.json').read_bytes())
    if set(actual)-set(expected)-{'created_at','admission'}or any(not exact(actual.get(k),v)for k,v in expected.items()):raise ValueError('complete sealing own job differs')
    for record in inputs:
        if backend.record_path({'path':record['path'],'sha256':record['sha256']}).stat().st_size!=record['bytes']:raise ValueError('sealed producer input bytes differ')
    if (not exact(receipt['original_pf'],value['original'])or not exact(receipt['original_binding'],original['receipt']['binding'])
        or not exact(receipt['reader_delta'],delta)or not exact(receipt['inputs'],records)or receipt['cache_root']!=str(root)
        or receipt['run_manifest_sha256']!=backend.file_sha(Path(reference['run_dir'])/'run.json')
        or receipt['semantic'].get('scope')!='current_cache_revalidated_not_historical_PF_bytes'
        or receipt['semantic'].get('scientific_training')is not False):raise ValueError('sealed current semantic input/provenance differs')
    if guard is None:
        guard=compact.make_guard(backend.native_for(Path(original['value']['native_root'])),original['value'])
    views=readonly_views(root,records,receipt['semantic']['files'],guard)
    after=context(Path(request['path']),request['sha256'])
    if not exact(after[:2],(value,original))or after[2]!=root or not exact(after[3:],(records,delta)):
        raise ValueError('reader source/input/DAG closure changed')
    if not exact(inventory(root),receipt['semantic']['files']):raise ValueError('opened cache bytes changed')
    if verify_producer(path,Path(reference['run_dir']),reference['pin'])!=proof:raise ValueError('sealing producer evidence changed')
    views_closure(views)
    return {'rows':views['train'],'queries':views['validation'],'scientific_binding':original['receipt']['binding'],
            'reader_identity':{'schema':'p-cache-reader-dual-identity-v1','source':value['reader']['source'],'runtime':value['reader']['runtime'],
                               'delta':delta,'seal_producer':proof['binding'],'original_pf_producer':original['proof']['binding']}}

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True);sub.add_parser('dry')
    for name in ('plan','seal'):
        p=sub.add_parser(name);p.add_argument('--request',required=True,type=Path);p.add_argument('--request-sha256',required=True)
        if name=='seal':p.add_argument('--execute-readonly',action='store_true',required=True)
    p=sub.add_parser('verify');p.add_argument('--reference',required=True,type=Path);p.add_argument('--reference-sha256',required=True)
    args=parser.parse_args(argv)
    try:
        if args.command=='dry':result={'status':'dry','training':False,'cache_opened':False,'execution_allowed':False,'schema':SCHEMA}
        elif args.command=='seal':result=seal(args.request,args.request_sha256)
        elif args.command=='plan':
            value,proof,root,records,delta=context(args.request,args.request_sha256)
            job,inputs,_,_=expected_job(value,proof,root,args.request,args.request_sha256)
            result={'status':'planned','training':False,'submitted':False,'expected_job':job,'inputs':inputs,'semantic_scan_executed':False}
        else:
            reference=backend.strict_json(backend.record_path({'path':str(args.reference),'sha256':args.reference_sha256}).read_bytes())
            value=consume(reference,None)
            result={'status':'verified','verified':True,'training':False,'rows':len(value['rows']),'queries':len(value['queries']),'reader_identity':value['reader_identity']}
    except (ValueError,KeyError,TypeError,OSError,RuntimeError)as exc:
        print(json.dumps({'status':'rejected','verified':False,'training':False,'error_type':type(exc).__name__}));return 2
    print(json.dumps(result,sort_keys=True));return 0


if __name__=='__main__':raise SystemExit(main())
