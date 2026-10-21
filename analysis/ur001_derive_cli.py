"""UR001 fixed train/validation derivation, read-only preflight and manifest DAG."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
REVISION = '084a18386e149a454dd6dbe0123478e75c6b09ce'
A_SHA = '64f7eec73ac04c92b4b1a283e6e6223a8a3b9a2f88e20328b0a0c67e19ced158'
INVENTORIES = {
    'full': ('configs/ur001-preparation/hf-files.json', '19cd71f02aff4fce93349907dbc5fcdbf006afc679ee899d9c98965123270fa0', 'subsamples/ur0.01'),
    'subset': ('configs/a40-subset/hf-files.json', 'd7df6268431bb0d0bcd77d56809bf8920914b4b8858f58f0dac409c7df81ff49', 'subsamples/ur0.01_ir0.01'),
}
SCHEMA = 'UR001-HIST_V2_OBS-L50-v1'
RAM_BYTES = 96 * 1024**3  # Conservative first admission, not_measured.
DISK_BYTES = 200 * 1024**3


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'))+'\n').encode()


def progress(phase, **fields):
    print(json.dumps({'progress': phase, 'monotonic': time.monotonic(), **fields}, sort_keys=True), flush=True)


def secure(path):
    path = Path(path)
    if not path.is_absolute() or '..' in path.parts:
        raise ValueError('canonical absolute path required')
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('symlink task path refused')
    if any('holdout' in p.lower() or p.lower() == 'test' or 'week_26' in p.lower() for p in path.parts):
        raise ValueError('test/holdout path forbidden')
    return path


def observed_rows(rows, guard, interval=100000):
    for i, row in enumerate(rows):
        if i % interval == 0:
            guard()
        yield row


def records_for(profile):
    rel, sha, prefix = INVENTORIES[profile]
    path = ROOT/rel
    if digest(path) != sha:
        raise ValueError('fixed input inventory differs')
    inventory = json.loads(path.read_text())
    if inventory['revision'] != REVISION or inventory['dataset'] != 'deepvk/VK-LSVD':
        raise ValueError('dataset/revision differs')
    records = []
    for row in inventory['files']:
        if row['path'].startswith(prefix+'/'):
            records.append({**row, 'path': row['path'][len(prefix)+1:]})
    return records, sha, prefix


def inspect_inputs(data, records, workers):
    from analysis.ur001_derivation import week_files
    import pyarrow as pa
    import pyarrow.parquet as pq
    data = secure(data)
    if workers not in (1, 2, 4):
        raise ValueError('bounded workers 1/2/4 required')
    paths = week_files(data/'train', range(25)) + week_files(data/'validation', [25])
    expected = [str(p.relative_to(data)) for p in paths]
    if [r['path'] for r in records] != expected:
        raise ValueError('exact train00-24/validation25 inventory required')
    result = []
    for path, row in zip(paths, records):
        secure(path)
        if path.stat().st_size != row['size'] or digest(path) != row['sha256']:
            raise ValueError('input size/SHA differs')
        parquet = pq.ParquetFile(path)
        for key, dtype in (('user_id', pa.uint32()), ('item_id', pa.uint32()), ('like', pa.bool_())):
            if parquet.schema_arrow.field(key).type != dtype:
                raise ValueError('interaction schema differs')
        groups = [parquet.metadata.row_group(i).num_rows for i in range(parquet.metadata.num_row_groups)]
        result.append({**row, 'rows': parquet.metadata.num_rows, 'row_groups': groups})
        progress('input_verified', file=row['path'], rows=parquet.metadata.num_rows)
    return {'schema': 'UR001-input-manifest-v1', 'revision': REVISION, 'files': result,
            'train_rows': sum(r['rows'] for r in result[:-1]), 'validation_rows': result[-1]['rows'],
            'row_order': 'file row group order then row order; ordinal starts at zero', 'test_paths': 0,
            'workers': workers, 'read_buffer_bound_bytes': max(max(r['row_groups'], default=0) for r in result)*(workers+1)*1024}


def guards(output, *, projected_memory=0, projected_disk=0):
    from vkgr.throughput_config import Engineering, guard
    cfg = Engineering(memory_limit_bytes=RAM_BYTES, planning_memory_bytes=513000000000)
    guard(cfg, output.parent, projected_memory=projected_memory, projected_disk=projected_disk)


def source_identity():
    if subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain','--untracked-files=all'], text=True).strip():
        raise ValueError('clean fixed deployment required')
    commit = subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'], text=True).strip()
    names = subprocess.check_output(['git','-C',str(ROOT),'ls-tree','-r','--name-only',commit], text=True).splitlines()
    files = {}
    for rel in names:
        raw = subprocess.check_output(['git','-C',str(ROOT),'show',commit+':'+rel])
        if (ROOT/rel).is_symlink() or (ROOT/rel).read_bytes() != raw:
            raise ValueError('source differs from Git')
        files[rel] = hashlib.sha256(raw).hexdigest()
    return {'commit':commit,'files':files}


def write_json(path, value):
    with path.open('xb') as stream:
        stream.write(encoded(value))


def oracle_checks(parts, validation, emitted_hash, result):
    from vkgr.retrieval_data import prefix_examples
    from vkgr.data_audit import build_queries, approved_policy
    from vkgr.history_v2 import build_observed, overlay_examples
    old = list(prefix_examples(parts))
    queries = build_queries(parts, validation, approved_policy(), set(result['catalog']))['queries']
    observed = build_observed(parts, old, queries, result['catalog'], max_len=50)
    old_train = overlay_examples(old, observed['train'], 50, set(result['catalog']))
    h = hashlib.sha256()
    for row in old_train:
        h.update(encoded(row))
    if (h.hexdigest() != emitted_hash or encoded(observed['validation']) != encoded(result['validation'])
            or encoded(observed['slices']) != encoded(result['slices'])):
        raise ValueError('real subset oracle differs bitwise')
    return {'bitwise_equal':True,'train_sha256':emitted_hash,'validation_rows':len(queries),
            'oracles':['prefix_examples','build_queries','build_observed','overlay_examples']}


def write_derivation(data, output, inputs, *, workers, guard, source, runtime, oracle, final_check=None):
    from analysis.ur001_derivation import derive, parquet_rows
    data, output = secure(data), secure(output)
    output.mkdir(exist_ok=False)
    write_json(output/'UR001-input-manifest.json', inputs)
    guard()
    # Regression intentionally materializes only the original small subset.
    if oracle:
        parts = [(i, list(parquet_rows(data/f'train/week_{i:02d}.parquet', workers))) for i in range(25)]
        validation = (25,list(parquet_rows(data/'validation/week_25.parquet', workers)))
    else:
        parts = [(i, observed_rows(parquet_rows(data/f'train/week_{i:02d}.parquet',workers),guard)) for i in range(25)]
        validation = (25,observed_rows(parquet_rows(data/'validation/week_25.parquet',workers),guard))
    h = hashlib.sha256();written = 0
    with (output/'UR001-train.jsonl').open('xb', buffering=1024**2) as stream:
        def emit(row):
            nonlocal written
            raw = encoded(row);stream.write(raw);h.update(raw);written += 1
            if written % 10000 == 0:
                guard();progress('train_targets_written', examples=written)
        progress('derive_start', oracle=oracle)
        result = derive(parts, validation, emit)
    guard()
    checked = oracle_checks(parts,validation,h.hexdigest(),result) if oracle else None
    result['counts']['catalog_items'] = len(result['catalog'])
    progress('derive_complete', counts=result['counts'])
    for name, value in (('validation',result['validation']),('catalog',result['catalog']),
                        ('supported',result['supported']),('S0',result['slices']),('counts',result['counts'])):
        if name == 'validation':
            with (output/'UR001-validation.jsonl').open('xb') as stream:
                for row in value:stream.write(encoded(row))
        else:
            write_json(output/f'UR001-{name}.json',value)
        guard()
    artifacts = {}
    upstream = {'UR001-input-manifest.json':digest(output/'UR001-input-manifest.json')}
    for path in sorted(output.iterdir()):
        if path.name == 'UR001-input-manifest.json':continue
        artifacts[path.name] = {'sha256':digest(path),'size':path.stat().st_size,'schema':SCHEMA,'upstream':upstream}
    completion = final_check(result['counts']) if final_check is not None else {'fixture':True}
    manifest = {'schema':'UR001-derivation-manifest-v1','completion':completion,'history_schema':SCHEMA,'revision':REVISION,
                'source':source,'runtime':runtime,'upstream':upstream,'artifacts':artifacts,
                'counts':result['counts'],'oracle':checked,'train_fields':['example_id','user_id','target','target_position','history','history_positions','past_likes'],
                'catalog_mapping':'sorted catalog; PAD=0, observed items enumerate from 1; OOV absent',
                'resources':{'peak_rss_bytes':int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform=='darwin' else 1024)},
                'test_query_constructed':False,'science_training':False}
    write_json(output/'UR001-manifest.json',manifest)
    return manifest


def verify_dag(root, *, require_profile=None, source=None, runtime=None):
    root = secure(root)
    manifest = json.loads((root/'UR001-manifest.json').read_text())
    if manifest['schema'] != 'UR001-derivation-manifest-v1' or manifest['test_query_constructed'] is not False:
        raise ValueError('DAG identity differs')
    if set(manifest['upstream']) != {'UR001-input-manifest.json'} or not manifest.get('source') or not manifest.get('runtime'):
        raise ValueError('DAG missing source/runtime/upstream')
    for rel, sha in manifest['upstream'].items():
        if rel != 'UR001-input-manifest.json' or digest(secure(root/rel)) != sha:
            raise ValueError('DAG upstream differs')
    inputs = json.loads((root/'UR001-input-manifest.json').read_text())
    if inputs.get('revision') != REVISION or inputs.get('test_paths') != 0:
        raise ValueError('DAG input revision/test boundary differs')
    if manifest.get('revision') != REVISION or manifest.get('history_schema') != SCHEMA or manifest.get('science_training') is not False:
        raise ValueError('DAG science identity differs')
    if require_profile is not None:
        records, inventory_sha, prefix = records_for(require_profile)
        if (inputs.get('profile') != require_profile or inputs.get('inventory_sha256') != inventory_sha
                or inputs.get('a_report_sha256') != A_SHA or inputs.get('source') != manifest['source']
                or inputs.get('runtime') != manifest['runtime']):
            raise ValueError('DAG formal input provenance differs')
        if manifest.get('completion') != {'verified':True,'input_files':{r['path']:r['sha256'] for r in inputs['files']}}:
            raise ValueError('DAG missing post-read completion checks')
        recorded = [{key:r[key] for key in ('path','size','sha256')} for r in inputs['files']]
        if recorded != records:
            raise ValueError('DAG fixed raw inventory differs')
        if source is not None and manifest['source'] != source:
            raise ValueError('DAG formal source differs')
        if runtime is not None and manifest['runtime'] != runtime:
            raise ValueError('DAG formal runtime differs')
    if json.loads((root/'UR001-counts.json').read_text()) != manifest['counts']:
        raise ValueError('DAG counts differ')
    expected = {'UR001-train.jsonl','UR001-validation.jsonl','UR001-catalog.json','UR001-supported.json','UR001-S0.json','UR001-counts.json'}
    if set(manifest['artifacts']) != expected:
        raise ValueError('DAG artifact set differs')
    for rel, record in manifest['artifacts'].items():
        if record['upstream'] != manifest['upstream'] or digest(secure(root/rel)) != record['sha256'] or (root/rel).stat().st_size != record['size']:
            raise ValueError('DAG artifact differs')
    if manifest.get('oracle') is not None:
        if (manifest['oracle'].get('bitwise_equal') is not True
                or manifest['oracle'].get('train_sha256') != manifest['artifacts']['UR001-train.jsonl']['sha256']):
            raise ValueError('DAG oracle train stream differs')
    if require_profile == 'full' and manifest['counts'] != inputs.get('expected_counts'):
        raise ValueError('DAG full A count conservation differs')
    return {'verified':True,'manifest_sha256':digest(root/'UR001-manifest.json'),'counts':manifest['counts'],
            'oracle':manifest['oracle'],'peak_rss_bytes':manifest['resources']['peak_rss_bytes'],'science_training':False,'test_query_constructed':False}


def expected_counts(aggregate):
    if digest(aggregate) != A_SHA:
        raise ValueError('fixed A report differs')
    a = json.loads(Path(aggregate).read_text());train, val = a['train'], a['validation']
    return {'train_rows':train['rows'],'train_users':train['users'],'train_like_events':train['like_events'],
            'train_like_users':train['like_users'],'train_examples':train['like_events']-train['like_users'],'catalog_items':train['items'],
            'validation_rows':val['rows'],'validation_like_events':val['like_events'],**val['candidate_counts']}


def check_preflight(run_path, inputs, args):
    path = secure(run_path)
    run = json.loads((path/'run.json').read_text())
    terminal = json.loads((path/'terminal.json').read_text())
    stdout = path/'logs/stdout.log'
    # The final stdout record is the original producer receipt; preceding records are progress.
    with stdout.open('rb') as stream:
        last = None
        for line in stream:
            if line.strip():last = line
    receipt = json.loads(last or b'{}')
    argv = run.get('argv', [])
    required = [sys.executable,'-B','-m','analysis.ur001_derive_cli','preflight',
                '--data',str(args.data),'--output',str(args.output),'--profile',args.profile,
                '--aggregate',str(args.aggregate),'--lock',str(args.lock),'--workers',str(args.workers),
                '--execute-on-new-instance']
    if (terminal.get('status') != 'succeeded' or terminal.get('exit_code') != 0
            or terminal.get('run_id') != path.name or run.get('run_id') != path.name
            or run.get('kind') != 'ops' or run.get('git',{}).get('commit') != inputs['source']['commit']
            or argv != required or receipt.get('status') != 'read_only_preflight'
            or receipt.get('engineering') is not False or receipt.get('payload_written') is not False
            or receipt.get('verified') is not True or receipt.get('inputs') != inputs):
        raise ValueError('formal preflight producer/source/input/output binding differs')
    return {name:digest(path/name) for name in ('run.json','terminal.json','logs/stdout.log')}


def execute(args):
    from analysis.new_instance_prepare import runtime
    data, output = secure(args.data), secure(args.output)
    if output.exists() or output == data or output in data.parents or data in output.parents:
        raise FileExistsError('new nonoverlapping output required')
    if ROOT == output or ROOT in output.parents or output in ROOT.parents:
        raise ValueError('data outputs outside source tree required')
    env, source = runtime(args.lock), source_identity()
    records, inventory_sha, prefix = records_for(args.profile)
    data = data/prefix
    guards(output,projected_memory=32*1024**3,projected_disk=DISK_BYTES if args.profile=='full' else 4*1024**3)
    progress('preflight_start', profile=args.profile)
    inputs = inspect_inputs(data,records,args.workers)
    inputs.update({'profile':args.profile,'inventory_sha256':inventory_sha,'data_root':str(data),'source':source,'runtime':env})
    if inputs['read_buffer_bound_bytes'] > 32*1024**3:
        raise MemoryError('row-group prefetch estimate exceeds conservative budget')
    expected = expected_counts(args.aggregate)
    inputs['expected_counts'] = expected
    inputs['a_report_sha256'] = A_SHA
    if args.profile == 'full' and (inputs['train_rows'] != expected['train_rows'] or inputs['validation_rows'] != expected['validation_rows']):
        raise ValueError('A input rows differ')
    if args.command == 'preflight':
        return {'verified':True,'status':'read_only_preflight','engineering':False,'payload_written':False,
                'inputs':inputs,'ram_declaration':'not_measured conservative 96GiB','disk_upper_bytes':DISK_BYTES,
                'science_training':False,'test_query_constructed':False}
    if args.preflight_run is None:
        raise ValueError('formal read-only preflight producer required')
    inputs['preflight_producer'] = check_preflight(args.preflight_run, inputs, args)
    if args.profile == 'full':
        if args.subset_verified is None:
            raise ValueError('original real subset regression required before full derivation')
        prior = verify_dag(secure(args.subset_verified), require_profile='subset', source=source, runtime=env)
        m = json.loads((args.subset_verified/'UR001-manifest.json').read_text())
        if not prior['oracle'] or not prior['oracle']['bitwise_equal'] or m['source'] != source or m['runtime'] != env:
            raise ValueError('subset regression source/runtime differs')
        inputs['subset_regression_sha256'] = prior['manifest_sha256']
    def finalize(counts):
        if args.profile == 'full' and counts != expected:
            raise ValueError('A counts differ; preserve partial outputs and STOP')
        for record in inputs['files']:
            if digest(data/record['path']) != record['sha256']:
                raise ValueError('raw input changed during derivation')
        return {'verified':True,'input_files':{r['path']:r['sha256'] for r in inputs['files']}}
    write_derivation(data,output,inputs,workers=args.workers,guard=lambda:guards(output),
                     source=source,runtime=env,oracle=args.profile=='subset',final_check=finalize)
    return verify_dag(output, require_profile=args.profile, source=source, runtime=env)


def job(args):
    from analysis.a40_ops import load_ops
    _, launcher, scheduler, _ = load_ops(args.ops_code)
    stage = args.stage
    argv = [sys.executable,'-B','-m','analysis.ur001_derive_cli',stage,'--data',str(args.data),
            '--output',str(args.output),'--profile',args.profile,'--aggregate',str(args.aggregate),
            '--lock',str(args.lock),'--workers',str(args.workers),'--execute-on-new-instance']
    if args.preflight_run:argv += ['--preflight-run',str(args.preflight_run)]
    if args.subset_verified:argv += ['--subset-verified',str(args.subset_verified)]
    flags = ['--root',str(args.ops_root),'--run-id',args.run_id,'--kind','ops','--cwd',str(ROOT),
             '--config',str(ROOT/INVENTORIES[args.profile][0]),'--input',str(args.lock),'--input',str(args.aggregate),
             '--env','PYTHONPATH='+str(ROOT/'src'),'--env','OMP_NUM_THREADS=1','--env','OPENBLAS_NUM_THREADS=1',
             '--env','MKL_NUM_THREADS=1','--cpu-cores',str(args.workers+1),'--ram-mib','98304','--vram-mib','0',
             '--disk-gib','200' if args.profile=='full' else '4','--wall-hours','48']
    if args.preflight_run:
        flags += ['--input',str(args.preflight_run/'run.json'),'--input',str(args.preflight_run/'terminal.json'),'--input',str(args.preflight_run/'logs/stdout.log')]
    if args.subset_verified:flags += ['--input',str(args.subset_verified/'UR001-manifest.json')]
    spec = launcher.spec_from_args(launcher.build_parser().parse_args([*flags,'--',*argv]))
    metadata = launcher.prepare(spec)
    if args.command=='prepare':return {'status':'prepared_dry','spec':spec,'metadata':metadata}
    root = launcher.resolve_root(str(args.ops_root))
    if (root/'runs'/args.run_id).exists():raise FileExistsError('run exists')
    for state in ('pending','claimed','rejected'):
        if list((root/'queue'/state).glob('*-'+args.run_id+'.json')):raise FileExistsError('run queued')
    path = scheduler.submit(root,spec)
    return {'status':'written','queue_file':path.name,'run_id':args.run_id}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command',required=True)
    subs.add_parser('dry')
    check = subs.add_parser('verify');check.add_argument('--output',type=Path,required=True);check.add_argument('--profile',choices=INVENTORIES,required=True)
    for command in ('preflight','derive','prepare','submit'):
        p = subs.add_parser(command)
        for name in ('data','output','lock','aggregate'):p.add_argument('--'+name,type=Path,required=True)
        p.add_argument('--profile',choices=INVENTORIES,required=True)
        p.add_argument('--workers',type=int,choices=(1,2,4),default=4)
        p.add_argument('--subset-verified',type=Path)
        p.add_argument('--preflight-run',type=Path)
        if command in ('prepare','submit'):
            p.add_argument('--ops-code',required=True);p.add_argument('--ops-root',type=Path,required=True)
            p.add_argument('--run-id',required=True);p.add_argument('--stage',choices=('preflight','derive'),required=True)
        else:p.add_argument('--execute-on-new-instance',action='store_true',required=True)
    args = parser.parse_args(argv)
    if args.command=='dry':result={'schema':SCHEMA,'science_training':False,'test_query_constructed':False,'profiles':list(INVENTORIES)}
    elif args.command=='verify':result=verify_dag(args.output,require_profile=args.profile)
    elif args.command in ('prepare','submit'):result=job(args)
    else:result=execute(args)
    print(json.dumps(result,sort_keys=True),flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
