"""Fixed full observed-train SGNS preparation; four-gate training is fail closed."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time

from analysis.ur001_derive_cli import digest, encoded, secure, source_identity, guards
from analysis.sgns_gpu_subset import checked_file, snapshot, progress

ROOT = Path(__file__).resolve().parents[1]
PINS_SHA = 'c3ddc90c2265910bfe2362fe86a21d421cd1fbba4a033bdebf0f00fa26175064'
NEW_PATHS = {'analysis/sgns_full_input_bridge.py','configs/full-gpu-sgns/source-bridge-pins.json',
             'tests/test_sgns_full_input_bridge.py','analysis/sgns_subset_acceptance.py','analysis/sgns_subset_acceptance_a1.py',
             'configs/full-gpu-sgns/acceptance-pins.json','tests/test_sgns_subset_acceptance.py',
             'analysis/sgns_gpu_full.py', 'src/vkgr/sgns_full_verify.py',
             'configs/full-gpu-sgns/fixed-dag.json', 'tests/test_sgns_gpu_full.py',
             'tests/fixtures/sgns_full_launcher_chain.py',
             'tests/fixtures/sgns_full_mutations.py',
             'docs/superpowers/reviews/2026-10-21-full-gpu-sgns-implementation.md'}


def pinned_json(path, sha):
    raw = secure(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha:
        raise ValueError('consumed JSON SHA differs')
    return json.loads(raw)


def fixed_dag():
    path = ROOT/'configs/full-gpu-sgns/fixed-dag.json'
    if digest(path) != PINS_SHA:
        raise ValueError('fixed full DAG pin drift')
    return pinned_json(path, PINS_SHA)


def fixed_records():
    from analysis.sgns_gpu_subset import fixed_records as subset_records
    pins = fixed_dag()
    return [{k:r[k] for k in ('path','size','sha256')} | {'path':'subsamples/ur0.01/'+r['path']}
            for r in pins['train_records']] + [subset_records()[-1]]


def check_source(source, deployment_sha):
    pin = fixed_dag()
    base = pin['base_source']['files']
    if (source.get('commit') != deployment_sha or len(deployment_sha) != 40
            or set(source['files']) != set(base) | NEW_PATHS
            or any(source['files'].get(k) != v for k,v in base.items())):
        raise ValueError('exact pinned source delta/deployment differs')
    # Base is content-addressed in fixed-dag.json; new files are pinned by the
    # clean deployment commit, also bound to launcher metadata and every receipt.
    return {'base_commit':pin['base_source']['commit'],
            'stage2_source':pin['stage2_source'],
            'delta':{k:source['files'][k] for k in sorted(NEW_PATHS)}}


def runtime(lock):
    from analysis.new_instance_prepare import runtime as original
    value = original(secure(lock))
    if value != fixed_dag()['runtime']:
        raise ValueError('original full runtime/lock identity differs')
    return value


def receipt_from_bytes(raw):
    return json.loads(next((line for line in reversed(raw.splitlines()) if line.strip()),b'{}'))


def last_receipt(path):
    return receipt_from_bytes(secure(path).read_bytes())


def file_record(path, sha):
    path = secure(path); raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha:
        raise ValueError('fixed launcher input differs')
    return {'path':str(path),'sha256':sha,'bytes':len(raw)}


def full_job_contract(run_path, argv, source):
    if len(argv) == 25:
        from analysis.sgns_full_input_bridge import bridge_inputs
        if argv[4] != 'prepare' or argv[19:23:2] != ['--source-bridge-request','--source-bridge-sha256']:
            raise ValueError('fixed bridged prepare argv differs')
        result = full_job_contract(run_path,argv[:19]+argv[23:],source)
        result['argv'] = argv
        result['inputs'] += [file_record(Path(r['path']),r['sha256']) for r in bridge_inputs(argv[20],argv[22])]
        return result
    path = secure(run_path); pin = fixed_dag()
    names = ('data','embedding-root','stage2','stage2-run','lock','output','deployment-sha')
    if (len(argv) not in (19,21) or argv[:4] != [sys.executable,'-B','-m','analysis.sgns_gpu_full']
            or argv[4] not in ('preflight','prepare') or argv[5:19:2] != ['--'+n for n in names]):
        raise ValueError('fixed full predecessor argv contract differs')
    values = dict(zip(names,argv[6:19:2])); command = argv[4]
    if (values['deployment-sha'] != source['commit']
            or (command == 'preflight' and len(argv) != 19)
            or (command == 'prepare' and (len(argv) != 21 or argv[19] != '--preflight-run'))):
        raise ValueError('fixed full predecessor stage/deployment differs')
    inputs = [file_record(Path(values['lock']),pin['runtime']['lock_sha256']),
              file_record(Path(values['stage2'])/'UR001-manifest.json',pin['stage2_manifest_sha256'])]
    if command == 'prepare':
        prior = secure(Path(argv[20]))
        # These files are also parsed/validated by prepare's producer chain.
        inputs += [file_record(prior/n,digest(secure(prior/n))) for n in ('run.json','terminal.json','logs/stdout.log')]
    env = {'PYTHONPATH':str(ROOT/'src'),'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
           'MKL_NUM_THREADS':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8',
           'VKGR_RUN_ID':path.name,'VKGR_PROGRESS_PATH':str(path/'progress.json')}
    return {'schema':'vkgr-run-v1','run_id':path.name,'kind':'ops','cwd':str(ROOT),'argv':argv,'seed':42,
            'git':{'commit':source['commit'],'dirty':False,'dirty_files':[]},
            'config':{'path':str(ROOT/'configs/full-gpu-sgns/fixed-dag.json'),'sha256':PINS_SHA},
            'inputs':inputs,'env':env,'resources':{'cpu_cores':1,'ram_mib':196608 if command == 'prepare' else 262144,'vram_mib':0,
                'disk_gib':16,'wall_seconds':86400.0},
            'ops_source':{'commit':'5840e6b686ec9651314a9e28a1a2c0befcc05158',
                'sha256':'7a6e105d77f5105b66c39f28fe8849589bc8624b65c95102a53a3964707d4040'},
            'overlay':None,'pinned_files':{},'stop_mode':'signal'}


def producer(run_path, argv, source, receipt, *, expected_job=None, historical=None):
    path = secure(run_path)
    names = ('run.json','terminal.json','logs/stdout.log')
    blobs = {name:secure(path/name).read_bytes() for name in names}
    proof = {name:hashlib.sha256(raw).hexdigest() for name,raw in blobs.items()}
    run = json.loads(blobs['run.json']); terminal = json.loads(blobs['terminal.json'])
    if historical is not None:
        if historical != fixed_dag()['stage2_job'] or proof != historical or source != fixed_dag()['stage2_source']:
            raise ValueError('historical producer binding differs')
        # Complete bytes are independently fixed; this is not an inferred cwd/config.
        expected_job = {k:run.get(k) for k in run}
        if (run.get('seed') is not None or run.get('schema') != 'vkgr-run-v1'
                or run.get('git') != {'commit':source['commit'],'dirty':False,'dirty_files':[]}):
            raise ValueError('historical producer binding semantics differ')
    required_run = {'schema','run_id','kind','cwd','argv','seed','git','config','inputs','env',
                    'resources','ops_source','overlay','pinned_files','stop_mode'}
    required_terminal = {'schema','run_id','status','exit_code','signal','wall_cap_exceeded',
                         'started_at','ended_at','wall_seconds'}
    if (type(expected_job) is not dict or not required_run <= set(expected_job)
            or any(encoded(run.get(k)) != encoded(v) for k,v in expected_job.items())
            or set(run)-set(expected_job)-{'created_at','admission'}
            or run.get('run_id') != path.name or run.get('kind') != 'ops'
            or run.get('git',{}).get('commit') != source['commit'] or run.get('argv') != argv
            or set(terminal) != required_terminal
            or terminal.get('schema') != 'vkgr-terminal-v1' or terminal.get('run_id') != path.name
            or terminal.get('status') != 'succeeded' or type(terminal.get('exit_code')) is not int
            or terminal['exit_code'] != 0 or terminal.get('signal') is not None
            or terminal.get('wall_cap_exceeded') is not False
            or receipt_from_bytes(blobs['logs/stdout.log']) != receipt):
        raise ValueError('job/stdout/manifest/source producer binding differs')
    if any(secure(path/name).read_bytes() != raw for name,raw in blobs.items()):
        raise ValueError('producer snapshot changed during verification')
    return proof


def stage2_binding(stage2, stage2_run):
    pin = fixed_dag(); stage2 = secure(stage2)
    manifest = secure(stage2/'UR001-manifest.json')
    if digest(manifest) != pin['stage2_manifest_sha256']:
        raise ValueError('fixed full stage2 manifest differs')
    value = pinned_json(manifest,pin['stage2_manifest_sha256'])
    if (value['source'] != pin['stage2_source'] or value['runtime'] != pin['runtime']
            or value['counts'] != pin['counts'] or value.get('test_query_constructed') is not False
            or value.get('science_training') is not False
            or value.get('completion',{}).get('verified') is not True):
        raise ValueError('full stage2 producer identity differs')
    input_path = secure(stage2/'UR001-input-manifest.json')
    if digest(input_path) != pin['stage2_input_sha256']:
        raise ValueError('stage2 physical input manifest differs')
    inputs = pinned_json(input_path,pin['stage2_input_sha256'])
    if inputs['files'][:25] != pin['train_records'] or inputs['test_paths'] != 0:
        raise ValueError('stage2 train inventory differs')
    catalog_path = checked_file(stage2/'UR001-catalog.json',pin['catalog'])
    catalog = pinned_json(catalog_path,pin['catalog']['sha256'])
    if (len(catalog) != pin['counts']['catalog_items']
            or any(type(x) is not int or x <= 0 for x in catalog)
            or any(a >= b for a,b in zip(catalog,catalog[1:]))):
        raise ValueError('fixed full catalog shape/order differs')
    receipt = last_receipt(secure(stage2_run)/'logs/stdout.log')
    if (receipt.get('verified') is not True or receipt.get('manifest_sha256') != pin['stage2_manifest_sha256']
            or receipt.get('counts') != pin['counts'] or receipt.get('test_query_constructed') is not False
            or receipt.get('science_training') is not False):
        raise ValueError('original stage2 verified stdout missing')
    run = json.loads(secure(Path(stage2_run)/'run.json').read_bytes())
    argv = run.get('argv',[])
    if (len(argv) < 5 or argv[1:5] != ['-B','-m','analysis.ur001_derive_cli','derive']
            or '--profile' not in argv or argv[argv.index('--profile')+1] != 'full'
            or '--output' not in argv or argv[argv.index('--output')+1] != str(stage2)):
        raise ValueError('original stage2 formal argv differs')
    proof = producer(stage2_run, argv, pin['stage2_source'], receipt,historical=pin['stage2_job'])
    return catalog, {'stage2_manifest_sha256':pin['stage2_manifest_sha256'],
                     'stage2_input_sha256':pin['stage2_input_sha256'], 'stage2_job':proof}


def load_sequences(paths, records):
    """Original observed-event semantics, explicit row-group/ordinal/count guards."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    if len(paths) != len(records): raise ValueError('one record per train week required')
    histories = {}; catalog = set()
    for week,(path,record) in enumerate(zip(paths,records)):
        raw = snapshot(path,record)
        parquet = pq.ParquetFile(pa.BufferReader(raw))
        if (parquet.metadata.num_rows != record['rows']
                or [parquet.metadata.row_group(i).num_rows for i in range(parquet.metadata.num_row_groups)] != record['row_groups']):
            raise ValueError('physical row-group/count differs')
        for key in ('user_id','item_id'):
            if parquet.schema_arrow.field(key).type != pa.uint32():
                raise ValueError('raw uint32 schema required')
        ordinal = 0
        for batch in parquet.iter_batches(batch_size=65536,columns=['user_id','item_id']):
            for row in batch.to_pylist():
                user,item = row['user_id'],row['item_id']
                if any(type(x) is not int or not 0 <= x < 2**32 for x in (user,item)):
                    raise ValueError('raw uint32 IDs required')
                histories.setdefault(user,[]).append(item); catalog.add(item)
                ordinal += 1
        if ordinal != record['rows']: raise ValueError('file ordinal/count continuity differs')
        progress('full_train_week_read',week=week,rows=ordinal)
    return histories, sorted(catalog)


def input_argv(command, args):
    argv = [sys.executable,'-B','-m','analysis.sgns_gpu_full',command]
    for name in ('data','embedding_root','stage2','stage2_run','lock','output','deployment_sha'):
        argv += ['--'+name.replace('_','-'),str(getattr(args,name))]
    if command == 'prepare' and getattr(args,'source_bridge_request',None) is not None:
        argv += ['--source-bridge-request',str(args.source_bridge_request),'--source-bridge-sha256',args.source_bridge_sha256]
    return argv


def prepare(args, *, readonly):
    import numpy as np
    from vkgr.retrieval_data import content_join
    from vkgr.sid_coll_sgns import _sequence_digest
    from vkgr.sid_tok_audit import feature_digest
    output = secure(args.output)
    if output.exists(): raise FileExistsError('exclusive full output required')
    if ROOT == output or ROOT in output.parents or output in ROOT.parents:
        raise ValueError('output must be outside deployment')
    source = source_identity(); delta = check_source(source,args.deployment_sha)
    env = runtime(args.lock)
    bridge_context = None
    if not readonly and getattr(args,'source_bridge_request',None) is not None:
        from analysis.sgns_full_input_bridge import preflight as bridge_preflight
        bridge_context = bridge_preflight(args,source)
    catalog, binding = stage2_binding(args.stage2,args.stage2_run)
    records = fixed_records(); data = secure(args.data); embedding = secure(args.embedding_root)
    paths = [data/r['path'] for r in records[:25]] + [embedding/records[-1]['path']]
    if set((data/'subsamples/ur0.01/train').glob('*.parquet')) != set(paths[:25]):
        raise ValueError('one exact train file per week required')
    guards(output,projected_memory=96*1024**3,projected_disk=16*1024**3 if not readonly else 0)
    for path,record in zip(paths,records): checked_file(path,record)
    histories,observed_catalog = load_sequences(paths[:25],fixed_dag()['train_records'])
    if observed_catalog != catalog or len(histories) != fixed_dag()['counts']['train_users'] or sum(map(len,histories.values())) != fixed_dag()['counts']['train_rows']:
        raise ValueError('full train count/catalog conservation differs')
    with np.load(io.BytesIO(snapshot(paths[-1],records[-1])),allow_pickle=False) as features:
        content = content_join(catalog,features['item_id'],features['embedding'])
    if content.shape != (len(catalog)+1,64) or content.dtype != np.float32:
        raise ValueError('full original content shape differs')
    buffer = io.BytesIO(); np.save(buffer,content,allow_pickle=False)
    binding.update({'catalog_sha256':fixed_dag()['catalog']['sha256'],
                    'content_feature_sha256':feature_digest(content[1:])})
    for path,record in zip(paths,records): checked_file(path,record)
    if source_identity() != source or runtime(args.lock) != env:
        raise ValueError('source/runtime changed during full preparation')
    receipt = {'schema':'gpu-sgns-full-input-v1','status':'verified' if readonly else 'prepared',
               'verified':readonly,'engineering':False,'training':False,'test_read':False,
               'source':source,'source_delta':delta,'runtime':env,'records':records,'binding':binding,
               'histories':'all observed train events, original physical order, user ascending in fit',
               'train_sequence_sha256':_sequence_digest(histories),'users':len(histories),
               'events':sum(map(len,histories.values())), 'payload_written':not readonly,
               'content_npy_sha256':hashlib.sha256(buffer.getbuffer()).hexdigest()}
    if readonly: return receipt
    if bridge_context is not None:
        from analysis.sgns_full_input_bridge import finish
        receipt['source_bridge'] = finish(bridge_context,receipt)
        receipt['preflight_job'] = receipt['source_bridge']['preflight_job']
    else:
        prior = last_receipt(secure(args.preflight_run)/'logs/stdout.log')
        expected = {**receipt,'status':'verified','verified':True,'payload_written':False}
        if prior != expected: raise ValueError('full read-only preflight input differs')
        prior_argv = input_argv('preflight',args)
        receipt['preflight_job'] = producer(args.preflight_run,prior_argv,source,prior,
            expected_job=full_job_contract(args.preflight_run,prior_argv,source))
    receipt['prepare_argv'] = input_argv('prepare',args) + ['--preflight-run',str(args.preflight_run)]
    output.mkdir(exist_ok=False)
    with (output/'catalog.json').open('xb') as f: f.write(encoded(catalog))
    with (output/'content.npy').open('xb') as f: f.write(buffer.getbuffer())
    with (output/'histories.jsonl').open('xb') as f:
        for user in sorted(histories): f.write(encoded([user,histories[user]]))
    receipt['artifacts'] = {n:digest(output/n) for n in ('catalog.json','content.npy','histories.jsonl')}
    with (output/'manifest.json').open('xb') as f: f.write(encoded(receipt))
    return receipt


def read_bundle(bundle, manifest_sha, source, env):
    import numpy as np
    from vkgr.sid_coll_sgns import _sequence_digest
    from vkgr.sgns_full_verify import check_full_binding
    bundle = secure(bundle); manifest = secure(bundle/'manifest.json')
    if digest(manifest) != manifest_sha: raise ValueError('full input manifest drift')
    value = pinned_json(manifest,manifest_sha)
    if (value.get('schema') != 'gpu-sgns-full-input-v1' or value.get('status') != 'prepared'
            or value.get('engineering') is not False or value.get('training') is not False
            or value.get('test_read') is not False or value.get('source') != source
            or value.get('runtime') != env or value.get('records') != fixed_records()
            or not value.get('preflight_job')):
        raise ValueError('formal full input producer differs')
    if 'source_bridge' in value:
        from analysis.sgns_full_input_bridge import verify_manifest
        verify_manifest(value,source)
    elif len(value.get('prepare_argv',[])) == 25:
        raise ValueError('bridged prepare proof missing')
    names = ('catalog.json','content.npy','histories.jsonl')
    if set(value['artifacts']) != set(names): raise ValueError('full artifact set differs')
    for name in names:
        if digest(secure(bundle/name)) != value['artifacts'][name]: raise ValueError('full payload drift')
    catalog_raw = secure(bundle/'catalog.json').read_bytes()
    if hashlib.sha256(catalog_raw).hexdigest() != fixed_dag()['catalog']['sha256']:
        raise ValueError('consumed catalog drift')
    catalog = json.loads(catalog_raw)
    content_raw = secure(bundle/'content.npy').read_bytes()
    if hashlib.sha256(content_raw).hexdigest() != value['content_npy_sha256']:
        raise ValueError('consumed content drift')
    content = np.load(io.BytesIO(content_raw),allow_pickle=False)
    if content.shape != (len(catalog)+1,64) or content.dtype != np.float32:
        raise ValueError('formal full content shape differs')
    check_full_binding(catalog,content[1:],value['binding'],False)
    histories = {}; previous = -1; consumed = hashlib.sha256()
    with secure(bundle/'histories.jsonl').open('rb') as f:
        for line in f:
            consumed.update(line); user,events = json.loads(line)
            if type(user) is not int or not previous < user < 2**32 or not isinstance(events,list):
                raise ValueError('unique ascending full users required')
            histories[user] = events; previous = user
    pin = fixed_dag()['counts']
    if (consumed.hexdigest() != value['artifacts']['histories.jsonl']
            or _sequence_digest(histories) != value['train_sequence_sha256']
            or len(histories) != value['users'] or value['users'] != pin['train_users']
            or sum(map(len,histories.values())) != value['events'] or value['events'] != pin['train_rows']):
        raise ValueError('consumed full observed sequence/count drift')
    return (catalog,histories,content[1:]),value


def require_acceptance(path, *, sha256=None, scratch=None):
    from analysis.sgns_subset_acceptance import code_pins, consume
    code_pins()
    if path is None or sha256 is None or scratch is None:
        raise ValueError('four-gate total receipt SHA and exclusive replay scratch required')
    return consume(path,sha256,scratch)


def fit(args):
    acceptance = require_acceptance(args.acceptance,sha256=args.acceptance_sha256,scratch=args.acceptance_scratch)
    import numpy as np
    import torch
    from vkgr.sid_coll_sgns import fit_sgns, save_sgns_output
    from vkgr.sgns_full_verify import verify_full_audit
    source = source_identity(); check_source(source,args.deployment_sha); env = runtime(args.lock)
    inputs, manifest = read_bundle(args.bundle,args.bundle_sha,source,env)
    prepare_job = producer(args.predecessor_run,manifest['prepare_argv'],source,manifest,
        expected_job=full_job_contract(args.predecessor_run,manifest['prepare_argv'],source))
    output = secure(args.output)
    if output.exists() or ROOT == output or ROOT in output.parents or output in ROOT.parents:
        raise ValueError('exclusive full fit output outside deployment required')
    guards(output,projected_memory=96*1024**3)
    started = time.monotonic()
    vectors,audit = fit_sgns(*inputs,device='cuda',engineering=False)
    read_bundle(args.bundle,args.bundle_sha,source,env)
    save_sgns_output(output,vectors,audit)
    raw = {n:secure(output/n).read_bytes() for n in ('vectors.npy','audit.json')}
    if json.loads(raw['audit.json']) != audit: raise ValueError('persisted audit differs')
    proof = verify_full_audit(*inputs,np.load(io.BytesIO(raw['vectors.npy']),allow_pickle=False),audit,
                              binding=manifest['binding'],engineering=False)
    if source_identity() != source or runtime(args.lock) != env: raise ValueError('full fit source/runtime drift')
    artifacts = {n:hashlib.sha256(b).hexdigest() for n,b in raw.items()}
    if any(digest(output/n) != sha for n,sha in artifacts.items()): raise ValueError('persisted full output drift')
    receipt = {'schema':'gpu-sgns-full-stage-v1','status':'verified','verified':True,'engineering':False,
               'authority':'ad1c428','device':'cuda','seed':42,'source':source,'runtime':env,
               'input_manifest_sha256':args.bundle_sha,'input_producer':manifest,'prepare_job':prepare_job,'acceptance':acceptance,
               'run_id':os.environ.get('VKGR_RUN_ID'),'cuda_version':torch.version.cuda,
               'cublas_workspace_config':os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
               'seconds':time.monotonic()-started,'artifacts':artifacts,'original_audit_verification':proof,
               'test_read':False,'validation_read':False}
    with (output/'stage-receipt.json').open('xb') as f: f.write(encoded(receipt))
    return receipt


def job(args):
    from analysis.a40_ops import load_ops
    _,launcher,_,_ = load_ops(args.ops_code)
    argv = input_argv(args.stage,args)
    if args.stage == 'prepare':
        argv += ['--preflight-run',str(args.preflight_run)]
    flags = ['--root',str(args.ops_root),'--run-id',args.run_id,'--kind','ops','--cwd',str(ROOT),
             '--config',str(ROOT/'configs/full-gpu-sgns/fixed-dag.json'),'--seed','42',
             '--input',str(args.lock),'--input',str(args.stage2/'UR001-manifest.json'),
             '--env','PYTHONPATH='+str(ROOT/'src'),'--env','OMP_NUM_THREADS=1',
             '--env','OPENBLAS_NUM_THREADS=1','--env','MKL_NUM_THREADS=1',
             '--env','CUBLAS_WORKSPACE_CONFIG=:4096:8','--cpu-cores','1',
             '--ram-mib','196608' if args.stage == 'prepare' else '262144','--vram-mib','0','--disk-gib','16','--wall-hours','24']
    if args.stage == 'prepare':
        for name in ('run.json','terminal.json','logs/stdout.log'):
            flags += ['--input',str(args.preflight_run/name)]
    if args.stage == 'prepare' and getattr(args,'source_bridge_request',None) is not None:
        from analysis.sgns_full_input_bridge import bridge_inputs
        for record in bridge_inputs(args.source_bridge_request,args.source_bridge_sha256):
            flags += ['--input',record['path']]
    spec = launcher.spec_from_args(launcher.build_parser().parse_args([*flags,'--',*argv]))
    metadata = launcher.prepare(spec)
    return {'status':'prepared_dry','spec':spec,'metadata':metadata,'training_enabled':False,
            'resources':'not_measured conservative first declaration; subsequent jobs require measured peaks'}


def common(parser):
    for name in ('data','embedding-root','stage2','stage2-run','lock','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--deployment-sha',required=True)


def bridge_flags(parser):
    parser.add_argument('--source-bridge-request',type=Path)
    parser.add_argument('--source-bridge-sha256')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='command',required=True)
    sub.add_parser('dry')
    for command in ('preflight','prepare'):
        p = sub.add_parser(command); common(p)
        if command == 'prepare':
            p.add_argument('--preflight-run',type=Path,required=True)
            bridge_flags(p)
    p = sub.add_parser('queue-dry'); common(p)
    p.add_argument('--stage',choices=('preflight','prepare'),required=True)
    for name in ('ops-code','ops-root'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--run-id',required=True); p.add_argument('--preflight-run',type=Path)
    bridge_flags(p)
    p = sub.add_parser('fit')
    for name in ('bundle','lock','output','acceptance','predecessor-run'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--acceptance-sha256',required=True); p.add_argument('--acceptance-scratch',type=Path,required=True)
    p.add_argument('--bundle-sha',required=True); p.add_argument('--deployment-sha',required=True)
    args = parser.parse_args(argv)
    if args.command in ('prepare','queue-dry'):
        if bool(args.source_bridge_request) != bool(args.source_bridge_sha256): parser.error('paired source bridge request/SHA required')
        if args.command == 'queue-dry' and args.stage != 'prepare' and args.source_bridge_request is not None:
            parser.error('source bridge only applies to prepare')
    if args.command == 'dry': value = {'status':'ready','train_files':len(fixed_records())-1,'training_enabled':False,'payload_read':False}
    elif args.command == 'fit': value = fit(args)
    elif args.command == 'queue-dry':
        if args.stage == 'prepare' and args.preflight_run is None: parser.error('formal preflight run required')
        value = job(args)
    else: value = prepare(args,readonly=args.command=='preflight')
    print(json.dumps(value,sort_keys=True),flush=True); return 0


if __name__ == '__main__': raise SystemExit(main())
