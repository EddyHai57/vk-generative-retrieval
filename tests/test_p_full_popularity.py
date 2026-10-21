"""Independent Popularity parity and lightweight data projection tests."""
import importlib,io,json,os,subprocess,sys,tarfile
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
@pytest.fixture(scope='module')
def native(tmp_path_factory):
    root=tmp_path_factory.mktemp('pop-native')
    raw=subprocess.check_output(['git','-C',os.environ.get('P_FULL_TEST_GIT_ROOT',str(ROOT)),'archive','0ca61bc','src'])
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:archive.extractall(root,filter='data')
    from analysis.throughput_step_harness import _package
    return _package('p_popularity_test_native',root)('retrieval_seqm1_run')

def test_native_like_counts_ties_full_oov_denominator(native):
    from analysis import p_full_popularity as p
    baseline=importlib.import_module(native.__package__+'.baselines')
    runner=importlib.import_module(native.__package__+'.baseline_run')
    catalog=[1,2,3,4];events=[(3,True),(3,True),(1,True),(2,True),(2,False),(4,False)]
    queries=[{'user_id':n,'history':[1],'target':item,'oov':item==99} for n,item in enumerate([3,3,2,99])]
    got=p.evaluate(native,catalog,queries,{1,3},iter(events))
    counts=baseline.popularity(catalog,iter(events));np.testing.assert_array_equal(counts,[1,1,2,0])
    # Compare the actual original whole baseline runner as an independent rank reference.
    old=runner.evaluate_inputs({'catalog':catalog,'features':np.ones((4,2),dtype=np.float32),'popularity':counts,'queries':{'validation':queries,'test':queries}},dict(runner.CONFIG))[1]['validation']
    assert old['popularity_predictions']==[[3,1,2,4]]*4
    rank=np.asarray(catalog)[np.lexsort((np.asarray(catalog),-counts))[:min(50,len(catalog))]].tolist()
    expected=[list(rank) for _ in queries]
    assert got['predictions']==expected==[[3,1,2,4]]*4
    assert got['metrics']==runner.evaluate_predictions(expected,queries,catalog,(10,20,50))
    assert got['metrics']['all']['count']==4 and got['metrics']['oov']['count']==1
    assert got['metrics']['all']['recall@50']==3/4
    assert got['support_metrics']['supported']['count']==2
    assert got['support_metrics']['unsupported']['count']==1
    assert got['support_metrics']['oov']['count']==1

def test_stream_projects_two_columns_and_keeps_duplicate_likes(tmp_path):
    import pyarrow as pa,pyarrow.parquet as pq
    from analysis import p_full_popularity as p
    path=tmp_path/'train.parquet';pq.write_table(pa.table({'item_id':[3,3,2],'like':[1,1,0],'unused':['a','b','c']}),path)
    seen=[]
    class Reader:
        def __init__(self,path):self.path=path
        def iter_batches(self,**kwargs):
            seen.append(kwargs)
            yield from pq.ParquetFile(self.path).iter_batches(**kwargs)
    assert list(p.train_events([str(path)],parquet=Reader))==[(3,True),(3,True),(2,False)]
    assert seen==[{'batch_size':65536,'columns':['item_id','like'],'use_threads':False}]

def test_dry_illegal_and_holdout_before_request(tmp_path):
    from analysis.ur001_science_entry import A_ROOT,FORBIDDEN_ROOTS,SCHEMA
    inputs=tmp_path/'inputs.json';inputs.write_text(json.dumps({'schema':SCHEMA,'train':[str(A_ROOT/f'subsamples/ur0.01/train/week_{w:02}.parquet') for w in range(25)],'validation':str(FORBIDDEN_ROOTS[2]/'week_26.parquet'),'metadata':[str(A_ROOT/'metadata'/n) for n in ('users_metadata.parquet','items_metadata.parquet','item_embeddings.npz')]}))
    prefix=[sys.executable,'-B','-m','analysis.p_full_popularity'];env={**os.environ,'PYTHONPATH':str(ROOT/'src')+os.pathsep+str(ROOT)}
    for args,code in [(['dry'],0),(['illegal'],2),(['preflight','--inputs',str(inputs),'--request',str(tmp_path/'ABSENT'),'--request-sha256','0'*64],2)]:
        r=subprocess.run(prefix+args,cwd=ROOT,env=env,capture_output=True,text=True)
        assert r.returncode==code,r.stderr
        if args[0]=='dry':assert json.loads(r.stdout)['training'] is False
        if args[0]=='preflight':assert json.loads(r.stdout)['status']=='HOLDOUT_INPUT_REJECTED' and json.loads(r.stdout)['data_opened'] is False

from test_p_full_backend import own_job_fixture

def test_own_job_and_plan_bind_new_entry_without_training(own_job_fixture,monkeypatch):
    from analysis import p_full_popularity as p
    backend,value,records,request,digest,run=own_job_fixture
    value['run_ids']={k:'p-full-popularity-'+k+'-fixture' for k in value['run_ids']}
    expected=p.job_contract(value,records,request,digest,'evaluate')
    assert expected['argv'][3:5]==['analysis.p_full_popularity','evaluate']
    assert expected['run_id']==value['run_ids']['train']
    assert '--execute-approved-science' in expected['argv']
    path=Path(value['ops_root'])/'runs'/expected['run_id'];path.mkdir(parents=True)
    (path/'run.json').write_text(json.dumps(expected))
    for k,v in expected['env'].items():monkeypatch.setenv(k,v)
    assert p.check_job(value,records,request,digest,'evaluate')==backend.file_sha(path/'run.json')
    for key,wrong in [('seed',42.),('argv',['wrong']),('env',{}),('inputs',list(reversed(records)))]:
        (path/'run.json').write_text(json.dumps({**expected,key:wrong}))
        with pytest.raises(ValueError):p.check_job(value,records,request,digest,'evaluate')

def test_bad_own_job_blocks_light_context(own_job_fixture,monkeypatch):
    from analysis import p_full_popularity as p
    backend,value,records,request,digest,run=own_job_fixture
    monkeypatch.setattr(p,'transport',lambda *a:value)
    monkeypatch.setattr(p,'input_records',lambda *a:records)
    def rejected(*a):raise ValueError('own job rejected')
    monkeypatch.setattr(p,'check_job',rejected)
    monkeypatch.setattr(p,'context',lambda *a:pytest.fail('context parsed before job'))
    with pytest.raises(ValueError,match='own job'):p.execute('preflight',records[1]['path'],request,digest)

def test_light_context_never_parses_train_or_embedding(tmp_path,native,monkeypatch):
    from analysis import p_full_popularity as p,new_instance_prepare,ur001_derive_cli,producer_dag_binding
    root=tmp_path/'stage2';root.mkdir();catalog=[1,2]
    query={'user_id':1,'history':[1],'history_positions':[[0,0]],'target_position':[25,0],'target':99,'oov':True}
    for name,val in [('catalog',catalog),('supported',[1]),('validation',query)]:
        (root/('UR001-'+name+('.jsonl' if name=='validation' else '.json'))).write_text(json.dumps(val)+'\n')
    artifacts={}
    for name in ('UR001-catalog.json','UR001-supported.json','UR001-validation.jsonl'):
        path=root/name;artifacts[name]={'sha256':__import__('hashlib').sha256(path.read_bytes()).hexdigest(),'size':path.stat().st_size}
    manifest={'counts':{'catalog_items':2,'eligible':1},'artifacts':artifacts};(root/'UR001-manifest.json').write_text(json.dumps(manifest))
    lock=tmp_path/'lock';lock.write_text('lock');inputs=tmp_path/'inputs';inputs.write_text(json.dumps({'train':['raw-week0']}))
    source={'commit':'a'*40,'files':{'src/example.py':'b'*64}};runtime={'runtime':{'python':'3.11.16'},'lock_sha256':p.backend.file_sha(lock)}
    value={'source':source,'runtime':runtime,'lock':{'path':str(lock),'sha256':p.backend.file_sha(lock)},'stage2':str(root),'stage2_run':str(tmp_path/'producer'),'producer_pin':{},'native_root':str(tmp_path/'native'),'config':native.default_config(),'output_root':str(tmp_path/'output'),'disk_cap_bytes':1024**3}
    monkeypatch.setattr(ur001_derive_cli,'source_identity',lambda:source)
    monkeypatch.setattr(new_instance_prepare,'runtime',lambda *a:runtime)
    monkeypatch.setattr(producer_dag_binding,'verify_producer',lambda *a:{'verified':True})
    monkeypatch.setattr(ur001_derive_cli,'verify_dag',lambda *a,**kw:{'counts':{'catalog_items':2,'eligible':1},'manifest_sha256':p.backend.file_sha(root/'UR001-manifest.json')})
    monkeypatch.setattr(p.backend,'native_for',lambda *a:native)
    monkeypatch.setattr(p.backend,'close_context',lambda *a:None)
    monkeypatch.setattr(native.b1,'resource_guard',lambda *a:lambda:None)
    got=p.context(value,inputs)
    assert got[1:5]==(catalog,[query],{1},['raw-week0'])
    value['disk_cap_bytes']=True
    with pytest.raises(ValueError,match='disk cap'):p.context(value,inputs)
    value['disk_cap_bytes']=1024**3
    monkeypatch.setattr(ur001_derive_cli,'source_identity',lambda:{**source,'commit':'c'*40})
    with pytest.raises(ValueError,match='consumer exact'):p.context(value,inputs)

def test_readonly_pf_and_verify_replay_refuse_changes(tmp_path,native,monkeypatch):
    from analysis import p_full_popularity as p
    catalog=[1,2];queries=[{'user_id':1,'history':[1],'target':1,'oov':False}]
    binding={'ur001':{'counts':{'catalog_items':2,'eligible':1,'train_rows':3,'train_like_events':2}}}
    value={'output_root':str(tmp_path/'output'),'native_root':str(tmp_path/'native'),'stage2':str(tmp_path/'stage2'),'ops_root':str(tmp_path/'ops'),'source':{'commit':'a'*40},'runtime':{},'run_ids':{'preflight':'p-full-popularity-pf','train':'p-full-popularity-evaluate','verify':'p-full-popularity-verify'}}
    monkeypatch.setattr(p,'transport',lambda *a:value);monkeypatch.setattr(p,'input_records',lambda *a:[])
    monkeypatch.setattr(p,'check_job',lambda *a,**kw:'b'*64)
    monkeypatch.setattr(p,'context',lambda *a:(native,catalog,queries,{1},['raw'],binding,lambda:None))
    monkeypatch.setattr(p.backend,'close_context',lambda *a:None)
    monkeypatch.setattr(p,'train_events',lambda *a:iter([(1,True),(1,True),(2,False)]))
    receipts={}
    def accepted(*args):return receipts[args[4]]
    monkeypatch.setattr(p,'accepted',accepted)
    original=p.evaluate
    monkeypatch.setattr(p,'evaluate',lambda *a:pytest.fail('PF computed metrics'))
    receipts['preflight']=p.execute('preflight',tmp_path/'inputs',tmp_path/'request','c'*64)
    assert receipts['preflight']['training'] is False and receipts['preflight']['metrics_computed'] is False
    monkeypatch.setattr(p,'evaluate',original)
    receipts['evaluate']=p.execute('evaluate',tmp_path/'inputs',tmp_path/'request','c'*64)
    assert receipts['evaluate']['training'] is False
    assert p.execute('verify',tmp_path/'inputs',tmp_path/'request','c'*64)['scientific_verified'] is True
    # Fresh output phase with retained evidence: prediction corruption must fail independently.
    value['output_root']=str(tmp_path/'tampered');Path(value['output_root']).mkdir()
    (Path(value['output_root'])/'popularity-predictions.json').write_text('[[2,1]]')
    (Path(value['output_root'])/'popularity-metrics.json').write_text(json.dumps({k:v for k,v in original(native,catalog,queries,{1},iter([(1,True),(1,True),(2,False)])).items() if k!='predictions'}))
    with pytest.raises(ValueError,match='replay'):p.execute('verify',tmp_path/'inputs',tmp_path/'request','c'*64)

    binding['ur001']['counts']['train_like_events']=1
    value['output_root']=str(tmp_path/'bad-counts')
    with pytest.raises(ValueError,match='conservation'):p.execute('preflight',tmp_path/'inputs',tmp_path/'request','c'*64)


def test_popularity_cli_plan_real_launcher_prepare(tmp_path):
    """Actual backend plan and frozen launcher; only fixture admission roots/pins relocate."""
    import hashlib
    deployment = tmp_path / 'deployed'; deployment.mkdir()
    ops_source = tmp_path / 'ops-source'; ops_source.mkdir()
    files = ['analysis/ur001_science_entry.py', 'analysis/ur001_prepare.py', 'analysis/ur001_derive_cli.py',
             'analysis/profile_anchor_steps.py', 'configs/ur001-preparation/hf-files.json']
    for target, args in ((deployment, ['HEAD', *files]), (ops_source, ['5840e6b', 'ops'])):
        raw = subprocess.check_output(['git', '-C', os.environ.get('P_FULL_TEST_GIT_ROOT', str(ROOT)), 'archive', *args])
        with tarfile.open(fileobj=io.BytesIO(raw)) as archive: archive.extractall(target, filter='data')
    (deployment / 'analysis/p_full_training_backend.py').write_bytes((ROOT / 'analysis/p_full_training_backend.py').read_bytes())
    (deployment / 'analysis/p_full_popularity.py').write_bytes((ROOT / 'analysis/p_full_popularity.py').read_bytes())
    data = tmp_path / 'data'; data.mkdir()
    def put(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = json.dumps(value, sort_keys=True).encode() if not isinstance(value, bytes) else value
        path.write_bytes(raw)
        return {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(), 'size': len(raw)}
    # Relocation belongs only to this clean synthetic deployment fixture.
    admission = deployment / 'analysis/ur001_science_entry.py'
    admission.write_text(admission.read_text().replace('/workspace/vkgr-ur001-data-20261020-v1', str(data)))
    inventory_path = deployment / 'configs/ur001-preparation/hf-files.json'
    inventory = json.loads(inventory_path.read_bytes())
    for rec in inventory['files']:
        fixed = put(data / rec['path'], b'synthetic payload\n')
        rec.update(size=fixed['size'], sha256=fixed['sha256'])
    inventory_path.write_text(json.dumps(inventory))
    derive = deployment / 'analysis/ur001_derive_cli.py'
    derive.write_text(derive.read_text().replace('19cd71f02aff4fce93349907dbc5fcdbf006afc679ee899d9c98965123270fa0', hashlib.sha256(inventory_path.read_bytes()).hexdigest()))
    native = tmp_path / 'native'
    native_file = put(native / 'src/vkgr/__init__.py', b'')
    p3c = put(native / 'docs/p3c_validation.md', b'synthetic data-channel fixture\n')
    pins = {'schema': 'anchor-profile-pins-v1', 'g': {}, 'p': {
        'source_files': {'src/vkgr/__init__.py': native_file['sha256']}, 'p3c_data_sha256': p3c['sha256']}}
    pins_file = put(deployment / 'configs/anchor-profile/source-pins.json', pins)
    profile = deployment / 'analysis/profile_anchor_steps.py'
    profile.write_text(profile.read_text().replace('1227ff9a3b6dbe7cb6f6e2b05be154a12e3a6d224bf980d4df33380b05e4747b', pins_file['sha256']))
    stage2 = tmp_path / 'stage2'
    physical = put(stage2 / 'UR001-input-manifest.json', {
        'profile': 'full', 'inventory_sha256': hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
        'data_root': str(data / 'subsamples/ur0.01'),
        'files': [{**r, 'path': r['path'].removeprefix('subsamples/ur0.01/')} for r in inventory['files'][:26]]})
    artifacts = {}
    for name in ('train.jsonl', 'validation.jsonl', 'catalog.json', 'supported.json', 'S0.json', 'counts.json'):
        rec = put(stage2 / ('UR001-' + name), b'{}\n')
        artifacts['UR001-' + name] = {'sha256': rec['sha256'], 'size': rec['size']}
    manifest = put(stage2 / 'UR001-manifest.json', {'upstream': {'UR001-input-manifest.json': physical['sha256']}, 'artifacts': artifacts})
    producer = tmp_path / 'producer'; producer_pins = {}
    for name in ('run.json', 'terminal.json', 'logs/stdout.log'):
        producer_pins[name] = put(producer / name, {})['sha256']
    inputs = tmp_path / 'inputs.json'
    admitted = {'schema': 'ur001-science-inputs-v1', 'train': [str(data / r['path']) for r in inventory['files'][:25]],
                'validation': str(data / inventory['files'][25]['path']),
                'metadata': [str(data / 'metadata' / n) for n in ('users_metadata.parquet', 'items_metadata.parquet', 'item_embeddings.npz')]}
    inp = put(inputs, admitted); lock = put(tmp_path / 'lock', b'synthetic runtime lock')
    subprocess.run(['git', 'init', '-q', str(deployment)], check=True)
    subprocess.run(['git', '-C', str(deployment), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(deployment), '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'transport fixture'], check=True)
    source = subprocess.check_output(['git', '-C', str(deployment), 'rev-parse', 'HEAD'], text=True).strip()
    fixed = {r['path']: r for r in inventory['files']}
    env = {'PYTHONPATH': str(deployment / 'src'), 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
           'MKL_NUM_THREADS': '1', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}
    contract = {'env': env, 'resources': {'cpu_cores': 1, 'ram_mib': 98304, 'vram_mib': 32768, 'disk_gib': 4, 'wall_seconds': 7200.0},
                'ops_source': {'commit': '5840e6b686ec9651314a9e28a1a2c0befcc05158', 'sha256': 'b' * 64},
                'overlay': None, 'pinned_files': {}, 'stop_mode': 'signal'}
    value = {'schema': 'p-full-backend-request-v1', 'source': {'commit': source}, 'runtime': {},
             'lock': {k: lock[k] for k in ('path', 'sha256')}, 'native_root': str(native), 'stage2': str(stage2),
             'stage2_run': str(producer), 'producer_pin': {'manifest_sha256': manifest['sha256'], 'job_sha256': producer_pins},
             'items_metadata': {'path': str(data / 'metadata/items_metadata.parquet'), 'sha256': fixed['metadata/items_metadata.parquet']['sha256']},
             'embedding': {'path': str(data / 'metadata/item_embeddings.npz'), 'sha256': fixed['metadata/item_embeddings.npz']['sha256']},
             'config': {'seed': 42}, 'ops_root': str(tmp_path / 'ops-root'),
             'run_ids': {'preflight': 'p-full-popularity-pf-fixture', 'train': 'p-full-popularity-evaluate-fixture', 'verify': 'p-full-popularity-verify-fixture'},
             'output_root': str(tmp_path / 'output'), 'admission_sha256': inp['sha256'], 'predecessor': None,
             'disk_cap_bytes': 1024**3, 'job_contract': {stage: contract for stage in ('preflight', 'train', 'verify')}}
    request = tmp_path / 'request.json'; req = put(request, value)
    cli = [sys.executable, '-B', '-m', 'analysis.p_full_popularity', 'plan', '--stage', 'preflight',
           '--inputs', str(inputs), '--request', str(request), '--request-sha256', req['sha256']]
    planned = subprocess.run(cli, cwd=deployment, env={**os.environ, 'PYTHONPATH': str(deployment)}, capture_output=True, text=True)
    assert planned.returncode == 0, planned.stdout + planned.stderr
    plan = json.loads(planned.stdout)
    assert plan['status'] == 'planned_dry' and len(plan['inputs']) > 30
    assert plan['training'] is False
    assert plan['expected_job']['argv'][3:5] == ['analysis.p_full_popularity','preflight']
    assert len({r['path'] for r in plan['inputs']}) == len(plan['inputs'])
    assert any(r['path'] == str(native / 'docs/p3c_validation.md') for r in plan['inputs'])
    script = '''import json,sys
from vkgr_ops import launcher
plan=json.loads(sys.argv[1]);args=launcher.build_parser().parse_args(plan['launcher_argv'])
metadata=launcher.prepare(launcher.spec_from_args(args))
for key in ('git','config','inputs','cwd'):
    assert metadata[key] == plan['expected_job'][key], key
assert metadata['env'] == {k:v for k,v in plan['expected_job']['env'].items() if k not in ('VKGR_RUN_ID','VKGR_PROGRESS_PATH')}
print(json.dumps({'status':'prepared_dry','inputs':len(metadata['inputs'])}))
'''
    prepared = subprocess.run([sys.executable, '-B', '-c', script, planned.stdout],
                              env={**os.environ, 'PYTHONPATH': str(ops_source / 'ops')}, capture_output=True, text=True)
    assert prepared.returncode == 0, prepared.stderr
    assert json.loads(prepared.stdout)['inputs'] == len(plan['inputs'])
    assert not Path(value['ops_root']).exists()


@pytest.mark.parametrize('attack',['none','float_event_count','boolean_exit','wall_cap','stdout','binding','job_env','pf_metrics'])
def test_original_pf_receipt_job_stdout_and_types_closed(own_job_fixture,monkeypatch,attack):
    import copy
    from analysis import p_full_popularity as p
    backend,value,records,request,digest,_=own_job_fixture
    value['run_ids']={k:'p-full-popularity-'+k+'-receipt-fixture' for k in value['run_ids']}
    value['runtime']={};value['output_root']=str(request.parent/'output')
    expected=p.job_contract(value,records,request,digest,'preflight');run=p.run_path(value,'preflight');(run/'logs').mkdir(parents=True)
    (run/'run.json').write_text(json.dumps(expected))
    counts={'catalog_items':2,'eligible':1,'train_rows':3,'train_like_events':2};binding={'ur001':{'counts':counts}}
    receipt={'schema':'p-full-popularity-receipt-v1','status':'verified','verified':True,'stage':'preflight','training':False,'engineering':False,'test_read':False,'run_id':expected['run_id'],'run_manifest_sha256':backend.file_sha(run/'run.json'),'request_sha256':digest,'source':value['source'],'runtime':value['runtime'],'binding':binding,'validation_queries':1,'catalog_items':2,'scientific_verified':False,'payload_written':False,'metrics_computed':False,'raw_train_events':3,'raw_train_like_events':2}
    terminal={'schema':'vkgr-terminal-v1','run_id':expected['run_id'],'status':'succeeded','exit_code':0,'signal':None,'wall_cap_exceeded':False}
    if attack=='float_event_count':receipt['raw_train_events']=3.
    if attack=='boolean_exit':terminal['exit_code']=False
    if attack=='wall_cap':terminal['wall_cap_exceeded']=True
    if attack=='binding':receipt['binding']=copy.deepcopy(binding);receipt['binding']['ur001']['counts']['eligible']=1.
    if attack=='pf_metrics':receipt['metrics']={'all':{}}
    if attack=='job_env':expected['env']['PYTHONPATH']='/unexpected';(run/'run.json').write_text(json.dumps(expected));receipt['run_manifest_sha256']=backend.file_sha(run/'run.json')
    Path(value['output_root']).mkdir();p.receipt_path(value,'preflight').write_text(json.dumps(receipt));(run/'terminal.json').write_text(json.dumps(terminal))
    (run/'logs/stdout.log').write_text(json.dumps({**receipt,'verified':False} if attack=='stdout' else receipt))
    monkeypatch.setattr(p,'input_records',lambda *a:records)
    if attack=='none':assert p.accepted(value,records[1]['path'],request,digest,'preflight',binding)==receipt
    else:
        with pytest.raises(ValueError):p.accepted(value,records[1]['path'],request,digest,'preflight',binding)


def test_popularity_transport_refuses_m1_job_ids_and_predecessor(monkeypatch):
    from analysis import p_full_popularity as p
    value={'predecessor':None,'run_ids':{'preflight':'p-full-popularity-pf','train':'p-full-popularity-evaluate','verify':'p-full-popularity-verify'}}
    monkeypatch.setattr(p.backend,'read_transport',lambda *a:value)
    assert p.transport(None,None,None)==value
    value['run_ids']['train']='p-full-train-s42'
    with pytest.raises(ValueError):p.transport(None,None,None)
    value['run_ids']['train']='p-full-popularity-evaluate';value['predecessor']={}
    with pytest.raises(ValueError):p.transport(None,None,None)


def test_parsed_projection_is_bound_to_consumed_bytes(tmp_path,monkeypatch):
    import hashlib
    from analysis import p_full_popularity as p
    for name,data in [('UR001-catalog.json',b'[1,2]'),('UR001-supported.json',b'[1]'),('UR001-validation.jsonl',b'{"target":1}\n')]:
        (tmp_path/name).write_bytes(data)
    artifacts={path.name:{'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'size':path.stat().st_size} for path in tmp_path.iterdir()}
    assert p.read_projection(tmp_path,{'artifacts':artifacts})==([1,2],[{'target':1}],[1])
    original=Path.read_bytes
    monkeypatch.setattr(Path,'read_bytes',lambda path:b'[2,1]' if path.name=='UR001-catalog.json' else original(path))
    with pytest.raises(ValueError,match='consumed'):p.read_projection(tmp_path,{'artifacts':artifacts})

def test_rank_top50_boundary_is_not_catalog_or_query_sampling(native):
    from analysis import p_full_popularity as p
    runner=importlib.import_module(native.__package__+'.baseline_run')
    catalog=list(range(1,61));queries=[{'user_id':n,'history':[1],'target':item,'oov':item==61} for n,item in enumerate([49,50,61])]
    events=[(i,False) for i in catalog]+[(60,True),(60,True),(1,True)]
    got=p.evaluate(native,catalog,queries,{1,60},iter(events))
    expected=[[60,*range(1,50)]]*3
    assert got['predictions']==expected
    assert got['metrics']==runner.evaluate_predictions(expected,queries,catalog,(10,20,50))
    assert got['metrics']['all']['recall@50']==1/3
    assert got['metrics']['all']['recall@20']==0.
