"""Synthetic-only parity of the explicit full-P parallel evaluation consumer."""
import importlib
import io
import os
from pathlib import Path
import subprocess
import tarfile

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def synthetic_capacity(monkeypatch):
    # This host has fewer than nine cores: simulate admission only for tiny
    # fixtures, while exercising the real eight-process scoring pool.
    from vkgr import p_query_blocks
    monkeypatch.setattr(p_query_blocks,'cpu_capacity',lambda:9)


@pytest.fixture(scope='module')
def native(tmp_path_factory):
    root=tmp_path_factory.mktemp('p-parallel-native')
    raw=subprocess.check_output(['git','-C',os.environ.get('P_FULL_TEST_GIT_ROOT',str(ROOT)),
        'archive','0ca61bc','src','configs/seqm1','.python-version','pyproject.toml','uv.lock'])
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:archive.extractall(root,filter='data')
    from analysis.throughput_step_harness import _package
    return _package('p_parallel_native_test',root)('retrieval_seqm1_run')


def test_multiblock_tail_ties_all_return_fields(native,tmp_path):
    from analysis.p_full_parallel_eval import evaluate_vectors
    from analysis.tensor_cache_parity import freeze
    old=importlib.import_module(native.__package__+'.retrieval_experiment')
    rng=np.random.RandomState(42)
    users=rng.standard_normal((65,8)).astype('float32')
    items=rng.standard_normal((4103,8)).astype('float32')
    users[-1]=0;items[-1]=0;items[1]=items[0]
    catalog=list(range(1,len(items)+1));support=set(catalog[::2])
    queries=[{'target':catalog[i],'oov':False,'history':[catalog[(i+1)%len(items)]]} for i in range(len(users))]
    expected=old.evaluate_vectors(users,items,catalog,queries,support);expected.pop('exact_seconds')
    actual=evaluate_vectors(old,users,items,catalog,queries,support,temp_root=tmp_path/'new')
    actual.pop('exact_seconds')
    assert freeze(actual)==freeze(expected)
    files=list((tmp_path/'new').glob('*/*.npy'))
    assert len(files)==3
    assert all(p.stat().st_size>0 for p in files)


def test_model_diagnostics_and_original_code_identity(native,tmp_path):
    from analysis.p_full_parallel_eval import bind
    from analysis.tensor_cache_parity import freeze
    data=native.fr.synthetic_data();config=native.default_config()
    model=native._model(data,config,'cpu');_,queries=native.fr._view(data,config['history_len'])
    before=dict(native.__dict__);fr_before=dict(native.fr.__dict__)
    consumer=bind(native,tmp_path/'parallel',output_root=tmp_path/'new',disk_cap_bytes=1024**3)
    assert freeze(consumer.evaluate_model(model,data,queries,config))==freeze(native.evaluate_model(model,data,queries,config))
    assert consumer.verify_run.__globals__['evaluate_model'] is consumer.evaluate_model
    assert consumer.verify_run.__globals__['ablation_g0'] is consumer.ablation_g0
    assert consumer.verify_run.__code__ is native.verify_run.__code__
    assert consumer.evaluate_model.__code__ is native.evaluate_model.__code__
    assert consumer.ablation_g0.__code__ is native.ablation_g0.__code__
    assert dict(native.__dict__)==before and dict(native.fr.__dict__)==fr_before
    assert consumer.parallel_eval_identity['workers']==8


def test_training_and_original_verify_share_wrapper_and_failures_propagate(native,tmp_path):
    from analysis.p_full_parallel_eval import bind
    from analysis import p_full_training_backend as backend
    from analysis.tensor_cache_parity import freeze
    data=native.fr.synthetic_data();config={**native.default_config(),'max_epochs':1,'min_epochs':0,'batch_size':32}
    consumer=bind(native,tmp_path/'parallel',output_root=tmp_path/'new',disk_cap_bytes=1024**3)
    old=native.run_session(tmp_path/'old',data,config)
    new=backend.run_session(consumer,tmp_path/'new',data,config,expected_rows=len(data['rows']))
    for key in ('records','evaluations','selection','stats','data_diagnostics','ablation_g0'):
        assert freeze(new[key])==freeze(old[key])
    before_verify=len(list((tmp_path/'parallel').iterdir()))
    assert consumer.verify_run(tmp_path/'new/manifest.json',new['manifest_sha256'],data,config)['status']=='verified'
    assert len(list((tmp_path/'parallel').iterdir()))>before_verify
    with pytest.raises(ValueError):
        consumer.verify_run(tmp_path/'new/manifest.json','0'*64,data,config)
    assert len(list((tmp_path/'parallel').iterdir()))>=2


def test_feature_diagnostic_tail_is_original_ast(native):
    import ast,inspect,textwrap
    from analysis import p_full_parallel_eval as adapter
    def tail(function):
        body=ast.parse(textwrap.dedent(inspect.getsource(function))).body[0].body
        start=next(i for i,node in enumerate(body) if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Tuple))
        return [ast.dump(node,include_attributes=False) for node in body[start:]]
    assert tail(adapter._evaluate_features)==tail(native.fr.evaluate_model)


def test_live_function_replacement_is_rejected(native,tmp_path,monkeypatch):
    from analysis.p_full_parallel_eval import bind
    monkeypatch.setattr(native,'verify_run',lambda *a,**k: {'status':'verified'})
    with pytest.raises(ValueError,match='function identity'):bind(native,tmp_path/'parallel',output_root=tmp_path/'new',disk_cap_bytes=1024**3)


def test_real_capacity_rejection_is_preserved(native,tmp_path,monkeypatch):
    from analysis.p_full_parallel_eval import evaluate_vectors
    from vkgr import p_query_blocks
    original=importlib.import_module(native.__package__+'.retrieval_experiment')
    monkeypatch.setattr(p_query_blocks,'cpu_capacity',lambda:8)
    with pytest.raises(ValueError,match='CPU quota'):
        evaluate_vectors(original,np.zeros((1,2),dtype='float32'),np.zeros((50,2),dtype='float32'),list(range(50)),[{'target':1,'oov':False}],{1},temp_root=tmp_path/'parallel')


def test_metric_tail_is_original_ast(native):
    import ast,inspect,textwrap
    from analysis import p_full_parallel_eval as adapter
    original=importlib.import_module(native.__package__+'.retrieval_experiment')
    def tail(function):
        body=ast.parse(textwrap.dedent(inspect.getsource(function))).body[0].body
        start=next(i for i,node in enumerate(body) if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='result')
        return [ast.dump(node,include_attributes=False) for node in body[start:]]
    assert tail(adapter.evaluate_vectors)==tail(original.evaluate_vectors)


def test_cache_and_output_share_cap_and_files_survive_failure(native,tmp_path,monkeypatch):
    from analysis.p_full_parallel_eval import bind
    output=tmp_path/'output';output.mkdir();cache=tmp_path/'cache';cache.mkdir()
    (output/'checkpoint').write_bytes(b'x'*60)
    (cache/'old-mmap').write_bytes(b'y'*60)
    # The original live resource sampler requires Linux; isolate that boundary.
    calls=[]
    monkeypatch.setattr(native.b1,'resource_guard',lambda *a:lambda:calls.append('original-guard'))
    consumer=bind(native,cache,output_root=output,disk_cap_bytes=100)
    check=consumer.b1.resource_guard(output,100)
    with pytest.raises(OSError,match='combined output/cache'):check()
    assert calls==['original-guard']
    assert (output/'checkpoint').stat().st_size==60 and (cache/'old-mmap').stat().st_size==60


def test_projected_mmap_cap_fails_before_new_directory(native,tmp_path):
    from analysis.p_full_parallel_eval import bind
    output=tmp_path/'output';cache=tmp_path/'cache'
    consumer=bind(native,cache,output_root=output,disk_cap_bytes=100)
    data=native.fr.synthetic_data();config=native.default_config();model=native._model(data,config,'cpu')
    _,queries=native.fr._view(data,config['history_len'])
    with pytest.raises(OSError,match='combined output/cache'):
        consumer.evaluate_model(model,data,queries,config)
    assert not cache.exists() or not list(cache.iterdir())
