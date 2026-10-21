"""Closed consumer facade for original M1-mean evaluation and approved P blocks.

Frozen source modules are never modified. Only complete query-block scheduling
changes; original exports, metric functions, diagnostics and verify code remain.
"""
from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
import tempfile
import time
from types import CodeType, FunctionType, SimpleNamespace

import numpy as np

from vkgr.p_query_blocks import parallel_topk
from vkgr.throughput_config import Engineering, guard

PINS={
    'retrieval_seqm1_run':'e81c9183c0cda549a149ed46ae56efc2bfd11e9c3e90f22ff125353e0de559b7',
    'retrieval_feat02_run':'1e66de6586a183565bc9817ec971247ae140f762646c24411b56950e108984fd',
    'retrieval_experiment':'4b29d02710582a933c2aed8fc908f0113ee70a7111f09da112c6f5a935f7caa7'}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _clone(function,replacements):
    namespace=dict(function.__globals__)
    if not set(replacements)<=set(namespace):raise ValueError('closed original globals required')
    namespace.update(replacements)
    changed={key for key in namespace if namespace[key] is not function.__globals__[key]}
    if changed!=set(replacements):raise ValueError('closed replacement set differs')
    result=FunctionType(function.__code__,namespace,function.__name__,function.__defaults__,function.__closure__)
    result.__kwdefaults__=function.__kwdefaults__
    assert result.__code__ is function.__code__ and result.__closure__ is function.__closure__
    return result


def evaluate_vectors(original,users,items,catalog,queries,support,ks=(10,20,50),check=None,*,temp_root,disk_check=None):
    # Original metrics and statistics; one pool schedules complete original
    # 32-query blocks with the existing 4096-item float64 einsum arithmetic.
    metrics=importlib.import_module(original.__package__+'.baselines').metrics
    started=time.monotonic()
    if check:check()
    root=Path(temp_root)
    if not root.is_absolute() or any(p.is_symlink() for p in (root,*root.parents)):
        raise ValueError('absolute non-symlink parallel evaluation root required')
    projected=sum(np.asarray(v).nbytes+256 for v in (users,items,catalog))
    if disk_check:disk_check(projected)
    root.mkdir(parents=True,exist_ok=True)
    engineering=Engineering(use_mmap=True,eval_workers=8,temp_dir=str(root))
    guard(engineering,root,projected_disk=projected)
    directory=Path(tempfile.mkdtemp(prefix='evaluation-',dir=root))
    maps=[]
    for name,value in (('users',users),('items',items),('ids',np.asarray(catalog))):
        path=directory/(name+'.npy')
        with path.open('xb') as stream:np.save(stream,value,allow_pickle=False)
        maps.append(np.load(path,mmap_mode='r',allow_pickle=False))
    if check:check()
    predicted,_=parallel_topk(*maps,max(ks),engineering)
    if check:check()
    predictions=predicted.tolist()
    result=metrics(predictions,queries,catalog,ks);slices={}
    for name in ('supported','unsupported','oov'):
        positions=[j for j,q in enumerate(queries) if ('oov' if q['oov'] else 'supported' if q['target'] in support else 'unsupported')==name]
        slices[name]=metrics([predictions[j] for j in positions],[queries[j] for j in positions],catalog,ks)['all']
    return {'predictions':predictions,'metrics':result,'support_metrics':slices,'zero_item_vectors':int(np.count_nonzero(np.linalg.norm(items,axis=1)==0)),'zero_user_vectors':int(np.count_nonzero(np.linalg.norm(users,axis=1)==0)),'exact_seconds':time.monotonic()-started}


def _evaluate_features(fr,evaluate_vectors,model,data,queries,config,check=None):
    # Original ten-line feature evaluation tail, with an explicit score callback.
    export_vectors_side2=fr.export_vectors_side2
    f1=fr.f1
    top50_unsupported_share=fr.top50_unsupported_share
    item_norm_stats=fr.item_norm_stats
    users, items, norms = export_vectors_side2(model, data['catalog'], queries, data['idmap'], config['batch_size'], 50 if config['history_len'] <= 50 else config['history_len'], config['feature'], check)
    result = evaluate_vectors(users, items, data['catalog'], queries, data['support'], check=check); result.pop('exact_seconds')
    predictions = result['predictions']
    diagnostics = {'zero_user_vectors': result['zero_user_vectors'], 'zero_user_vector_fraction': result['zero_user_vectors'] / len(queries),
                   'top1_is_last_history_fraction': sum(1 for p, q in zip(predictions, queries) if p and p[0] == q['history'][-1]) / len(queries),
                   'history_support_slices': f1.history_support_slices(predictions, queries, data['support'], data['catalog']),
                   'top50_unsupported_share': top50_unsupported_share(predictions, data['support']),
                   'item_norm_stats': item_norm_stats(norms, data['catalog'], data['support'])}
    return result, diagnostics


def bind(native,temp_root,*,output_root,disk_cap_bytes):
    output_root=Path(output_root);temp_root=Path(temp_root)
    if (not output_root.is_absolute() or not temp_root.is_absolute() or type(disk_cap_bytes) is not int or disk_cap_bytes<=0
            or output_root==temp_root or output_root in temp_root.parents or temp_root in output_root.parents):
        raise ValueError('separate absolute output/cache paths and positive disk cap required')
    def disk_check(projected=0):
        total=projected
        for root in (output_root,temp_root):
            if any(p.is_symlink() for p in (root,*root.parents)):
                raise ValueError('symlink output/cache refused')
            if root.exists():
                for path in root.rglob('*'):
                    if path.is_symlink():raise ValueError('symlink output/cache refused')
                    if path.is_file():total+=path.stat().st_size
        if total>disk_cap_bytes:raise OSError('combined output/cache exceeds run disk cap')
    def resource_guard(output,cap):
        if Path(output)!=output_root or cap!=disk_cap_bytes:raise ValueError('fixed output/cache budget differs')
        original_guard=native.b1.resource_guard(output,cap)
        def combined():
            original_guard();disk_check()
        return combined
    original=importlib.import_module(native.__package__+'.retrieval_experiment')
    modules={'retrieval_seqm1_run':native,'retrieval_feat02_run':native.fr,'retrieval_experiment':original}
    for name,module in modules.items():
        if _sha(module.__file__)!=PINS[name]:raise ValueError('frozen evaluation source ledger differs')
    # Compare live function code with a fresh compilation of pinned source.
    for module,names in ((native,('evaluate_model','ablation_g0','verify_run')),(native.fr,('evaluate_model','export_vectors_side2'))):
        for name in names:
            function=getattr(module,name)
            compiled=compile(Path(module.__file__).read_bytes(),module.__file__,'exec',dont_inherit=True)
            expected=next(c for c in compiled.co_consts if isinstance(c,CodeType) and c.co_name==name)
            if (function.__module__!=module.__name__ or Path(function.__code__.co_filename).resolve()!=Path(module.__file__).resolve()
                    or function.__code__!=expected):
                raise ValueError('original evaluation function identity differs')
    def scoring(*args,**kwargs):return evaluate_vectors(original,*args,**kwargs,temp_root=temp_root,disk_check=disk_check)
    def features(*args,**kwargs):return _evaluate_features(native.fr,scoring,*args,**kwargs)
    fr=SimpleNamespace(**{**vars(native.fr),'evaluate_model':features})
    evaluate=_clone(native.evaluate_model,{'fr':fr})
    ablation=_clone(native.ablation_g0,{'fr':fr})
    verify=_clone(native.verify_run,{'evaluate_model':evaluate,'ablation_g0':ablation})
    b1=SimpleNamespace(**{**vars(native.b1),'resource_guard':resource_guard})
    identity={'source_sha256':_sha(__file__),'native_source_sha256':PINS.copy(),'workers':8,
              'query_block':32,'item_block':4096,'score_dtype':'float64','transport':'full-npy-mmap',
              'temp_root':str(temp_root),'output_root':str(output_root),'disk_cap_bytes':disk_cap_bytes,
              'disk_scope':'combined output plus retained parallel cache','temporary_files_retained':True}
    return SimpleNamespace(**{**vars(native),'evaluate_model':evaluate,'ablation_g0':ablation,
                             'verify_run':verify,'b1':b1,'parallel_eval_identity':identity})
