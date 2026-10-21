"""Compact input storage equivalence; synthetic only, no submit."""
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pytest
from test_p_full_backend import native


def module():
    return importlib.import_module('analysis.p_full_compact_inputs')


def write_rows(path,rows):
    path.write_text(''.join(json.dumps(r,sort_keys=True,separators=(',',':'))+'\n' for r in rows))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_offset_rows_and_lazy_view_are_exact_and_pinned(tmp_path):
    m=module();rows=[{'example_id':i,'history':[10,11],'history_positions':[[0,0],[0,1]],'target':12,'target_position':[1,2],'user_id':7,'past_likes':list(range(1000)),'side':[[1,2,3],[4,5,6]]} for i in range(3)]
    source=tmp_path/'rows.jsonl';digest=write_rows(source,rows);calls=[]
    compact=m.JsonRows.build(source,tmp_path/'offsets',3,digest,lambda:calls.append(True))
    assert not isinstance(compact,list) and len(compact)==3
    assert list(compact)==rows and compact[-1]==rows[-1] and compact[1:]==rows[1:]
    view=m.CutRows(compact,1)
    expected=[{**r,'history':r['history'][-1:],'history_positions':r['history_positions'][-1:],'side':r['side'][-1:]} for r in rows]
    assert list(view)==expected and compact[0]==rows[0] and calls
    assert m.digest(compact)==hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    assert m.digest(view)==m.digest(expected)
    with pytest.raises(IndexError):compact[3]
    with pytest.raises(ValueError,match='count'):m.JsonRows.build(source,tmp_path/'wrong-count',2,digest,lambda:None)
    with pytest.raises(ValueError,match='hash'):m.JsonRows.build(source,tmp_path/'wrong-hash',3,'0'*64,lambda:None)


@pytest.mark.parametrize('fault',['','missing','conflict','item','field'])
def test_dense_side_uses_original_encode_and_rejects_original_failures(tmp_path,native,fault):
    m=module();side=importlib.import_module(native.__package__+'.history_side')
    history={'train':[{'history':[10,11],'history_positions':[[0,0],[0,1]]}], 'validation':[{'history':[11],'history_positions':[[0,1]]}]}
    event=lambda item:{'item_id':item,'timespent':20,**{key:False for key in side.FEEDBACK_BITS}}
    events=[event(10),event(11),event(99)];durations={10:30,11:40}
    if fault=='missing':events=events[:1]
    if fault=='conflict':history['validation'][0]['history']=[12]
    if fault=='item':events[1]['item_id']=12
    if fault=='field':events[1]['timespent']=None
    if fault:
        with pytest.raises(ValueError):m.DenseSide.build(tmp_path/'side',history,[(0,iter(events))],durations,side._encode,lambda:None)
    else:
        expected=side.build_side([(0,events)],history,durations)
        stored=m.DenseSide.build(tmp_path/'side',history,[(0,iter(events))],durations,side._encode,lambda:None)
        assert {split:[stored.lookup(row) for row in rows] for split,rows in history.items()}==expected
        assert all(type(v)is int for event in stored.lookup(history['train'][0]) for v in event)
        assert sum(p.stat().st_size for p in (tmp_path/'side').iterdir())>0


def test_stream_digest_matches_original_canonical_bytes():
    m=module();values=[{'unicode':'中文/é','float':1.0,'a':[None,False,True,-1,0.25]},[[],{},[1,2]], {'quoted':'a\\b\n"','large':2**63}]
    for value in values:
        raw=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        assert b''.join(m.canonical_chunks(value))==raw
        assert m.digest(value)==hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError):m.digest(float('nan'))


def test_compact_facade_all_fields_binding_three_steps_and_evaluation(tmp_path,native):
    m=module();from analysis.p_full_engineering import capture
    original=native.fr.synthetic_data();data={**original};compacted={**data}
    for split,key in [('train','rows'),('validation','queries')]:
        source=tmp_path/(split+'.jsonl');digest=write_rows(source,data[key])
        compacted[key]=m.JsonRows.build(source,tmp_path/(split+'-offsets'),len(data[key]),digest,lambda:None)
    config={**native.default_config(),'batch_size':8}
    wrapped=m.facade(native,lambda:None)
    old_rows,old_queries=native.fr._view(data,50);rows,queries=wrapped.fr._view(compacted,50)
    assert list(rows)==old_rows and list(queries)==old_queries
    model=native._model(data,config,'cpu')
    assert native.binding_for4(data,config,old_rows,old_queries,model,'cpu',None)==wrapped.binding_for4(compacted,config,rows,queries,model,'cpu',None)
    assert capture(native,data,config,'cpu',lambda:None)==capture(wrapped,compacted,config,'cpu',lambda:None)
    assert wrapped.fr.train_step is native.fr.train_step and wrapped._model is native._model
    assert wrapped.make_optimizer is native.make_optimizer
    assert native.fr._view(data,50)==(old_rows,old_queries)


def test_compact_adapter_matches_original_side_and_validation(tmp_path,native):
    m=module();from analysis.p_full_training_backend import adapt_rows
    side=importlib.import_module(native.__package__+'.history_side')
    event=lambda item:{'item_id':item,'timespent':20,**{key:False for key in side.FEEDBACK_BITS}}
    events=[event(10),event(11)]
    rows=[{'example_id':0,'user_id':1,'history':[10],'history_positions':[[0,0]],'target':11,'target_position':[0,1],'past_likes':[10]}]
    queries=[{'user_id':1,'history':[10,11],'history_positions':[[0,0],[0,1]],'target':12,'target_position':[25,0],'oov':False}]
    features=np.zeros((4,64),dtype=np.float32)
    original=adapt_rows(native,rows,queries,[10,11,12],[10,11],[(0,events)],{10:30,11:40},features,synthetic=True)
    stored=m.adapt_rows(native,rows,queries,[10,11,12],[10,11],[(0,events)],{10:30,11:40},features,root=tmp_path/'side',guard=lambda:None,synthetic=True)
    assert list(stored['rows'])==original['rows'] and list(stored['queries'])==original['queries']
    assert stored['content'].equal(original['content'])
    for bad in ([{**rows[0],'example_id':2}],[{**rows[0],'target_position':[0,0]}]):
        with pytest.raises(ValueError):m.adapt_rows(native,bad,queries,[10,11,12],[10,11],[],{},features,root=tmp_path/'not-created',guard=lambda:None)
        assert not (tmp_path/'not-created').exists()


def test_train_support_membership_does_not_scan_list_per_row(tmp_path,native):
    class NoLinearMembership(list):
        def __contains__(self,value):
            raise AssertionError('linear supported membership per training row')
    m=module();side=importlib.import_module(native.__package__+'.history_side')
    event=lambda item:{'item_id':item,'timespent':20,**{key:False for key in side.FEEDBACK_BITS}}
    row={'example_id':0,'user_id':1,'history':[10],'history_positions':[[0,0]],
         'target':11,'target_position':[0,1],'past_likes':[10]}
    args=(native,[row],[],[10,11,12],NoLinearMembership([10,11]),
          [(0,[event(10),event(11)])],{10:30,11:40},np.zeros((4,64),dtype=np.float32))
    data=m.adapt_rows(*args,root=tmp_path/'side',guard=lambda:None,synthetic=True)
    assert data['support']=={10,11} and data['supported_indices']==[1,2]
    assert data['rows'][0]['target']==11
    with pytest.raises(ValueError,match='history/target identity'):
        m.adapt_rows(native,[{**row,'target':12}],*args[2:],
                     root=tmp_path/'rejected',guard=lambda:None,synthetic=True)
    assert not (tmp_path/'rejected').exists()


def test_guard_and_original_drift_fail_closed(tmp_path):
    m=module();source=tmp_path/'rows.jsonl';digest=write_rows(source,[{'history':[]}])
    def stop():raise OSError('synthetic resource headroom')
    with pytest.raises(OSError):m.JsonRows.build(source,tmp_path/'not-created',1,digest,stop)
    assert not (tmp_path/'not-created').exists()
    rows=m.JsonRows.build(source,tmp_path/'offsets',1,digest,lambda:None)
    source.write_text('{"drift":true}\n')
    with pytest.raises(ValueError,match='changed'):rows[0]
    with pytest.raises(OSError):m.DenseSide.build(tmp_path/'side',{'train':[],'validation':[]},[],{},lambda *a:[],stop)


def test_offset_cache_drift_and_facade_live_code_rejected(tmp_path,native,monkeypatch):
    m=module();source=tmp_path/'rows.jsonl';digest=write_rows(source,[{'history':[]}]);offsets=tmp_path/'offsets'
    rows=m.JsonRows.build(source,offsets,1,digest,lambda:None)
    offsets.write_bytes(b'\0'*16)
    with pytest.raises(ValueError,match='offset cache'):rows[0]
    with pytest.raises(ValueError,match='offset cache'):list(rows)
    monkeypatch.setattr(native.fr,'_view',lambda *a:None)
    with pytest.raises(ValueError,match='function code'):m.facade(native,lambda:None)


def test_full_digest_does_not_keep_all_decoded_rows(tmp_path):
    import tracemalloc
    m=module();source=tmp_path/'rows.jsonl'
    line=json.dumps({'past_likes':list(range(100000,103000))},sort_keys=True,separators=(',',':'))+'\n'
    source.write_text(line*64);digest=hashlib.sha256(source.read_bytes()).hexdigest()
    rows=m.JsonRows.build(source,tmp_path/'offsets',64,digest,lambda:None)
    tracemalloc.start()
    try:
        value=m.digest(rows);_,peak=tracemalloc.get_traced_memory()
    finally:tracemalloc.stop()
    assert len(value)==64 and peak<2*1024**2


def test_original_combined_cap_is_checked_before_mmap_allocation(tmp_path):
    m=module();output=tmp_path/'output';compact=tmp_path/'compact';parallel=tmp_path/'parallel'
    for root in (output,compact,parallel):root.mkdir()
    (output/'model').write_bytes(b'a'*10);(parallel/'vectors').write_bytes(b'b'*10)
    guard=m.CombinedGuard(lambda:None,[output,compact,parallel],35)
    guard.projected(15)
    with pytest.raises(OSError):guard.projected(16)
    source=tmp_path/'source.jsonl';digest=write_rows(source,[{'history':[]}])
    with pytest.raises(OSError):m.JsonRows.build(source,compact/'offsets',1,digest,guard)
    assert not (compact/'offsets').exists()
    (compact/'escape').symlink_to(output/'model')
    with pytest.raises(ValueError,match='symlink'):guard.projected(0)
