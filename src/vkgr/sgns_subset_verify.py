"""Original 079d6bd C1 audit verifier, verbatim function bodies; no SGD replay."""
import math
import re
import numpy as np
from . import sid_coll_sgns as sg, sid_tok_audit as au, sid_v2

def _hex(value,length=64):
    if type(value) is not str or not re.fullmatch('[0-9a-f]{'+str(length)+'}',value):raise ValueError('invalid pinned SHA')
    return value

def verify_c1_audit(ids,histories,content,vectors,audit,*,engineering=False):
    """Independent data/algebra/progress checks; does not replay SGD gradients."""
    if not engineering and len(ids)!=120415:raise ValueError('formal catalog count')
    cfg=sg.SGNSConfig(**audit['config']);cfg.validate(engineering)
    if cfg.dim!=64 or cfg.epochs!=5:raise ValueError('S1 requires 64D and five epochs')
    if vectors.dtype!=np.float32 or vectors.shape!=(len(ids),64) or not np.isfinite(vectors).all():raise ValueError('finite float32 vectors')
    if not np.allclose(np.linalg.norm(vectors.astype('float64'),axis=1),1,atol=2e-6,rtol=0):raise ValueError('unit vectors required')
    expected_contract={'loss_reduction':'sum','optimizer':'SGD','momentum':0,'weight_decay':0,'gradient_clipping':None,
      'frequency_power':.75,'subsampling':False,'distance_weighting':False,'watch_ratio_weighting':False,
      'early_stopping':False,'export_table':'center_only','initial_input_bound':.5/cfg.dim,
      'positive_order':'user ascending; center ascending; context oldest to newest',
      'negative_rng':'numpy.default_rng(seed), continuous across epochs'}
    if audit['training_contract']!=expected_contract:raise ValueError('frozen SUM/SGD contract mismatch')
    if audit['schema']!='vkgr.sid_coll_sgns.v1' or audit['engineering'] is not engineering or audit['threads']!=1 or audit['deterministic_algorithms'] is not True:
        raise ValueError('SGNS producer runtime/schema mismatch')
    if audit['epochs_completed']!=5:raise ValueError('five complete epochs required')
    rows={int(x):i for i,x in enumerate(ids)};counts=np.zeros(len(ids),dtype=np.int64);frequency=np.zeros(len(ids),dtype=np.int64)
    for events in histories.values():
        for item in events:frequency[rows[int(item)]]+=1
    pairs=0
    for c,p in sg.iter_positive_pairs(histories,cfg.window):counts[rows[c]]+=1;counts[rows[p]]+=1;pairs+=1
    if pairs<=0:raise ValueError('no positive pairs')
    steps=math.ceil(pairs/cfg.batch_size);total=pairs*5;last_batch=pairs%cfg.batch_size or cfg.batch_size
    if audit['positive_pairs_per_epoch']!=pairs or audit['processed_positive_pairs']!=total or audit['steps']!=steps*5:
        raise ValueError('positive progress/step mismatch')
    lr=audit['learning_rates']
    expected_lrs={'initial':cfg.initial_lr,'last_batch':sg.learning_rate(total-last_batch,total,cfg.initial_lr,cfg.final_lr_ratio),
                  'final':sg.learning_rate(total,total,cfg.initial_lr,cfg.final_lr_ratio)}
    if any(type(lr.get(k)) not in (float,int) or not math.isclose(lr[k],v,rel_tol=1e-12,abs_tol=1e-15) for k,v in expected_lrs.items()):
        raise ValueError('learning rate endpoint/progress mismatch')
    epochs=audit['epoch_audits']
    if len(epochs)!=5:raise ValueError('epoch ledger length')
    for epoch,record in enumerate(epochs,1):
        if record['epoch']!=epoch or record['processed_positive_pairs']!=epoch*pairs or record['steps']!=epoch*steps or not math.isfinite(record['summed_loss']) or record['summed_loss']<0:
            raise ValueError('epoch ledger mismatch')
    fallback=counts==0
    expected={'catalog_sha256':sid_v2.catalog_digest(ids),'vectors_sha256':au.feature_digest(vectors),
       'content_feature_sha256':au.feature_digest(content),'content_sha256':au.feature_digest(content),
       'train_sequence_sha256':sg._sequence_digest(histories),'raw_train_sequence_sha256':sg._sequence_digest(histories),
       'frequency_sha256':sg._array_digest(frequency),'positive_participation_sha256':sg._array_digest(counts),
       'fallback_mask_sha256':sg._array_digest(fallback)}
    if any(audit.get(k)!=v for k,v in expected.items()):raise ValueError('SGNS input/output/sequence digest mismatch')
    if audit['positive_participation']!=counts.tolist() or audit['fallback_mask']!=fallback.tolist() or audit['fallback_count']!=int(fallback.sum()) or audit['fallback_proportion']!=float(fallback.mean()):
        raise ValueError('fallback/participation ledger mismatch')
    norms=np.linalg.norm(content.astype('float64'),axis=1)
    if not np.isfinite(content).all() or np.any(norms[fallback]==0):raise ValueError('invalid fallback content')
    fallback_vectors=(content[fallback].astype('float64')/norms[fallback,None]).astype('float32')
    if not np.array_equal(vectors[fallback],fallback_vectors):raise ValueError('fallback algebra mismatch')
    health=sg.health_check(vectors,ids,histories)
    if audit['health']!=health:raise ValueError('health evidence mismatch')
    for key in ('negative_trace_sha256','input_initialization_sha256','output_table_sha256'):_hex(audit[key])
    if not engineering and au.feature_digest(content)!=au.CONTENT_V1_FEATURE_SHA256:raise ValueError('CONTENT_V1 source mismatch')
    return {'schema':'coll-input-verify-v1','status':'verified','gradient_replay':False,
       'negative_trace_replay':False,'negative_trace_sha256':audit['negative_trace_sha256'],
       'checks':['artifact_hashes','source_physical_hashes','sequence_recomputed','SUM_SGD_contract','five_epoch_progress',
                 'learning_rate_endpoints','fallback_algebra','unit_finite','health_recomputed','negative_trace_recorded_not_replayed'],
       'health':health}
