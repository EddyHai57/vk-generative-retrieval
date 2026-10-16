"""Pure S1 wrappers around frozen S0 A--D and supplemental D-prime math.

No paths, files, models, training or acceptance gates. Mapping keys are item IDs;
each stored full SID has exactly four positions (three semantic + suffix).
Both cooccurrence calculations use the same reusable histories, preserving
repeated events and user boundaries. Frozen catalog order is ascending item ID.
"""
from numbers import Integral
try:
    from . import sid_diag_metrics as metrics
    from .sid_diag_prime import cooccurrence_prime
except ImportError:
    import sid_diag_metrics as metrics
    from sid_diag_prime import cooccurrence_prime


def _mapping(mapping):
    if any(not isinstance(i,Integral) or isinstance(i,bool) or i<0 for i in mapping):
        raise ValueError('mapping item IDs must be nonnegative integers')
    codes=metrics._codes(mapping,4)
    if any(len(code)!=4 for code in codes.values()):
        raise ValueError('exactly four full-SID columns required')
    return {item:codes[item] for item in sorted(codes)}


def mapping_diagnostics(mapping,histories):
    """Return A, original D (self pairs retained), and Dprime at seed zero."""
    mapping=_mapping(mapping)
    # Reuse caller-owned sequences. Consume one-shot iterators only once, so D
    # and Dprime see identical events even with a streaming synthetic fixture.
    rows=[row if isinstance(row,(list,tuple)) else tuple(row) for row in histories]
    return {'A':metrics.structure_metrics(mapping,256,3),
            'D':metrics.cooccurrence_metrics(mapping,rows,seed=0),
            'Dprime':cooccurrence_prime(mapping,rows,seed=0)}


def prediction_diagnostics(targets,predictions,mapping,slices,reference_C):
    """Return S0 B/C and exact per-bin hit deltas versus same-seed T3 C.

    slices is the original G4 train-slices document, without fitted thresholds.
    Unknown reference C remains UNKNOWN. Predictions must already be accepted,
    ordered exact item archives; S0's OOV and deduplication rules are retained.
    """
    mapping=_mapping(mapping)
    thresholds=slices['popularity_thresholds']
    counts={int(k):v for k,v in slices['item_like_counts'].items()}
    b=metrics.prediction_metrics(targets,predictions,mapping)
    c=metrics.popularity_metrics(targets,predictions,counts,thresholds['median'],thresholds['p90'],mapping)
    if reference_C is None or reference_C.get('status')=='UNKNOWN':
        delta={'status':'UNKNOWN','reason':'same-seed T3 C reference unavailable'}
    else:
        ref=reference_C['bins'];delta={}
        for name,row in c['bins'].items():
            hits=ref[name]['hits']
            if not isinstance(hits,Integral) or isinstance(hits,bool) or hits<0:
                raise ValueError('reference bin hits must be nonnegative integers')
            delta[name]=row['hits']-int(hits)
    return {'B':b,'C':c,'C_delta_vs_T3':delta}
