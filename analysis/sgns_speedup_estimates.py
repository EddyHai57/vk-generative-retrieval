"""Read-only Bernoulli deletion/reconnection expectations; no sampled sequences.

Only aggregate count/cost scenarios leave this process. These are engineering
estimates, not a changed SGNS implementation or a full-data measurement.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

from analysis.sgns_speedup_profile import FULL_PAIRS,OBSERVED_RATE,subset_inputs

THRESHOLDS=(1e-3,1e-4,1e-5)
WINDOWS=(1,2,3,5)


def _flatten(histories):
    import numpy as np
    if not histories:raise ValueError('nonempty ordered histories required')
    total=sum(len(row) for row in histories.values())
    if not total:raise ValueError('nonempty events required')
    items=np.empty(total,dtype=np.int64);ends=np.empty(total,dtype=np.int64)
    cursor=0;possible=0
    for user in sorted(histories):
        row=histories[user];end=cursor+len(row)
        items[cursor:end]=row;ends[cursor:end]=end
        possible+=len(row)*(len(row)-1)
        cursor=end
    return items,ends,possible


def original_window_counts(items,ends):
    import numpy as np
    n=len(items);at=np.arange(n,dtype=np.int64);by_distance=[]
    for distance in range(1,max(WINDOWS)+1):
        if distance>=n:by_distance.append(0);continue
        valid=at[:-distance]+distance<ends[:-distance]
        by_distance.append(int(np.count_nonzero(valid & (items[:-distance]!=items[distance:]))))
    return {w:2*sum(by_distance[:w]) for w in WINDOWS}


def _project(pairs,reference):
    if reference<=0:return {'positive_pairs':'UNKNOWN','fit_hours':'UNKNOWN'}
    total=FULL_PAIRS*pairs/reference
    return {'positive_pairs':total,'fit_hours':total/OBSERVED_RATE/3600}


def estimate(histories,thresholds=THRESHOLDS,*,window=5,max_distance=64,chunk_size=65536):
    import numpy as np
    if type(window) is not int or not 1<=window<=5:raise ValueError('bounded window 1..5 required')
    if type(max_distance) is not int or not 1<=max_distance<=512:raise ValueError('bounded max_distance 1..512 required')
    if type(chunk_size) is not int or not 1<=chunk_size<=262144:raise ValueError('bounded chunk_size 1..262144 required')
    if not thresholds or any(type(t) not in (int,float) or not 0<t<1 for t in thresholds):raise ValueError('thresholds between zero and one required')
    start=time.perf_counter()
    items,ends,possible=_flatten(histories);n=len(items)
    _,inverse,counts=np.unique(items,return_inverse=True,return_counts=True)
    relative=counts.astype(np.float64)/n
    original=original_window_counts(items,ends);baseline=original[5]
    c1=[]
    for threshold in thresholds:
        p_by_item=np.minimum(1.,np.sqrt(threshold/relative)+threshold/relative)
        p=p_by_item[inverse]
        prefix=np.empty(n+1,dtype=np.float64);prefix[0]=0.;np.cumsum(p,out=prefix[1:])
        partial=omitted=0.
        for start_at in range(0,n,chunk_size):
            stop=min(n,start_at+chunk_size)
            at=np.arange(start_at,stop,dtype=np.int64);end=ends[start_at:stop]
            # q[k,i]=P(exactly k kept intermediates), k<window. Probability
            # mass beyond window-1 is discarded; it can never become eligible.
            q=np.zeros((window,len(at)),dtype=np.float64);q[0]=1.
            center=p[start_at:stop]
            for distance in range(1,max_distance+1):
                valid=at+distance<end
                if not np.any(valid):break
                target=np.minimum(at+distance,n-1)
                target_keep=p[target]
                different=items[start_at:stop]!=items[target]
                partial+=float(np.sum(center*target_keep*q.sum(axis=0)*valid*different))
                # Endpoint of this distance becomes an intermediate next time.
                # Reverse update preserves previous q[k-1] without a copy.
                for k in range(window-1,0,-1):
                    q[k]*=1-target_keep;q[k]+=q[k-1]*target_keep
                q[0]*=1-target_keep
            # For every farther endpoint j, first D intermediates must have
            # <window survivors. Dropping further intermediates and same-item
            # exclusion gives a conservative omitted-pair upper bound.
            tail_start=np.minimum(at+max_distance+1,end)
            tail_mass=np.maximum(0.,prefix[end]-prefix[tail_start])
            omitted+=float(np.sum(center*tail_mass*q.sum(axis=0)))
        partial*=2;omitted*=2
        event_bound=2*window*float(p.sum())
        # Analytic omitted bound above is rigorous in real arithmetic. This
        # deliberately conservative absolute IEEE roundoff allowance accounts
        # for cumsums, cancellation, recurrence, and positive reductions.
        allowance=64*np.finfo(np.float64).eps*(n+max_distance+window+1)*max(1,possible)
        lower=max(0.,partial-allowance)
        upper=min(float(possible),event_bound+allowance,partial+omitted+allowance)
        c1.append({'threshold':threshold,'expected_retained_events':float(p.sum()),
                   'truncated_expected_pairs_per_epoch':partial,'omitted_pairs_upper_bound':omitted,
                   'event_count_pairs_upper_bound':event_bound,'roundoff_allowance_pairs':allowance,
                   'lower_pairs_per_epoch':lower,'upper_pairs_per_epoch':upper,
                   'truncation_tail_bound_zero_in_float64':omitted==0.,
                   'full_scenario_from_subset_ratio':{'lower':_project(lower,baseline),'upper':_project(upper,baseline),
                       'measured':False,'distribution_transfer_error':'UNKNOWN','total_wall':'UNKNOWN'}})
    return {'schema':'sgns-readonly-expectation-estimates-v1','science_training':False,'sampled_sequences_created':False,
            'full_measured':False,'window_for_C1':window,'max_distance':max_distance,'chunk_size':chunk_size,
            'events':n,'users':len(histories),'catalog_observed_items':len(counts),
            'original_subset_pairs_per_epoch_window5':baseline,'C1':c1,
            'C3':[{'window':w,'subset_pairs_per_epoch':original[w],
                   'full_scenario_from_subset_ratio':_project(original[w],baseline),
                   'full_measured':False,'distribution_transfer_error':'UNKNOWN','total_wall':'UNKNOWN'} for w in WINDOWS],
            'elapsed_seconds':time.perf_counter()-start,
            'estimator_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'scope':['C1 independent event Bernoulli keep; delete then reconnect fixed window; same-item pairs excluded',
                     'C1 expectations, not sampled actual counts; no random RNG or event deletions executed',
                     'full scenarios scale original full pair baseline by subset ratio; not full measurements',
                     'fit hours reuse prior subset fit-scope throughput; setup and changed throughput not modeled',
                     'C1 analytic tail bound plus conservative float64 allowance; threshold formula is a proposal only']}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('dry')
    for name in ('synthetic','subset'):
        c=sub.add_parser(name);c.add_argument('--max-distance',type=int,default=64);c.add_argument('--chunk-size',type=int,default=65536)
        if name=='subset':
            c.add_argument('--subset-bundle',type=Path,required=True);c.add_argument('--subset-manifest-sha256',required=True);c.add_argument('--lock',type=Path,required=True)
    a=p.parse_args(argv)
    try:
        if a.command=='dry':r={'status':'dry','science_training':False,'full_measured':False,'thresholds':THRESHOLDS,'windows':WINDOWS}
        else:
            if a.command=='synthetic':
                histories={i:[1+(i*3+j*7)%31 for j in range(48)] for i in range(16)}
            else:
                _,histories,_=subset_inputs(a.subset_bundle,a.subset_manifest_sha256,a.lock)
            r=estimate(histories,max_distance=a.max_distance,chunk_size=a.chunk_size)
            r['synthetic']=a.command=='synthetic'
            if a.command=='subset':
                if r['original_subset_pairs_per_epoch_window5']!=34770216:raise ValueError('original subset positive-pair audit count differs')
                subset_inputs(a.subset_bundle,a.subset_manifest_sha256,a.lock)
                r['subset_manifest_sha256']=a.subset_manifest_sha256
        print(json.dumps(r,sort_keys=True));return 0
    except (ValueError,RuntimeError,KeyError,OSError,TypeError) as exc:
        print(json.dumps({'status':'rejected','reason':str(exc),'science_training':False},sort_keys=True));return 2


if __name__=='__main__':raise SystemExit(main())
