"""One-shot morning acceptance: fresh verifier bindings and frozen statistics.
Only validation predictions and JSON/log receipts; no writes/model/checkpoints.
"""
import datetime
import hashlib
import json
from pathlib import Path
import sys

OPS=Path('/workspace/vkgr-seqm1-night-20261016')

def mean_rules(deltas,p_pool,ratios):
    if len(deltas)!=3 or len(ratios)!=3:return {'row':'UNKNOWN'}
    d=sum(deltas)/3;kplus=sum(v>=6 for v in deltas);kzero=sum(v>=-5 for v in deltas);enabled=sum(v is not None and v>=.05 for v in ratios)
    row=1 if d<=-6 else 2 if d>=6 and kplus>=2 and p_pool<.05 else 3 if d>=6 else 4
    return {'row':row,'mean_delta_hits':d,'seeds_ge6':kplus,'seeds_ge_minus5':kzero,'branch_enabled_seeds':enabled,
            'm2_design_performance_and_branch':d>-6 and kzero>=2 and enabled>=2,'p_pool':p_pool}
def tok_state(deltas):
    if len(deltas)!=2:return 'UNKNOWN'
    return 'UP' if all(d>=6 for d in deltas) else 'DOWN' if all(d<=-6 for d in deltas) else 'FLAT'
def q1(a,b):
    if 'UP' in (a,b):return 'still_rising'
    if 'T_UP' in (a,b):return 'undetermined'
    if a==b=='DOWN':return 'peak_256'
    if a in ('UP','DOWN','FLAT') and b in ('UP','DOWN','FLAT'):return 'no_further_gain'
    return 'tentative_or_unknown'

def main():
    source=OPS/'code/analysis/seqm1_night.py'
    if hashlib.sha256(source.read_bytes()).hexdigest()!='9e2b1a2e5114882e25a56233883ccacdebed4235be6f53e457c3eac3d8c7edda':raise ValueError('night source pin mismatch')
    sys.path.insert(0,str(source.parent))
    import seqm1_night as night
    import phase2_validation_read as r
    started=night.now();plan=r.read(OPS/'plan.json')[0];states={}
    for j in plan['jobs']:
        try:states[j['run_id']]=night.snapshot(j)
        except Exception as exc:states[j['run_id']]={'run_id':j['run_id'],'verified':False,'unaccepted':True,'exceptions':[type(exc).__name__+': '+str(exc)]}
    for rid,s in states.items():
        for suffix,key in (('','train_terminal'),('-verify','verify_terminal')):
            p=r.ROOT/'runs'/(rid+suffix)/'terminal.json'
            if p.exists():s[key]=r.read(p)[0]
    hist,_=r.read(r.ROOT/'runs/h2-prepare/out/manifest.json',r.HISTORY_PIN)
    queries,_=r.read(r.ROOT/'runs/h2-prepare/out/validation-l50.json',hist['artifacts']['validation-l50.json'])
    hist['_base_queries_canonical_digest']=hist['binding']['query_sha']
    if len(queries)!=2520 or sum(q['oov'] for q in queries)!=623:raise ValueError('fixed population mismatch')
    data={};summaries={};paired={};diags={}
    for rid,s in states.items():
        if not s['verified']:continue
        adapter={'verified':True,'terminal':{'status':s['train_status']},'manifest_sha256':s['manifest_sha256']}
        data[rid]=r.load_archive(rid,queries,hist,adapter);summaries[rid]=data[rid]['summary']
        m,_=r.read(s['manifest_path'],s['manifest_sha256']);be=m['selection']['best_epoch'];ev=[v for v in m['evaluations'] if v['epoch']==be]
        if len(ev)!=1:raise ValueError('best diagnostics cardinality')
        diags[rid]=ev[0]['diagnostics']
    for seed in night.SEEDS:
        rid=f'p-php01-h1-t010-s{seed}';s=r.inspect_run(rid)
        if not s['verified']:raise ValueError('baseline not accepted')
        data[rid]=r.load_archive(rid,queries,hist,s)
    arms={}
    for arm in night.ARMS:
        pairs={};ratios=[];deltas=[];bs=cs=0
        for seed in night.SEEDS:
            rid=f'p-seqm1-{arm}-s{seed}';ctl=f'p-php01-h1-t010-s{seed}'
            if rid not in data:continue
            p=r.paired(data[rid],data[ctl]);pairs[str(seed)]=p;bs+=p['only_new'];cs+=p['only_control'];deltas.append(p['delta_hits']);ratios.append(states[rid].get('branch_ratio_median'))
        pool=r.mcnemar(bs,cs) if len(pairs)==3 else None
        arms[arm]={'paired_vs_f4':pairs,'p_pool':pool,'p_pool_counts':{'only_new':bs,'only_control':cs},'mean_hits':sum(p['new_hits'] for p in pairs.values())/3 if len(pairs)==3 else None,
                   'exact_significant_seeds':sum(p['mcnemar_exact_two_sided']<.05 for p in pairs.values())}
        if arm!='pure':
            arms[arm]['mechanical']=mean_rules(deltas,pool,ratios)
            arms[arm]['mechanical']['main_conclusion_eligible']=arm=='mean'
            if arm!='mean':arms[arm]['mechanical'].pop('m2_design_performance_and_branch',None)
        g0=[states[f'p-seqm1-{arm}-s{s}'].get('ablation_delta_hits') for s in night.SEEDS]
        arms[arm]['verified_seed_count']=len(pairs)
        arms[arm]['g0_positive_seeds']=sum(v>0 for v in g0) if arm!='pure' and all(v is not None for v in g0) else None
        arms[arm]['paired_vs_mean']={}
        if arm!='mean':
            for seed in night.SEEDS:
                a,b=f'p-seqm1-{arm}-s{seed}',f'p-seqm1-mean-s{seed}'
                if a in data and b in data:arms[arm]['paired_vs_mean'][str(seed)]=r.paired(data[a],data[b])
    representative=None
    anchored=('mean','last','attn','bi')
    if all(arms[a]['mean_hits'] is not None for a in anchored):
        representative=sorted(anchored,key=lambda a:(-arms[a]['mean_hits'],-sum(summaries[f'p-seqm1-{a}-s{s}']['ndcg'] for s in night.SEEDS)/3,anchored.index(a)))[0]
        if arms[representative]['mean_hits']-arms['mean']['mean_hits']<6:representative='mean'
        arms[representative]['p_pool_bonferroni4']=min(1.,arms[representative]['p_pool']*4) if representative!='mean' else arms[representative]['p_pool']
    dp=OPS/'dispatch-terminal.json'
    automatic={'status':'not_released','exact_twelve':False,'proofs':{}}
    if dp.exists():
        dispatch,dsha=r.read(dp);published={v['run_id']:v for v in dispatch.get('published',[])}
        automatic.update(status=dispatch['status'],released_at_utc=dispatch.get('utc'),dispatch_receipt_sha256=dsha,exact_twelve=set(published)==set(night.DIAGS))
        for rid in night.DIAGS:
            q=states[rid].get('train_queue',{});name=published.get(rid,{}).get('queue_file');matches=name in q.get('claimed',[])
            equal=matches and r.read(night.Q/'claimed'/name)[0]==night.queue_payload(next(j for j in plan['jobs'] if j['run_id']==rid))
            automatic['proofs'][rid]={'published_file':name,'claimed':q.get('claimed',[]),'rejected':q.get('rejected',[]),'accepted_original_file':matches,'claimed_payload_matches_frozen_plan':equal}
        dispatch_time=datetime.datetime.fromisoformat(dispatch['utc']).timestamp()
        automatic['canary_verified_before_release']=any(v.get('verified',False) and states[rid].get('verified',False) and v.get('manifest_sha256')==states[rid].get('manifest_sha256') and states[rid].get('verify_terminal',{}).get('ended_at',float('inf'))<=dispatch_time for rid,v in dispatch.get('mean_states',{}).items())
    files={}
    for name in ('dispatcher.jsonl','verifier.jsonl','summary-final.json','summary-deadline.json','verifier-fatal.json','dispatcher-launcher.log','verifier-launcher.log'):
        p=OPS/name
        if p.exists():
            payload=p.read_bytes();files[name]={'path':str(p),'sha256':hashlib.sha256(payload).hexdigest()}
            if name.endswith(('.jsonl','.log')):files[name]['tail']=payload.decode(errors='replace')[-2500:]
    ended=[]
    for p in (r.ROOT/'runs').glob('*/terminal.json'):
        t,_=r.read(p)
        if isinstance(t.get('ended_at'),(int,float)):ended.append({'run_id':p.parent.name,'ended_at':t['ended_at'],'status':t['status']})
    output={'read_at_utc':started,'finished_at_utc':night.now(),'states':states,'archive_summaries':summaries,'best_diagnostics':diags,'arms':arms,'representative_mechanical':representative,
            'automatic_continuation':automatic,'files':files,'latest_project_terminal':max(ended,key=lambda v:v['ended_at']) if ended else None,
            'read_artifacts_sha256':r.READS,'test_read':False,'checkpoint_read':False,'new_submit':False,
            'limitations':['p_pool pools repeated queries across seeds and is optimistic per frozen rule','representative selection is exploratory; mean remains main conclusion','no latency/leakage promotion verdict or M2 enqueue']}
    print(json.dumps(output,sort_keys=True,allow_nan=False))

if __name__=='__main__':main()
