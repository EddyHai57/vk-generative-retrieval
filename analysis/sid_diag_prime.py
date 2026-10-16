"""SID-FIX supplemental D-prime, independent of original read-only D.

Use --plan pinned-S0-plan.json --original-result result.json
--original-sha SHA --out NEW_DIRECTORY [--preflight]. No predictions are read.
"""
import argparse
from collections import Counter
import datetime
import json
import hashlib
from pathlib import Path
import random
try:
    from . import sid_diag as source
    from .sid_diag_metrics import _codes
except ImportError:
    import sid_diag as source
    from sid_diag_metrics import _codes


def cooccurrence_prime(mapping, histories, seed=0, random_pairs=None):
    """Streaming adjacency counts; optional index pairs are a synthetic oracle."""
    if type(seed) is not int:raise ValueError('integer seed required')
    codes=_codes(mapping,3)
    population=[codes[i] for i in sorted(codes)]
    n=len(population)
    observed=[0,0,0]; baseline=[0,0,0]; adjacent=same=pairs=0
    for history in histories:
        previous=None; have=False
        for item in history:
            if item not in codes: raise ValueError('history OOV: no skips/no bridging')
            if have:
                adjacent+=1
                if previous==item: same+=1
                else:
                    pairs+=1
                    for j in range(3):observed[j]+=codes[previous][:j+1]==codes[item][:j+1]
            previous=item;have=True
    if pairs and n<2: raise ValueError('distinct random pair requires two catalog items')
    rng=random.Random(seed)
    draws=(rng.sample(range(n),2) for _ in range(pairs)) if random_pairs is None else iter(random_pairs)
    n_draws=0
    for a,b in draws:
        if type(a) is not int or type(b) is not int or a==b or not 0<=a<n or not 0<=b<n: raise ValueError('distinct in-range random indices required')
        n_draws+=1
        for j in range(3):baseline[j]+=population[a][:j+1]==population[b][:j+1]
    if n_draws!=pairs:raise ValueError('random sample count differs from observed pairs')
    prefixes={}
    for j in range(3):
        maximum=max(Counter(c[:j+1] for c in population).values(),default=0)
        status='STRUCT_NA' if maximum<=1 else 'OK' if baseline[j] else 'UNKNOWN'
        lift='STRUCT_NA' if status=='STRUCT_NA' else observed[j]/baseline[j] if baseline[j] else None
        prefixes[str(j+1)]={'n':pairs,'max_prefix_group':maximum,'cooccurrence_matches':observed[j],'random_matches':baseline[j], 'cooccurrence_rate':observed[j]/pairs if pairs else None,'random_rate':baseline[j]/pairs if pairs else None,'lift':lift,'status':status}
    return {'seed':seed,'catalog_order':'ascending_item_id','catalog_size':n,'random_algorithm':'random.Random(seed).sample(range(N),2); distinct within pair; repeated pairs allowed','history_oov_policy':'fail_closed_no_skips_no_bridging','stored_prefix_positions':[1,2,3],'n_adjacent_events':adjacent,'n_same_item_excluded':same,'n_pairs':pairs,'n_random_pairs':n_draws,'prefix_by_k':prefixes}


def run(plan, original, out, expected_pairs=3630957, plan_ref=None):
    if Path(out).exists():raise ValueError('new exclusive output directory required')
    source.READS.clear();source.archive.READS.clear()
    inventory=source.collect_inventory(plan)
    original_value=source.read_json(original)
    if 'D' not in original_value:raise ValueError('original S0 D required')
    history=source.read_json(plan['history']);catalog=source.read_json(plan['catalog'])
    if len(catalog)!=120415 or len(set(catalog))!=len(catalog) or catalog!=sorted(catalog) or source.archive.plain_digest(catalog)!=history['catalog_sha256'] or plan['catalog']['sha256']!=history['binding']['catalog_sha']:raise ValueError('frozen ascending catalog mismatch')
    audit=source.read_json(plan['train_audit'])
    histories=source.raw_histories(plan,source.train_records(audit),catalog)
    values={}
    base=source.load_mapping(plan['bundles']['B16'],catalog)
    for label,ref in plan['bundles'].items():
        mapping=source.random_mapping(base) if label=='random' else base if label=='B16' else source.load_mapping(ref,catalog)
        values[label]=cooccurrence_prime(mapping,histories)
        if values[label]['n_pairs']!=expected_pairs:raise ValueError('frozen D pair count mismatch')
    result={'task_id':'SID-FIX-S1-Dprime','read_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'plan_sha256':source.archive.digest(plan),'plan_reference':plan_ref,'original_D_reference':original,'original_D':original_value['D'],'Dprime':values,'inventory':inventory,'read_artifacts_sha256':{**inventory['read_artifacts_sha256'],**source.READS,**source.archive.READS},'restrictions':{'predictions_read':False,'checkpoint_read':False,'original_modified':False}}
    write_result(out,result)
    return result


def write_result(out,result):
    p=Path(out)
    if p.exists() or any(x.is_symlink() for x in [p,*p.parents]):raise ValueError('new exclusive output directory required')
    p.mkdir(parents=True,exist_ok=False)
    (p/'result.json').write_bytes(source.archive.canonical(result))
    lines=['# SID-FIX supplemental read','', 'SHA256 receipts and actual read UTC: result.json. Original D is preserved as a pinned read-only reference.','', '|Map|Pairs|Lift1|Lift2|Lift3|','|---|---:|---|---|---|']
    for name,row in result.get('Dprime',{}).items():lines.append('|'+ '|'.join(map(str,[name,row['n_pairs'],*[row['prefix_by_k'][str(k)]['lift'] for k in (1,2,3)]]))+'|')
    if 'Dprime' not in result:lines.append(json.dumps(result,ensure_ascii=False,indent=2))
    (p/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True);parser.add_argument('--original-result',required=True);parser.add_argument('--original-sha',required=True);parser.add_argument('--out',required=True);parser.add_argument('--preflight',action='store_true')
    args=parser.parse_args();path=source.safe_path(args.plan)
    if path.stat().st_size>2*1024*1024:raise ValueError('bounded plan required')
    payload=path.read_bytes();plan=json.loads(payload)
    plan_ref={'path':str(path),'sha256':hashlib.sha256(payload).hexdigest(),'read_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    if args.preflight:write_result(args.out,source.collect_inventory(plan))
    else:run(plan,{'path':args.original_result,'sha256':args.original_sha},args.out,plan_ref=plan_ref)
if __name__=='__main__':main()
