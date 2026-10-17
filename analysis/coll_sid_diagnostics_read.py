"""Bounded, pinned aggregate intake; never opens predictions, mappings or models.

stdin JSON: schema=coll-sid-aggregate-request-v1, files=[{path,sha256,
run_id,role:mapping|prediction,variant:C1|C2,seed:null|42|2024|2025}].
Exactly two mapping and six prediction aggregates required. stdout is a compact
JSON readback with UTC, source pins and validated A/B/C/D/Dprime. No writes.
"""
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import sys

LIMIT=2*1024*1024
BINS=('high','mid','low','unsupported','oov')

def require(ok,message):
    if not ok: raise ValueError(message)

def integer(n):return type(n) is int and n>=0

def ratio(n,d):return n/d if d else None

def equal(a,b):
    return a is None and b is None or type(a) in (float,int) and type(b) in (float,int) and math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-12)

def no_duplicates(pairs):
    d={}
    for k,v in pairs:
        require(k not in d,'duplicate JSON key');d[k]=v
    return d

def decode(raw):
    return json.loads(raw,object_pairs_hook=no_duplicates,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('nonfinite JSON')))

def validate_c(c):
    require(set(c)=={'thresholds','bins','supplemental_longtail_unsupported'},'C schema')
    th=c['thresholds'];require(set(th)=={'median','p90'} and all(type(th[k]) in (float,int) and math.isfinite(th[k]) for k in th) and 0<=th['median']<=th['p90'],'C thresholds')
    require(set(c['bins'])==set(BINS),'five original C bins required')
    for k,row in c['bins'].items():
        require(set(row)=={'n','hits','rate'} and integer(row['n']) and integer(row['hits']) and row['hits']<=row['n'],'C count')
        require(equal(row['rate'],ratio(row['hits'],row['n'])),'C rate')
        require(k!='oov' or row['hits']==0,'OOV hits forbidden')
    s=c['supplemental_longtail_unsupported'];n=sum(c['bins'][k]['n'] for k in ('low','unsupported'));h=sum(c['bins'][k]['hits'] for k in ('low','unsupported'))
    require(s=={'n':n,'hits':h,'rate':ratio(h,n)},'supplemental C count')

def validate_prediction(v):
    require(set(v)=={'B','C','C_delta_vs_T3'},'prediction aggregate schema')
    b=v['B'];require(set(b)=={'n_total','n_in_catalog','n_oov','prefix_by_k','exact_item','full_sid','full_sid_is_item_equivalent'},'B schema')
    require(all(integer(b[k]) for k in ('n_total','n_in_catalog','n_oov')) and b['n_total']==b['n_in_catalog']+b['n_oov'],'B OOV denominator')
    require(type(b['full_sid_is_item_equivalent']) is bool and set(b['prefix_by_k'])=={'1','2','3'},'B prefixes')
    for row in [*b['prefix_by_k'].values(),b['exact_item'],b['full_sid']]:
        require(set(row)=={'n','top1_hits','top50_hits','top1_rate','top50_rate'} and row['n']==b['n_in_catalog'],'B stats schema')
        require(integer(row['top1_hits']) and integer(row['top50_hits']) and row['top1_hits']<=row['top50_hits']<=row['n'],'B hits')
        require(all(equal(row[k+'_rate'],ratio(row[k+'_hits'],row['n'])) for k in ('top1','top50')),'B rate')
    c=v['C'];validate_c(c)
    require(sum(x['n'] for x in c['bins'].values())==b['n_total'] and c['bins']['oov']['n']==b['n_oov'],'C/B denominator mismatch')
    require(sum(x['hits'] for x in c['bins'].values())==b['exact_item']['top50_hits'],'C/B hits mismatch')
    d=v['C_delta_vs_T3'];require(set(d)==set(BINS) and all(type(x) is int for x in d.values()),'missing C delta')

def compare_c(v,ref):
    validate_c(ref);c=v['C'];require(c['thresholds']==ref['thresholds'],'reference threshold mismatch')
    require(all(c['bins'][k]['n']==ref['bins'][k]['n'] for k in BINS),'reference target denominator mismatch')
    return {k:c['bins'][k]['hits']-ref['bins'][k]['hits'] for k in BINS}

def validate_lift_row(row,n,prime):
    keys={'n','cooccurrence_matches','random_matches','cooccurrence_rate','random_rate','lift'}|({'max_prefix_group','status'} if prime else set())
    require(set(row)==keys and row['n']==n,'D row schema')
    a,b=row['cooccurrence_matches'],row['random_matches'];require(integer(a) and integer(b) and a<=n and b<=n,'D match counts')
    require(equal(row['cooccurrence_rate'],ratio(a,n)) and equal(row['random_rate'],ratio(b,n)),'D rates')
    if prime:
        m=row['max_prefix_group'];require(integer(m),'Dprime max group')
        status='STRUCT_NA' if m<=1 else 'OK' if b else 'UNKNOWN'
        require(row['status']==status,'Dprime status')
        if status=='STRUCT_NA':
            require(row['lift']=='STRUCT_NA' and a==b==0,'Dprime STRUCT_NA');return
    require(equal(row['lift'],ratio(a,b)),'D lift / UNKNOWN')

def validate_mapping(v):
    require(set(v)=={'A','D','Dprime','physical_sources'},'mapping aggregate schema')
    require(type(v['physical_sources']) is dict and len(v['physical_sources'])<=64 and all(type(k) is str and type(h) is str and re.fullmatch('[0-9a-f]{64}',h) for k,h in v['physical_sources'].items()),'physical source pins')
    a=v['A'];require(set(a)=={'n_items','semantic_depth','layers','effective_layers','leaves','full_code_leaves'} and integer(a['n_items']) and a['semantic_depth']==3,'A schema')
    require(type(a['layers']) is list and len(a['layers'])==3,'A layer count')
    effective=0
    for k,row in enumerate(a['layers'],1):
        require(set(row)=={'k','branching','distinct_codes','utilization','normalized_entropy','parents','nodes','avg_children','avg_items_per_node'},'A layer schema')
        require(row['k']==k and row['branching']==256 and all(integer(row[x]) for x in ('distinct_codes','parents','nodes')),'A layer identity')
        require(row['distinct_codes']<=256 and equal(row['utilization'],row['distinct_codes']/256) and equal(row['avg_children'],ratio(row['nodes'],row['parents'])) and equal(row['avg_items_per_node'],ratio(a['n_items'],row['nodes'])),'A ratios')
        entropy=row['normalized_entropy'];require(entropy is None if not a['n_items'] else type(entropy) in (float,int) and math.isfinite(entropy) and -1e-12<=entropy<=1+1e-12,'A entropy')
        effective+=row['avg_children'] is not None and row['avg_children']>1
    require(a['effective_layers']==effective,'A effective layers')
    for leaf in (a['leaves'],a['full_code_leaves']):
        require(set(leaf)=={'count','avg_items','max_items','collision_items','collision_pairs','collided_leaves'} and all(integer(leaf[k]) for k in leaf if k!='avg_items'),'A occupancy schema')
        require(leaf['count']+leaf['collision_items']==a['n_items'] and equal(leaf['avg_items'],ratio(a['n_items'],leaf['count'])),'A occupancy ratio')
    for role in ('D','Dprime'):
        d=v[role];require(set(d)=={'seed','random_algorithm','catalog_order','catalog_size','history_oov_policy','n_adjacent_events','n_same_item_excluded','n_pairs','n_random_pairs','prefix_by_k'}|({'stored_prefix_positions'} if role=='Dprime' else set()),'D schema')
        require(all(type(d[k]) is str for k in ('random_algorithm','catalog_order','history_oov_policy')) and (role!='Dprime' or d['stored_prefix_positions']==[1,2,3]),'D metadata')
        require(d['seed']==0 and d['catalog_size']==a['n_items'] and integer(d['n_pairs']) and d['n_random_pairs']==d['n_pairs'],'D identity')
        require(all(integer(d[k]) for k in ('n_adjacent_events','n_same_item_excluded','n_pairs','n_random_pairs')) and d['n_adjacent_events']==d['n_same_item_excluded']+d['n_pairs'] and set(d['prefix_by_k'])=={'1','2','3'},'D adjacency')
        for row in d['prefix_by_k'].values():validate_lift_row(row,d['n_pairs'],role=='Dprime')

def read_aggregate(ref):
    require(set(ref)=={'path','sha256','run_id','role','variant','seed'},'file reference schema')
    role=ref['role'];require(role in ('mapping','prediction') and ref['variant'] in ('C1','C2'),'role/variant')
    seed=ref['seed'];require(seed is None if role=='mapping' else type(seed) is int and seed in (42,2024,2025),'seed')
    run='sid-coll-'+ref['variant'].lower()+('-sid_verify-v1' if role=='mapping' else '-verify'+str(seed)+'-v1')
    require(ref['run_id']==run,'run role binding')
    p=Path(ref['path']);require(p.is_absolute() and '..' not in p.parts and p.name==role+'-diagnostics.json' and p.parent.name=='out' and p.parents[1].name==run and p.parents[2].name=='runs','aggregate path binding')
    require(not any(x.is_symlink() for x in (p,*p.parents)),'symlink forbidden')
    require(type(ref['sha256']) is str and re.fullmatch('[0-9a-f]{64}',ref['sha256']),'SHA pin')
    require(p.is_file() and p.stat().st_size<=LIMIT,'aggregate file size')
    with p.open('rb') as f:raw=f.read(LIMIT+1)
    require(len(raw)<=LIMIT and hashlib.sha256(raw).hexdigest()==ref['sha256'],'SHA mismatch or size')
    v=decode(raw);(validate_mapping if role=='mapping' else validate_prediction)(v)
    return v

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stdin',action='store_true',required=True);parser.parse_args()
    raw=sys.stdin.buffer.read(65537);require(len(raw)<=65536,'request size');request=decode(raw);require(set(request)=={'schema','files'} and request['schema']=='coll-sid-aggregate-request-v1','request schema')
    refs=request['files'];require(type(refs) is list and len(refs)==8,'exactly eight aggregate refs')
    identities={(x['variant'],x['role'],x['seed']) for x in refs}
    require(identities=={(v,r,s) for v in ('C1','C2') for r,seeds in [('mapping',(None,)),('prediction',(42,2024,2025))] for s in seeds},'exact two mappings/six predictions')
    values=[{'source':ref,'aggregate':read_aggregate(ref)} for ref in refs]
    print(json.dumps({'schema':'coll-sid-aggregate-read-v1','status':'verified','read_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'test_read':False,'checkpoint_read':False,'prediction_rows_read':False,'files':values},sort_keys=True,allow_nan=False))

if __name__=='__main__':main()
