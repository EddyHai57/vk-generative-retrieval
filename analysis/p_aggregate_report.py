"""Read-only aggregate report for the completed P watch-positive runs."""
import datetime
import hashlib
import json
import math
import re
import sys
from pathlib import Path

SEEDS = (42, 2024, 2025)
SOURCE_COMMIT = '2c4b3ed2adff41ba298273a72d48d849fc9a028a'
ANCHOR_SOURCE_COMMIT='0ca61bc35ad68bd79b86103b54e470445cbab07f'
P3C_SHA='c89b6466c509628f6482b7d7dbc9f51c202ab665075caf580beed64bb3274b5f'
STATS_SHA='d7b1fc4c459e2d6f870abd6379059a78657065771ebed4404e17ed5d4151c4cf'
STATS_PATH='/workspace/vkgr-infra01-root/runs/p-watch-positive-stats-v1/logs/stdout.log'
SOURCE_MAP_SHA='c963dab1d8f654ebbf71e5d982bd8fa5483e2a71babef52d2e1e090ee07faab6'
SOURCE_MAP_PATH='/workspace/vkgr-p-watch-positive-train-ops-v6/source-map-train-v1.json'
NEW_DEPLOYMENT='/workspace/vkgr-p-watch-positive-train-v6'
ANCHOR_DEPLOYMENT='/workspace/vkgr-seqm1-p-v1'
ANCHOR_RUNS = {42:'p-seqm1-mean-s42',2024:'p-seqm1-mean-s2024',2025:'p-seqm1-mean-s2025'}
ANCHOR_MANIFESTS = {42:'bdb0e39d6abb721ac8de78dbdd11ebf2da2614b0e25292c3154ca096ec8cdc6e',2024:'8387b60206f116a3c4958e7a0a8ac597b829077b2013f8f0e0e8136a404ba9d8',2025:'783bbd9e101476383a711f62df6de54e139c156a903ccdc6643e57eacbc16ccb'}
ANCHOR_STDOUT = {42:'12ef936b63e42a289b3c1d61a04499c375cf55c3a4d65db3b6a534cc9b6bb18c',2024:'0422a9b5d324ff79b3d37c51fbfbff09240ed2bea9b98589dc3294f211aaf591',2025:'cd0eeb35e2f28038dc9374e9648a9d74174ca360f3c85f5ac07a3ecb7a8583ec'}
NEW_RUNS = {42:'p-watch-positive-s42',2024:'p-watch-positive-s2024',2025:'p-watch-positive-s2025'}
NEW_MANIFESTS = {42:'6b1edb9b1ea4676d4e6e166bfe4672a364e7e4581f81443bdb6a25fe1204a303',2024:'68b28d21bafd3595b36456e8448c5223aef8baa73837f0219e8cd991191d179c',2025:'27cf653099f83ddd0addedf77f70325e069e70278704205c960b41730ce80617'}
N=2520
REPORT_ID='p-watch-positive-report-v1'
HISTORY_SHA='8b929d10c41cebbac3976398ad6f48f74f4d4441675d6b054fd19ec0c95c8d43'
BASE_QUERY_SHA='9077a1c9495d494bc8ec209c0bf3305fd68db5bfcca24abcadc4a1e552e60a97'
QUERIES_SHA='c703bbcc3e3004a58a2a14c2c7c2420b51b3b7ae0c47f6b284d587e67819ab1a'
CATALOG_SHA='80201c574fad26bd29f2e2425a5ad98db8fe7c2a9be91e07d2fc6aedc48d47f3'
SLICES_SHA='6c534c35c01a79c43ed6387390e5753f33503365f3ea8c867e35a8657d62bd26'
EXPECTED_PATHS={'stats_stdout':STATS_PATH,'history':'/workspace/vkgr-infra01-root/runs/h2-prepare/out/manifest.json','queries':'/workspace/vkgr-infra01-root/runs/h2-prepare/out/validation-l50.json','slices':'/workspace/vkgr-infra01-root/runs/h2-prepare/out/train-slices.json','catalog':'/workspace/vkgr-p4b-20261007t-pipeline-03/prepared-v3/catalog.json'}
BIN_DENOMINATORS={'high':963,'mid':548,'low':155,'unsupported':231,'oov':623}
DURATION_BUCKETS=('1-10','11-20','21-40','41-60','61-120','121-255')


def require(ok, message):
    if not ok: raise ValueError(message)
def _reject_constant(value): raise ValueError('nonfinite JSON value')
def _unique_pairs(pairs):
    out={}
    for key,value in pairs:
        if key in out: raise ValueError('duplicate JSON field')
        out[key]=value
    return out
def _json(raw): return json.loads(raw,object_pairs_hook=_unique_pairs,parse_constant=_reject_constant)

def validate_sha256_map(values,label):
    require(type(values) is dict and all(type(value) is str and re.fullmatch('[a-f0-9]{64}',value) for value in values.values()),label+' must contain only 64 lowercase hex SHA256 values')
    return True

def validate_fixed_pins():
    require(re.fullmatch('[a-f0-9]{40}',SOURCE_COMMIT) and re.fullmatch('[a-f0-9]{40}',ANCHOR_SOURCE_COMMIT),'fixed source commits must be 40 lowercase hex')
    for label,pins in (('anchor manifests',ANCHOR_MANIFESTS),('anchor stdout',ANCHOR_STDOUT),('new manifests',NEW_MANIFESTS)):
        require(type(pins) is dict and set(pins)==set(SEEDS),label+' seed mapping differs')
        validate_sha256_map(pins,label)
    validate_sha256_map({'p3c':P3C_SHA,'stats':STATS_SHA,'source_map':SOURCE_MAP_SHA,'history':HISTORY_SHA,'base_query':BASE_QUERY_SHA,'queries':QUERIES_SHA,'catalog':CATALOG_SHA,'slices':SLICES_SHA},'fixed source pins')
    return True

def mcnemar_exact(b,c):
    from analysis import phase2_validation_read as reader
    return reader.mcnemar(b,c)

def validate_request(req):
    validate_fixed_pins()
    fields={'schema','action','source_commit','inputs','stdout_sha256'}
    require(type(req) is dict and set(req)==fields,'closed aggregate request required')
    require(req['schema']=='p-watch-positive-report-request-v1' and req['action']=='report','fixed report request required')
    require(req['source_commit']==SOURCE_COMMIT,'fixed P science source commit required')
    require(type(req['stdout_sha256']) is dict and set(req['stdout_sha256'])=={str(s) for s in SEEDS} and all(type(req['stdout_sha256'][str(s)]) is str and re.fullmatch('[a-f0-9]{64}',req['stdout_sha256'][str(s)]) for s in SEEDS),'three verified stdout pins required')
    expected={'history','queries','catalog','slices','stats_stdout','run_roots','run_manifests'}
    require(type(req['inputs']) is dict and set(req['inputs'])==expected,'fixed aggregate inputs required')
    fixed_sha={'history':HISTORY_SHA,'queries':QUERIES_SHA,'catalog':CATALOG_SHA,'slices':SLICES_SHA,'stats_stdout':STATS_SHA}
    for key in ('history','queries','catalog','slices'):
        ref=req['inputs'][key]
        require(type(ref) is dict and set(ref)=={'path','sha256'} and ref['path']==EXPECTED_PATHS[key] and ref['sha256']==fixed_sha[key],'fixed input path/SHA required')
    stats_ref=req['inputs']['stats_stdout']; require(type(stats_ref) is dict and set(stats_ref)=={'path','sha256'} and stats_ref=={'path':STATS_PATH,'sha256':STATS_SHA},'fixed accepted stats stdout ref required')
    roots=req['inputs']['run_roots']
    require(type(roots) is dict and roots=={'new':'/workspace/vkgr-infra01-root','anchor':'/workspace/vkgr-infra01-root'},'fixed new/anchor run roots required')
    manifests=req['inputs']['run_manifests']
    require(type(manifests) is dict and set(manifests)=={'42','2024','2025'} and all(manifests[str(s)]==NEW_MANIFESTS[s] for s in SEEDS),'fixed new P manifests required')
    return req

def aggregate_seed(p_hits, anchor_hits, bins, in_catalog):
    n=len(p_hits)
    require(n==N and len(anchor_hits)==n and len(bins)==n and len(in_catalog)==n,'query coverage mismatch')
    require(sum(bool(v) for v in in_catalog)==1897 and all((b=='oov')==(not bool(c)) for b,c in zip(bins,in_catalog)),'frozen catalog/OOV split differs')
    require(all(x in (0,1) for x in p_hits+anchor_hits),'binary hit vectors required')
    b=sum(p and not a for p,a in zip(p_hits,anchor_hits)); c=sum(a and not p for p,a in zip(p_hits,anchor_hits))
    delta=sum(p_hits)-sum(anchor_hits)
    bybin={name:{'denominator':sum(1 for x in bins if x==name),'p_hits':sum(p for p,x in zip(p_hits,bins) if x==name),'anchor_hits':sum(a for a,x in zip(anchor_hits,bins) if x==name)} for name in BIN_DENOMINATORS}
    require({k:v['denominator'] for k,v in bybin.items()}==BIN_DENOMINATORS,'frozen heat-bin denominators differ')
    in_idx=[i for i,v in enumerate(in_catalog) if v]; oov_idx=[i for i,v in enumerate(in_catalog) if not v]
    require(not any(p_hits[i] or anchor_hits[i] for i in oov_idx),'OOV structural zero violated')
    for value in bybin.values(): value['delta_hits']=value['p_hits']-value['anchor_hits']
    return {'p_hits':sum(p_hits),'anchor_hits':sum(anchor_hits),'denominator':n,'delta_hits':delta,'in_catalog':{'denominator':len(in_idx),'p_hits':sum(p_hits[i] for i in in_idx),'anchor_hits':sum(anchor_hits[i] for i in in_idx),'delta_hits':sum(p_hits[i]-anchor_hits[i] for i in in_idx)},'oov':{'denominator':len(oov_idx),'p_hits':sum(p_hits[i] for i in oov_idx),'anchor_hits':sum(anchor_hits[i] for i in oov_idx),'delta_hits':sum(p_hits[i]-anchor_hits[i] for i in oov_idx)},'heat_bins':bybin,'only_new':b,'only_control':c,'mcnemar_exact_two_sided':mcnemar_exact(b,c)}

def pooled_exact(seeds):
    b=sum(x['only_new'] for x in seeds.values()); c=sum(x['only_control'] for x in seeds.values())
    return {'only_new':b,'only_control':c,'mcnemar_exact_two_sided':mcnemar_exact(b,c),'query_repeats_correlated_across_seeds':True,'pooled_queries_independent':False}

def mechanical_checks(seeds):
    require(set(seeds)=={str(s) for s in SEEDS},'three seed summaries required')
    deltas=[seeds[str(s)]['delta_hits'] for s in SEEDS]
    D=sum(deltas)/3
    kplus=sum(d>=6 for d in deltas)
    exact_ps=[seeds[str(s)]['mcnemar_exact_two_sided'] for s in SEEDS]
    require(all(type(p) in (int,float) and math.isfinite(p) and 0<=p<=1 for p in exact_ps),'per-seed exact McNemar p-values must be finite probabilities')
    significant=sum(p<.05 for p in exact_ps)
    pooled=pooled_exact(seeds)
    return {'seed_exact_p_lt_005_count':significant,'at_least_2_seed_exact_p_lt_005':significant>=2,'D_mean_hits':D,'D_le_minus_6':D<=-6,'D_ge_6':D>=6,'kplus':kplus,'kplus_ge_2':kplus>=2,'p_pool':pooled['mcnemar_exact_two_sided'],'p_pool_lt_005':pooled['mcnemar_exact_two_sided']<.05}

def frozen_training_statistics(stats):
    require(type(stats.get('buckets')) is dict and set(stats['buckets'])==set(DURATION_BUCKETS),'accepted stats duration table differs')
    table=[]
    for duration in DURATION_BUCKETS:
        values=stats['buckets'][duration]
        require(values['supported_aux']+values['unsupported_excluded']==values['after_like_dislike_filters'],'accepted duration table count conservation differs')
        table.append({'duration':duration,'n_b':values['n_b'],'t_b':values['threshold'],
                      'ratio_screened_count':values['ratio_screened_count'],
                      'after_like_dislike':values['after_like_dislike_filters'],
                      'supported_aux':values['supported_aux'],'unsupported_excluded':values['unsupported_excluded']})
    require(sum(x['supported_aux'] for x in table)==stats['N_aux'] and sum(x['after_like_dislike'] for x in table)==stats['N_aux_candidates'],'accepted train-only bucket totals differ')
    return table

def _load_stats(ref):
    p=Path(ref['path']); require(p.is_absolute() and p.suffix=='.log' and '..' not in p.parts and not any(x.is_symlink() for x in (p,*p.parents)),'bounded stats stdout path required')
    require(p.is_file() and p.stat().st_size<=1024*1024,'bounded accepted stats stdout required')
    raw=p.read_bytes(); require(hashlib.sha256(raw).hexdigest()==STATS_SHA,'accepted stats stdout SHA differs')
    stats=_json(raw)
    require(stats.get('schema')=='p-watch-positive-train-only-report-v1' and stats.get('run_id')=='p-watch-positive-stats-v1' and stats.get('engineering') is False and stats.get('synthetic') is False,'accepted train-only stats identity differs')
    gate=stats.get('gate',{})
    require(gate.get('week_ge_25_in_auxiliary') is False and gate.get('N_aux_positive') is True and gate.get('invalid_ratio_fraction_at_or_below_1_percent') is True and gate.get('stop_reasons')==[],'accepted train-only stats gates differ')
    return stats

def _validate_exclusion(manifest,stats):
    watch=manifest.get('watch_positive',{})
    require(watch.get('stats_stdout_sha256')==STATS_SHA,'manifest stats stdout binding differs')
    require(watch.get('N_like')==stats.get('N_like') and watch.get('N_aux')==700633,'manifest watch pool counts differ from accepted stats contract')
    diag=watch.get('watch_aux_exclusion')
    fields={'raw_N_aux','empty_history_excluded','N_aux','raw_pool_sha256','filtered_pool_sha256','excluded_by_duration_bucket','excluded_by_heat_bin','fully_excluded_item_count','raw_bucket_counts_verified','s0_train_slices_sha256'}
    require(type(diag) is dict and set(diag)==fields,'closed watch exclusion evidence required')
    require(diag['s0_train_slices_sha256']==SLICES_SHA,'watch exclusion S0 train-slices pin differs')
    require(diag['raw_N_aux']==stats['N_aux'] and diag['N_aux']==watch['N_aux'] and diag['empty_history_excluded']==stats['empty_history_aux_count'],'watch exclusion/stat counts differ')
    require(diag['raw_bucket_counts_verified'] is True and sum(diag['excluded_by_duration_bucket'].values())==diag['empty_history_excluded'] and sum(diag['excluded_by_heat_bin'].values())==diag['empty_history_excluded'],'watch exclusion counts do not conserve')
    require(type(diag['fully_excluded_item_count']) is int and diag['fully_excluded_item_count']>=0,'watch fully-excluded item count invalid')
    return diag

def _source_map(reader):
    path=Path(SOURCE_MAP_PATH); ledger,_=reader.read(path,SOURCE_MAP_SHA)
    require(ledger.get('schema')=='vkgr-pinned-v1' and ledger.get('base_commit')==SOURCE_COMMIT and ledger.get('target_commit')==SOURCE_COMMIT and ledger.get('overrides')=={} and type(ledger.get('files')) is dict,'fixed watch source ledger identity differs')
    require(all(type(k) is str and type(v) is str and re.fullmatch('[a-f0-9]{64}',v) for k,v in ledger['files'].items()) and 'docs/p3c_validation.md' not in ledger['files'],'fixed watch source ledger file pins invalid')
    return ledger

def _validate_run_identity(reader,root,rid,role,runmeta,trainjob,ledger):
    commit=SOURCE_COMMIT if role=='new' else ANCHOR_SOURCE_COMMIT
    cwd=NEW_DEPLOYMENT if role=='new' else ANCHOR_DEPLOYMENT
    require(runmeta.get('run_id')==rid and runmeta.get('schema')=='vkgr-run-v1' and runmeta.get('kind')=='real' and runmeta.get('cwd')==cwd,'original run metadata identity differs')
    require(runmeta.get('git')=={'commit':commit,'dirty':True,'dirty_files':[{'path':'docs/p3c_validation.md','sha256':P3C_SHA}]},'original science source/p3c dirty binding differs')
    p3c={'docs/p3c_validation.md':P3C_SHA}
    if role=='new':
        expected={**ledger['files'],**p3c}
        overlay={'base_commit':SOURCE_COMMIT,'target_commit':SOURCE_COMMIT,'overrides':{},'pinned_manifest_sha256':SOURCE_MAP_SHA}
        pinned_manifest=SOURCE_MAP_PATH
    else:
        expected=p3c; overlay=None; pinned_manifest=None
    require(runmeta.get('pinned_files')==expected and runmeta.get('overlay')==overlay,'original run pinned source ledger differs')
    require(trainjob.get('schema')=='vkgr-queue-job-v1' and trainjob.get('kind')=='real' and trainjob.get('run_id')==rid and trainjob.get('cwd')==cwd and trainjob.get('argv')==runmeta.get('argv') and trainjob.get('pinned_files')==['docs/p3c_validation.md='+P3C_SHA] and trainjob.get('pinned_manifest')==pinned_manifest,'original claimed job/source ledger binding differs')
    return commit

def _load_ref(ref, roots):
    p=Path(ref['path'])
    require(p.is_absolute() and '..' not in p.parts and p.suffix=='.json','absolute JSON input required')
    root=next((Path(x) for x in roots if p==Path(x) or Path(x) in p.parents),None)
    require(root is not None,'input outside registered roots')
    require(not any(x.is_symlink() for x in (p,*p.parents)),'input path symlink refused')
    require(p.is_file() and p.stat().st_size<=32*1024*1024,'bounded input JSON required')
    raw=p.read_bytes(); require(hashlib.sha256(raw).hexdigest()==ref['sha256'],'input SHA differs')
    return _json(raw)

def _bins(queries,slices,catalog):
    require(slices.get('popularity_thresholds',{}).get('median')==1 and slices.get('popularity_thresholds',{}).get('p90')==11,'frozen train-only thresholds differ')
    counts={int(k):v for k,v in slices.get('item_like_counts',{}).items()}; cat=set(catalog); out=[]
    for q in queries:
        target=q.get('target'); require(type(target) is int and type(q.get('oov')) is bool and q['oov']==(target not in cat),'query/catalog identity differs')
        if q['oov']: out.append('oov')
        else:
            c=counts.get(target,0); out.append('unsupported' if c==0 else 'low' if c<=1 else 'mid' if c<=11 else 'high')
    require({k:out.count(k) for k in BIN_DENOMINATORS}==BIN_DENOMINATORS,'frozen heat denominators differ')
    return out,[not q['oov'] for q in queries]


def _precheck_run_files(root,rid):
    root=Path(root); rd=root/'runs'/rid; vd=root/'runs'/(rid+'-verify')
    for p in (rd,rd/'terminal.json',rd/'out/manifest.json',rd/'run.json',vd,vd/'terminal.json',vd/'logs/stdout.log'):
        require(root==p or root in p.parents,'run receipt path escaped fixed root')
        cur=p
        while True:
            require(not cur.is_symlink(),'run receipt symlink refused')
            if cur==root: break
            cur=cur.parent
    require(rd.is_dir() and (rd/'terminal.json').is_file() and (rd/'out/manifest.json').is_file() and (rd/'run.json').is_file(),'original train receipts missing')
    require(vd.is_dir() and (vd/'terminal.json').is_file() and (vd/'logs/stdout.log').is_file(),'original verify receipts missing')
    for identity in (rid,rid+'-verify'):
        for status in ('pending','claimed','rejected'):
            folder=root/'queue'/status
            require(folder.is_dir() and not any(x.is_symlink() for x in (folder,*folder.parents)),'queue receipt directory unsafe')
            entries=sorted(folder.glob('*-'+identity+'.json'))
            for item in entries: require(not any(x.is_symlink() for x in (item,*item.parents)),'queue receipt symlink refused')
            require(len(entries)==1 if status=='claimed' else not entries,'original queue claim state differs')

def preflight_report(req,request_sha):
    req=validate_request(req)
    from analysis import phase2_validation_read as reader
    inp=req['inputs']
    roots={Path(x) for x in inp['run_roots'].values()}
    fixed={Path(inp[k]['path']) for k in ('history','queries','catalog','slices')}|{Path(SOURCE_MAP_PATH)}
    original_read=reader.read
    def guarded_read(path,pin=None):
        lexical=Path(path)
        require(lexical.is_absolute() and '..' not in lexical.parts and lexical.suffix=='.json' and not any(w in lexical.name.lower() for w in ('test','checkpoint','embedding')),'registered aggregate JSON path required')
        require(lexical in fixed or any(root==lexical or root in lexical.parents for root in roots),'unregistered archive path refused')
        require(not any(x.is_symlink() for x in (lexical,*lexical.parents)),'archive symlink path refused')
        return original_read(path,pin)
    reader.read=guarded_read
    history=_load_ref(inp['history'],[Path(inp['history']['path']).parent])
    queries=_load_ref(inp['queries'],[Path(inp['queries']['path']).parent])
    catalog=_load_ref(inp['catalog'],[Path(inp['catalog']['path']).parent])
    slices=_load_ref(inp['slices'],[Path(inp['slices']['path']).parent])
    require(isinstance(queries,list) and len(queries)==N and isinstance(catalog,list) and len(catalog)==120415 and len(set(catalog))==len(catalog),'frozen query/catalog shape differs')
    require(history.get('artifacts',{}).get('validation-l50.json')==inp['queries']['sha256'] and history.get('artifacts',{}).get('train-slices.json')==inp['slices']['sha256'],'history artifact pins differ')
    require(history.get('binding',{}).get('query_sha')==BASE_QUERY_SHA and reader.plain_digest(catalog)==history.get('catalog_sha256') and inp['catalog']['sha256']==history.get('binding',{}).get('catalog_sha'),'frozen base-query/catalog binding differs')
    bins,_=_bins(queries,slices,catalog)
    stats=_load_stats(inp['stats_stdout']); ledger=_source_map(reader)
    records={}; exclusions={}
    for role,ids,manifests,stdouts,commit,cwd in (('new',NEW_RUNS,NEW_MANIFESTS,{int(k):v for k,v in req['stdout_sha256'].items()},SOURCE_COMMIT,NEW_DEPLOYMENT),('anchor',ANCHOR_RUNS,ANCHOR_MANIFESTS,ANCHOR_STDOUT,ANCHOR_SOURCE_COMMIT,ANCHOR_DEPLOYMENT)):
        root=Path(inp['run_roots'][role]); require(root.is_absolute() and not any(x.is_symlink() for x in (root,*root.parents)),'run root path unsafe')
        reader.ROOT=root
        for seed in SEEDS:
            rid=ids[seed]; _precheck_run_files(root,rid); state=reader.inspect_run(rid)
            require(state.get('verified') is True and state.get('manifest_sha256')==manifests[seed],'run not raw verified against frozen manifest')
            require((state.get('terminal') or {}).get('status')=='succeeded','original training terminal differs')
            q=state.get('queue',{}); require(len(q.get('claimed',[]))==1 and not q.get('pending') and not q.get('rejected'),'original train queue/job binding differs')
            runmeta_path=root/'runs'/rid/'run.json'; runmeta,runmeta_sha=reader.read(runmeta_path)
            trainjob_path=root/'queue'/'claimed'/q['claimed'][0]; trainjob,_=reader.read(trainjob_path)
            _validate_run_identity(reader,root,rid,role,runmeta,trainjob,ledger)
            manifest,_=reader.read(root/'runs'/rid/'out/manifest.json',manifests[seed])
            if role=='new': exclusions[rid]=_validate_exclusion(manifest,stats)
            v=state.get('verify',{}); require(v.get('verified') is True and len(v.get('jobs',[]))==1 and v['jobs'][0].get('job_receipt',{}).get('target_binding_matches') is True,'original verify job/manifest binding differs')
            raw=root/'runs'/(rid+'-verify')/'logs/stdout.log'
            require(raw.is_file() and not any(x.is_symlink() for x in (raw,*raw.parents)) and hashlib.sha256(raw.read_bytes()).hexdigest()==stdouts[seed],'raw verifier stdout pin differs')
            lines=[]
            for line in raw.read_text().splitlines():
                try:
                    val=json.loads(line)
                    if isinstance(val,dict): lines.append(val)
                except ValueError: pass
            require(any(x.get('status')=='verified' for x in lines),'raw verify stdout has no verified record')
            records[rid]={'manifest_sha256':manifests[seed],'verify_stdout_sha256':stdouts[seed],'source_commit':commit,'source_ledger_sha256':SOURCE_MAP_SHA if role=='new' else None,'verified':True}
    for seed in SEEDS:
        require(exclusions[NEW_RUNS[seed]]==exclusions[NEW_RUNS[SEEDS[0]]],'watch exclusion differs by seed')
    return {'schema':'p-watch-positive-report-preflight-v1','status':'preflight_ok','run_id':'p-watch-positive-report-preflight-v1','source_commit':_repo_commit(),'request_sha256':request_sha,'engineering':False,'test_read':False,'checkpoint_read':False,'embedding_read':False,'model_run':False,'validation_metrics_read':False,'query_count':len(queries),'stats_stdout_sha256':STATS_SHA,'source_ledger_sha256':SOURCE_MAP_SHA,'heat_bin_counts':{name:bins.count(name) for name in BIN_DENOMINATORS},'runs':records}


def manifest_watch_summary(stats,diag,summaries):
    n_like=stats['N_like']; n_aux_raw=stats['N_aux']; n_aux=diag['N_aux']
    require(type(n_like) is int and type(n_aux_raw) is int and type(n_aux) is int and n_like>0 and n_aux>0 and n_like==stats['N_like_train_examples'],'accepted watch pool counts invalid')
    require(n_aux_raw==716194 and diag['empty_history_excluded']==15561 and n_aux==700633 and diag['fully_excluded_item_count']==86,'frozen watch exclusion self-check differs')
    return {'N_like':n_like,'N_aux_raw':n_aux_raw,'N_aux_filtered':n_aux,'draw_per_epoch':min(n_like,n_aux),
            'empty_history_excluded':diag['empty_history_excluded'],'items_removed_from_training_pool':diag['fully_excluded_item_count'],
            'excluded_by_duration_bucket':diag['excluded_by_duration_bucket'],'excluded_by_heat_bin':diag['excluded_by_heat_bin'],
            'train_only_threshold_table':frozen_training_statistics(stats),'stats_stdout_sha256':STATS_SHA,
            'per_seed_training':{s:{r:{k:summaries[s][r][k] for k in ('best_epoch','stop_epoch','total_steps')} for r in ('new','anchor')} for s in (str(x) for x in SEEDS)}}

def execute_report(req,request_sha):
    req=validate_request(req)
    # Validate every seed's original verified receipts before any evaluation archive is read.
    preflight_report(req,request_sha)
    from analysis import phase2_validation_read as reader
    inp=req['inputs']; roots={Path(v) for v in inp['run_roots'].values()}
    fixed={Path(inp[k]['path']) for k in ('history','queries','catalog','slices')}|{Path(SOURCE_MAP_PATH)}
    original_read=reader.read
    def guarded_read(path,pin=None):
        lexical=Path(path)
        require(lexical.is_absolute() and '..' not in lexical.parts and lexical.suffix=='.json' and not any(w in lexical.name.lower() for w in ('test','checkpoint','embedding')),'registered aggregate JSON path required')
        require(lexical in fixed or any(root==lexical or root in lexical.parents for root in roots),'unregistered archive path refused')
        require(not any(x.is_symlink() for x in (lexical,*lexical.parents)),'archive symlink path refused')
        return original_read(path,pin)
    reader.read=guarded_read
    history=_load_ref(inp['history'],[Path(inp['history']['path']).parent]); queries=_load_ref(inp['queries'],[Path(inp['queries']['path']).parent]); catalog=_load_ref(inp['catalog'],[Path(inp['catalog']['path']).parent]); slices=_load_ref(inp['slices'],[Path(inp['slices']['path']).parent])
    require(isinstance(queries,list) and len(queries)==N and isinstance(catalog,list) and len(catalog)==120415 and len(set(catalog))==len(catalog),'frozen query/catalog shape differs')
    require(history.get('artifacts',{}).get('validation-l50.json')==inp['queries']['sha256'] and history.get('artifacts',{}).get('train-slices.json')==inp['slices']['sha256'],'history artifact pins differ')
    require(history.get('binding',{}).get('query_sha')==BASE_QUERY_SHA and reader.plain_digest(catalog)==history.get('catalog_sha256') and inp['catalog']['sha256']==history.get('binding',{}).get('catalog_sha'),'frozen base-query/catalog binding differs')
    stats=_load_stats(inp['stats_stdout']); ledger=_source_map(reader)
    hist=dict(history); hist['_base_queries_canonical_digest']=history['binding']['query_sha']; hist['_g_catalog_digest']=reader.digest(catalog)
    bins,in_catalog=_bins(queries,slices,catalog); details={}; summaries={}; evidence={}; exclusions={}
    roles=(('new',NEW_RUNS,NEW_MANIFESTS,{int(k):v for k,v in req['stdout_sha256'].items()}),('anchor',ANCHOR_RUNS,ANCHOR_MANIFESTS,ANCHOR_STDOUT))
    for role,ids,manifests,stdouts in roles:
        commit=SOURCE_COMMIT if role=='new' else ANCHOR_SOURCE_COMMIT
        root=Path(inp['run_roots'][role]); require(root.is_absolute() and not any(x.is_symlink() for x in (root,*root.parents)),'run root must be absolute nonsymlink')
        reader.ROOT=root
        for seed in SEEDS:
            rid=ids[seed]
            _precheck_run_files(root,rid); state=reader.inspect_run(rid)
            require(state.get('verified') is True and state.get('manifest_sha256')==manifests[seed],'run not raw verified against frozen manifest')
            raw=root/'runs'/(rid+'-verify')/'logs/stdout.log'
            require(raw.is_file() and not raw.is_symlink() and hashlib.sha256(raw.read_bytes()).hexdigest()==stdouts[seed],'raw verifier stdout pin differs')
            records=[]
            for line in raw.read_text().splitlines():
                try:
                    val=json.loads(line)
                    if isinstance(val,dict): records.append(val)
                except ValueError: pass
            require(any(x.get('status')=='verified' for x in records),'raw verify stdout has no verified record')
            qstate=state.get('queue',{})
            require(len(qstate.get('claimed',[]))==1 and not qstate.get('pending') and not qstate.get('rejected'),'original train queue/job binding differs')
            runmeta_path=root/'runs'/rid/'run.json'; runmeta,_=reader.read(runmeta_path)
            trainjob_path=root/'queue'/'claimed'/qstate['claimed'][0]; trainjob,_=reader.read(trainjob_path)
            _validate_run_identity(reader,root,rid,role,runmeta,trainjob,ledger)
            vstate=state.get('verify',{})
            require(vstate.get('verified') is True and len(vstate.get('jobs',[]))==1 and vstate['jobs'][0].get('job_receipt',{}).get('target_binding_matches') is True,'original verify job/manifest binding differs')
            manifest,_=reader.read(root/'runs'/rid/'out/manifest.json',manifests[seed])
            if role=='new': exclusions[rid]=_validate_exclusion(manifest,stats)
            data=reader.load_archive(rid,queries,hist,state)
            require(sum(data['hits'])==data['summary']['hits'],'aggregate hits differ from archive summary')
            training={'best_epoch':data['summary']['best_epoch'],'stop_epoch':data['summary']['stop_epoch'],'total_steps':manifest.get('steps')}
            require(type(training['total_steps']) is int and training['total_steps']>0,'manifest total steps missing')
            details.setdefault(str(seed),{})[role]=data['hits']; summaries.setdefault(str(seed),{})[role]={**data['summary'],**training}; evidence[rid]={'manifest_sha256':manifests[seed],'verify_stdout_sha256':stdouts[seed],'verified':True,'source_commit':commit,'source_ledger_sha256':SOURCE_MAP_SHA if role=='new' else None,'p3c_data_pin_sha256':P3C_SHA}
    for seed in SEEDS: require(exclusions[NEW_RUNS[seed]]==exclusions[NEW_RUNS[SEEDS[0]]],'watch exclusion differs by seed')
    watch=manifest_watch_summary(stats,exclusions[NEW_RUNS[SEEDS[0]]],summaries)
    seeds={str(s):aggregate_seed(details[str(s)]['new'],details[str(s)]['anchor'],bins,in_catalog) for s in SEEDS}
    pooled=pooled_exact(seeds)
    mechanical=mechanical_checks(seeds)
    for s,expected,stop in zip(SEEDS,(466,481,471),(15,10,14)):
        require(summaries[str(s)]['anchor']['hits']==expected,'fixed anchor hits differ')
        require(summaries[str(s)]['anchor']['stop_epoch']==stop and stop<100,'prechecked M1-mean stop epoch differs')
    return {'schema':'p-watch-positive-report-v1','status':'read_only_complete','test_read':False,'checkpoint_read':False,'embedding_read':False,'model_run':False,'submit':False,'seeds':seeds,'pooled_exact':pooled,'mechanical':mechanical,'summaries':summaries,'run_evidence':evidence,'inputs_sha256':{k:v['sha256'] for k,v in inp.items() if isinstance(v,dict) and 'sha256' in v},'read_artifacts_sha256':reader.READS,'watch_positive':watch,'limitations':{'query_sets_repeat_across_seeds':True,'pooled_query_test_assumes_independence':False,'about_double_steps_per_epoch_is_a_registered_confound':True,'anchor_stop_epochs_below_100_prechecked':[15,10,14],'registered_boundaries':['D-1同类偏离','约0.2%内容回退','623 OOV结构性零命中','早停右删失'],'oov_structural_zero_hits':True,'latency':'UNKNOWN; not remeasured','interpretation':'mechanical counts only; no efficacy, harm, anchor, or promotion conclusion'},'read_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}

def _repo_commit():
    import subprocess
    return subprocess.check_output(['git','-C',str(Path(__file__).resolve().parents[1]),'rev-parse','HEAD'],text=True).strip()

def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument('--request',required=True); parser.add_argument('--request-sha',required=True); parser.add_argument('--preflight',action='store_true'); args=parser.parse_args(argv)
    path=Path(args.request); require(path.is_absolute() and not any(x.is_symlink() for x in (path,*path.parents)) and path.is_file() and path.stat().st_size<=1024*1024,'bounded nonsymlink request file required'); raw=path.read_bytes(); require(hashlib.sha256(raw).hexdigest()==args.request_sha,'request SHA differs'); req=_json(raw); validate_request(req)
    if args.preflight: out=preflight_report(req,args.request_sha)
    else:
        out=execute_report(req,args.request_sha)
        out['run_id']=REPORT_ID
        out['request_sha256']=args.request_sha
        out['source_commit']=_repo_commit()
        out['report_script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        out['science_source_commit']=SOURCE_COMMIT
    print(json.dumps(out,sort_keys=True,allow_nan=False)); return 0

if __name__=='__main__':
    raise SystemExit(main())
