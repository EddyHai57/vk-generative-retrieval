"""P4F validation-only consumer CLI; immutable inputs and recoverable shards."""
import argparse
import hashlib
from pathlib import Path
import sys
import numpy as np
from .retrieval_ann import (canonical, checked, file_hash, read_json, write_json,
                            validate_binding, export_vectors, build_index, consumer_identity, audit_index, require_consumer)
from .retrieval_fusion import weight_candidates, weighted_rrf, redistribute_quota, select_weights
from .baselines import metrics

SOURCE = '573c1bdc586f27ead094bf20d497ae16a447264faedbaf421302e230b147a508'
PLAN = '8d8e2a657824743f02cb9f7360dfeda9b120b7b26f81a9deec6ac49040257f3d'


def validate_producer_chain(m):
    if m['binding']['source_sha256'] != SOURCE or m['binding'].get('plan_sha256') != PLAN:
        raise ValueError('frozen scientific source/plan mismatch')
    refs=m.get('producer_manifests',{})
    shapes={'prepared':('P4B','prepare','success'),'baselines':('P4C','baselines','success'),
            'core_selection':('P4D','core_selection','success'),'core_run':('P4D','core_train','complete')}
    if set(refs)!=set(shapes):raise ValueError('complete producer chain required')
    producers={role:read_json(checked(ref['path'],ref['sha256'])) for role,ref in refs.items()}
    for role,(task,kind,status) in shapes.items():
        p=producers[role]
        if (p.get('task_id'),p.get('kind'),p.get('status'))!=(task,kind,status):raise ValueError('producer role/schema mismatch')
    prepared=producers['prepared'];baseline=producers['baselines'];selection=producers['core_selection'];run=producers['core_run']
    for filename,role in (('catalog.json','catalog'),('content.npy','content'),('idmap.json','idmap'),('support.json','supported_indices')):
        if prepared['artifacts'][filename]!=m['files'][role]['sha256']:raise ValueError('prepared artifact pin mismatch')
    for p in (baseline,run):
        b=p['binding']
        if (b['prepared_manifest_sha256']!=refs['prepared']['sha256'] or b['prepared_artifacts']!=prepared['artifacts'] or
                b['prepared_producer']!=prepared['binding']['consumer'] or b['catalog_sha256']!=m['binding']['catalog_sha256'] or
                b.get('validation_query_sha256',b.get('query_sha256'))!=m['binding']['query_sha256']):
            raise ValueError('producer input lineage mismatch')
    if baseline['artifacts']['graph.json']!=m['files']['graph']['sha256']:raise ValueError('baseline graph pin mismatch')
    if not any(p['role']=='validation-queries.json' and p['sha256']==m['binding']['query_sha256'] for p in prepared['binding']['query_pins']):
        raise ValueError('prepared validation query pin mismatch')
    b=run['binding']
    if m['checkpoint_binding']!=b or m['config']!=b['config'] or b['config']['seed']!=42:
        raise ValueError('historical checkpoint config/binding mismatch')
    chosen=selection['selected_configs'][selection['best_architecture']]
    if (selection['representative_seed']!=42 or selection['best_architecture']!=m['config']['architecture'] or
            chosen['config']!={k:m['config'][k] for k in ('architecture','learning_rate','temperature','seed')} or
            chosen['manifest_sha256']!=refs['core_run']['sha256'] or chosen['best']!=run['best'] or
            run['best']['sha256']!=m['binding']['checkpoint_sha256']):
        raise ValueError('frozen core selection/checkpoint mismatch')
    shared={key:value for key,value in b.items() if key not in ('config','runtime_config_sha256','architecture','selection_sha256')}
    shared['shared_batch_size']=b['config']['batch_size']
    if selection['input_binding']!=shared:raise ValueError('selection shared producer input mismatch')


def load_input(path, sha, require_routes=True, *, consumer_release=None):
    consumer = consumer_identity(consumer_release=consumer_release) if consumer_release is not None else None
    path = checked(path,sha); m = read_json(path)
    if m.get('kind') != 'p4f_input' or m.get('split') != 'validation':
        raise ValueError('only pinned validation inputs allowed; test gate closed')
    validate_binding(m['binding'])
    if not m.get('synthetic'):
        validate_producer_chain(m)
    required = {'catalog','idmap','queries','rankings','supported_indices','content','checkpoint'}
    if not m.get('synthetic'):required=required-{'rankings'}|{'graph'}
    if set(m['files']) != required:raise ValueError('input roles mismatch')
    paths = {}
    for role,ref in m['files'].items():
        p = Path(ref['path'])
        if not p.is_absolute():
            if '..' in p.parts:raise ValueError('input path escape')
            p = path.parent/p
        paths[role] = checked(p,ref['sha256'])
    for role,key in (('catalog','catalog_sha256'),('idmap','idmap_sha256'),('queries','query_sha256'),('checkpoint','checkpoint_sha256')):
        if m['files'][role]['sha256'] != m['binding'][key]:raise ValueError('scientific pin mismatch: '+role)
    catalog=read_json(paths['catalog']);queries=read_json(paths['queries'])
    queries=[{**q,'query_id':q.get('query_id',q['user_id'])} for q in queries]
    if m.get('synthetic'):
        rankings=read_json(paths['rankings'])
    elif require_routes:
        ref=m['route_export'];ex=read_json(checked(ref['path'],ref['sha256']))
        if ex['kind']!='p4f_export' or ex['binding']!=m['binding']:raise ValueError('route export binding mismatch')
        if consumer is not None or ex.get('consumer',{}).get('identity_kind') == 'verified_deployment_release':
            require_consumer(ex,consumer or consumer_identity())
        routing=read_json(checked(Path(ref['path']).parent/'routes.json',ex['files']['routes.json']))
        eligible=[q for q in queries if q['history']]
        if routing['query_records']!=eligible or routing['graph_sha256']!=m['files']['graph']['sha256'] or routing['popularity']!=m['popularity']:
            raise ValueError('cached routes query/graph/Pop lineage mismatch')
        lookup=dict(zip(routing['query_ids'],routing['rankings']))
        if len(lookup)!=len(eligible):raise ValueError('route query duplicate/loss')
        rankings=[lookup[q['query_id']] if q['history'] else [m['popularity'][:134]]*4 for q in queries]
        m['route_fill_counts']={'query_ids':routing['query_ids'],'top100':routing['itemcf_fill_top100'],'top134':routing['itemcf_fill_top134']}
    else:
        rankings=[[m['popularity'][:134]]*4 for q in queries]
    if not catalog or catalog != sorted(set(catalog)) or any(type(i) is not int or i<0 for i in catalog):raise ValueError('invalid catalog')
    ids=[q['query_id'] for q in queries]
    if not ids or len(ids)!=len(set(ids)) or any(type(q) is not int or q<0 for q in ids):raise ValueError('duplicate/invalid queries')
    if len(rankings)!=len(queries):raise ValueError('route/query shape mismatch')
    for q,routes in zip(queries,rankings):
        if type(q['target']) is not int or q['oov'] != (q['target'] not in catalog) or type(q['oov']) is not bool:
            raise ValueError('query OOV mismatch')
        if any(type(i) is not int or i<0 for i in q['history']) or len(q['history'])>50:raise ValueError('invalid history')
        if len(routes)!=4:raise ValueError('four routes required')
        for ranks in routes:
            if (len(ranks)!=min(134,len(catalog)) or len(set(ranks))!=len(ranks) or
                    any(type(i) is not int or i not in catalog for i in ranks)):
                raise ValueError('route quota/ID mismatch; need top134 for fair leave-one-out')
    popularity=m['popularity']
    if len(popularity)!=len(set(popularity)) or set(popularity)!=set(catalog):raise ValueError('invalid Pop ranking')
    map_record=read_json(paths['idmap'])
    if not m.get('synthetic'):
        if map_record['items']!=catalog or (map_record['pad'],map_record['start'])!=(0,1):raise ValueError('prepared idmap mismatch')
        idmap={i:j+1 for j,i in enumerate(catalog)}
    else:idmap={int(k):v for k,v in map_record.items()}
    if idmap!={i:j+1 for j,i in enumerate(catalog)}:raise ValueError('idmap mismatch')
    if m['config']['seed']!=42 or m['config']['batch_size']!=512:raise ValueError('frozen core representative mismatch')
    return m,paths,catalog,queries,rankings,idmap


def prediction(routes, weights, quotas=None):
    quotas = quotas or {r:100 for r in range(4)}
    return weighted_rrf([ranks[:quotas.get(r,0)] for r,ranks in enumerate(routes)],weights)


def fuse(m,catalog,queries,rankings,output,input_sha, *, consumer_release=None):
    consumer = consumer_identity(consumer_release=consumer_release)
    eligible=[i for i,q in enumerate(queries) if q['history']]
    if not eligible:raise ValueError('no eligible validation queries')
    qs=[queries[i] for i in eligible];rows=[]
    for weights in weight_candidates():
        result=metrics([prediction(rankings[i],weights) for i in eligible],qs,catalog,(10,20,50))
        rows.append({'weights':weights,**result['all']})
    selected=select_weights(rows)
    reports={};coverage={}
    for label,weights in [('equal',(1,1,1,1)),('selected',selected)]+[(f'single-{r}',tuple(int(i==r) for i in range(4))) for r in range(4)]:
        preds=[prediction(rankings[i],weights) for i in eligible]
        reports[label]=metrics(preds,qs,catalog,(10,20,50))
        covered=set(item for row in preds for item in row)
        coverage[label]={'count':len(covered),'fraction':len(covered)/len(catalog),'scope':'distinct items in final top50 / whole catalog'}
    leave=[]
    for removed in range(4):
        active=[r for r in range(4) if r!=removed];weights=tuple(0 if r==removed else selected[r] for r in range(4))
        # Removing the only contributing route produces an explicit empty list.
        for quota_name,quotas in (('original',{r:100 for r in active}),('equal_total',redistribute_quota(active))):
            preds=[prediction(rankings[i],weights,quotas) if any(weights) else [] for i in eligible]
            leave.append({'removed':removed,'quota_kind':quota_name,'quotas':quotas,'metrics':metrics(preds,qs,catalog,(10,20,50))})
    diagnostic=[]
    for i in eligible:
        sets=[set(r[:100]) for r in rankings[i]]
        diagnostic.append({'query_id':queries[i]['query_id'],'union':len(set.union(*sets)),
                           'overlap_pairs':[len(sets[a]&sets[b]) for a in range(4) for b in range(a+1,4)]})
    covered=set(item for i in eligible for ranks in rankings[i] for item in ranks[:100])
    coverage['route_union']={'count':len(covered),'fraction':len(covered)/len(catalog),'scope':'distinct items in four route top100 union / whole catalog'}
    out={'coverage':coverage,'consumer':consumer,'kind':'p4f_fusion','binding':m['binding'],'input_sha256':input_sha,'test_gate':'closed',
         'selected_weights':selected,'candidate_metrics':rows,'reports':reports,'leave_one_out':leave,
         'eligible_query_ids':[q['query_id'] for q in qs],'diagnostics':diagnostic,
         'itemcf_fill_counts':m.get('route_fill_counts'),'fill_scope':'counts measured during route export at top100/top134; no inference from overlap',
         'candidate_budget':65,'empty_history_count':len(queries)-len(qs)}
    output=Path(output);output.mkdir(parents=True,exist_ok=False);write_json(output/'manifest.json',out)
    return out


def selection_record(path,sha,m,input_sha):
    s=read_json(checked(path,sha))
    if s['kind']!='p4f_fusion' or s['binding']!=m['binding'] or s['input_sha256']!=input_sha or tuple(s['selected_weights']) not in weight_candidates():
        raise ValueError('frozen fusion selection/input mismatch')
    rows=s['candidate_metrics']
    if [tuple(r['weights']) for r in rows] != weight_candidates() or select_weights(rows) != tuple(s['selected_weights']):
        raise ValueError('fusion candidate budget/winner mismatch')
    if any(r['count'] != len(s['eligible_query_ids']) for r in rows):
        raise ValueError('fusion eligible denominator mismatch')
    return s


def _records(queries,rankings,weights,popularity):
    catalog=set(popularity);out=[]
    for q,routes in zip(queries,rankings):
        fallback=not q['history']
        out.append({'query_id':q['query_id'],'prediction':popularity[:50] if fallback else prediction(routes,weights),
                    'fallback':fallback,'unknown_ids':sum(i not in catalog for i in q['history'])+int(q['target'] not in catalog)})
    return out


def inspect_shards(output,identity,queries,require_complete=False):
    output=Path(output)
    original=read_json(output/'run.json')
    if original!=identity:raise ValueError('resume original identity mismatch')
    progress=sorted(output.glob('progress-*.json'))
    count=0
    if progress:
        latest=read_json(progress[-1]);count=latest['completed_shards']
        if count<1 or progress[-1].name!=f'progress-{count:06d}.json':raise ValueError('invalid progress')
    shard_paths=sorted(output.glob('shard-*.json'))
    # A completely written shard without its progress record can be recovered.
    if len(shard_paths) not in (count,count+1):raise ValueError('missing/extra shard')
    records=[];hashes={}
    for i,p in enumerate(shard_paths):
        if p.name!=f'shard-{i:06d}.json':raise ValueError('missing shard sequence')
        row=read_json(p)
        if row['identity_sha256']!=hashlib.sha256(canonical(identity)).hexdigest() or row['shard']!=i:
            raise ValueError('shard binding mismatch')
        start=i*identity['shard_size'];expected=[q['query_id'] for q in queries[start:start+identity['shard_size']]]
        if not expected or [r['query_id'] for r in row['records']]!=expected:raise ValueError('shard query loss/duplicate')
        records.extend(row['records']);hashes[p.name]=file_hash(p)
    if progress and any(hashes.get(name)!=digest for name,digest in latest['shards'].items()):raise ValueError('shard content changed')
    if require_complete and len(records)!=len(queries):raise ValueError('incomplete query coverage')
    return records,hashes


def infer(m,catalog,queries,rankings,output,input_sha,selection,selection_sha,shard_size=128,resume=False,max_shards=None,index=None, *, consumer_release=None):
    consumer = consumer_identity(consumer_release=consumer_release)
    require_consumer(selection,consumer)
    if type(shard_size) is not int or shard_size<1 or max_shards is not None and max_shards<1:raise ValueError('invalid shard budget')
    if index is not None:raise ValueError('ANN inference awaits independently accepted index audit gate; exact path retained')
    identity={'consumer':consumer,'kind':'p4f_infer','binding':m['binding'],'input_sha256':input_sha,'selection_sha256':selection_sha,
              'selection_ref':{'path':selection['_path'],'sha256':selection_sha},'shard_size':shard_size,'query_ids':[q['query_id'] for q in queries],'index_sha256':None}
    if selection['eligible_query_ids'] != [q['query_id'] for q in queries if q['history']]:
        raise ValueError('fusion eligible denominator/query mismatch')
    output=Path(output)
    if resume:
        if (output/'completion.json').exists():raise FileExistsError('inference already complete')
        records,hashes=inspect_shards(output,identity,queries)
    else:
        output.mkdir(parents=True,exist_ok=False);write_json(output/'run.json',identity);records=[];hashes={}
    if records != _records(queries[:len(records)],rankings[:len(records)],selection['selected_weights'],m['popularity']):
        raise ValueError('resume cached fusion result mismatch')
    done=len(records);new=0
    for start in range(done,len(queries),shard_size):
        shard=start//shard_size
        if start%shard_size:raise ValueError('partial shard coverage')
        rows=_records(queries[start:start+shard_size],rankings[start:start+shard_size],selection['selected_weights'],m['popularity'])
        p=output/f'shard-{shard:06d}.json';write_json(p,{'shard':shard,'identity_sha256':hashlib.sha256(canonical(identity)).hexdigest(),'records':rows})
        hashes[p.name]=file_hash(p);records.extend(rows);new+=1
        write_json(output/f'progress-{shard+1:06d}.json',{'completed_shards':shard+1,'shards':dict(hashes)})
        if max_shards is not None and new>=max_shards:break
    if len(records)<len(queries):return {'status':'partial','completed_queries':len(records)}
    completion=_completion(identity,records,hashes,queries,catalog)
    write_json(output/'completion.json',completion)
    return completion


def _completion(identity,records,hashes,queries,catalog):
    eligible=[i for i,q in enumerate(queries) if q['history']];fallback=[i for i,q in enumerate(queries) if not q['history']]
    report=metrics([records[i]['prediction'] for i in eligible],[queries[i] for i in eligible],catalog,(10,20,50))
    fallback_report=metrics([records[i]['prediction'] for i in fallback],[queries[i] for i in fallback],catalog,(10,20,50))
    return {'kind':'p4f_completion','identity':identity,'shards':hashes,'result_sha256':hashlib.sha256(canonical(records)).hexdigest(),
            'eligible_count':len(eligible),'fallback_count':len(fallback),'unknown_ids':sum(r['unknown_ids'] for r in records),
            'metrics':report,'fallback_metrics':fallback_report,'test_gate':'closed'}


def verify(output,m,catalog,queries,rankings,input_sha, *, consumer_release=None):
    consumer = consumer_identity(consumer_release=consumer_release)
    output=Path(output);completion=read_json(output/'completion.json');identity=completion['identity']
    require_consumer(identity,consumer)
    if identity['binding']!=m['binding'] or identity['input_sha256']!=input_sha:raise ValueError('completion input mismatch')
    ref=identity['selection_ref'];selected=selection_record(ref['path'],ref['sha256'],m,input_sha)
    require_consumer(selected,consumer)
    records,hashes=inspect_shards(output,identity,queries,True)
    if selected['eligible_query_ids'] != [q['query_id'] for q in queries if q['history']]:
        raise ValueError('fusion eligibility changed')
    if records != _records(queries,rankings,selected['selected_weights'],m['popularity']):
        raise ValueError('cached fusion prediction mismatch')
    for q,row in zip(queries,records):
        if row['fallback']!= (not q['history']) or row['unknown_ids']!=sum(i not in catalog for i in q['history'])+int(q['target'] not in catalog):raise ValueError('fallback/count mismatch')
    if completion!=_completion(identity,records,hashes,queries,catalog):raise ValueError('completion mismatch')
    return {'status':'verified','query_count':len(records),'numerical_model_replay':False}


def main(argv=None):
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    for name in ('export','index','fuse','infer','verify'):
        s=sub.add_parser(name);s.add_argument('--input',required=True);s.add_argument('--input-sha256',required=True);s.add_argument('--output',required=True)
        s.add_argument('--consumer-release');s.add_argument('--consumer-release-sha256')
        if name=='index':
            s.add_argument('--export',required=True);s.add_argument('--export-sha256',required=True)
            s.add_argument('--audit');s.add_argument('--audit-sha256');s.add_argument('--exact-predictions');s.add_argument('--exact-predictions-sha256')
        if name=='infer':
            s.add_argument('--selection',required=True);s.add_argument('--selection-sha256',required=True)
            s.add_argument('--shard-size',type=int,default=128);s.add_argument('--max-shards',type=int);s.add_argument('--resume',action='store_true')
    a=p.parse_args(argv)
    if bool(a.consumer_release) != bool(a.consumer_release_sha256):
        raise ValueError('paired consumer release path/SHA required')
    release={'path':a.consumer_release,'sha256':a.consumer_release_sha256} if a.consumer_release is not None else None
    if release is not None:consumer_identity(consumer_release=release)
    m,paths,catalog,queries,rankings,idmap=load_input(a.input,a.input_sha256,require_routes=a.command!='export',consumer_release=release)
    if a.command=='fuse':out=fuse(m,catalog,queries,rankings,a.output,a.input_sha256,consumer_release=release)
    elif a.command=='export':
        import torch
        ctx={'checkpoint':str(paths['checkpoint']),'checkpoint_binding':m['checkpoint_binding'],'config':m['config'],
             'content':torch.from_numpy(np.load(paths['content'],allow_pickle=False)),'supported_indices':read_json(paths['supported_indices']),
             'catalog':catalog,'idmap':idmap,'queries':[q for q in queries if q['history']]}
        if not m.get('synthetic'):
            ctx['supported_indices']=read_json(paths['idmap'])['supported_indices']
            ctx['graph']=read_json(paths['graph']);ctx['graph_sha256']=m['files']['graph']['sha256']
        out=export_vectors(ctx,a.output,m['binding'],consumer_release=release)
    elif a.command=='index':
        options=(a.audit,a.audit_sha256,a.exact_predictions,a.exact_predictions_sha256)
        if any(options) and not all(options):raise ValueError('all pinned audit inputs required')
        if not m.get('synthetic') and not all(options):raise ValueError('real index requires fixed audit inputs')
        out=build_index({'path':a.export,'sha256':a.export_sha256},a.output,m['binding'],consumer_release=release)
        if all(options):
            audit=read_json(checked(a.audit,a.audit_sha256));audit['vectors']=np.asarray(audit['vectors'],dtype=np.float32)
            if bool(audit.get('synthetic'))!=bool(m.get('synthetic')):raise ValueError('audit scientific mode mismatch')
            exact=read_json(checked(a.exact_predictions,a.exact_predictions_sha256))
            manifest=Path(a.output)/'manifest.json'
            out=audit_index({'path':str(manifest.resolve()),'sha256':file_hash(manifest)},audit,exact,Path(a.output)/'audit',m['binding'],consumer_release=release)
    elif a.command=='infer':
        s=selection_record(a.selection,a.selection_sha256,m,a.input_sha256);s['_path']=str(Path(a.selection).resolve())
        out=infer(m,catalog,queries,rankings,a.output,a.input_sha256,s,a.selection_sha256,a.shard_size,a.resume,a.max_shards,consumer_release=release)
    else:out=verify(a.output,m,catalog,queries,rankings,a.input_sha256,consumer_release=release)
    # Print only stage/role status; full private payload remains in output files.
    print(canonical({'command':a.command,'status':out.get('status','success'),'kind':out.get('kind')}).decode())

if __name__=='__main__':
    try:main()
    except (ValueError,KeyError,OSError,RuntimeError) as error:
        print(type(error).__name__+': '+str(error),file=sys.stderr);raise SystemExit(1)
