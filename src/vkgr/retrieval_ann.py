"""P4F checkpoint exports and fixed CPU HNSW audit; no dependency installation."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np

AUDIT_SHA = '084290bdc4184487dc1190505d9fbb4ff3d5e938b4189f73912b3109bd3714a2'

class DependencyUnavailable(RuntimeError):
    pass


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def checked(path, expected):
    if not isinstance(expected, str) or len(expected) != 64 or file_hash(path) != expected:
        raise ValueError('input hash mismatch')
    return Path(path)


def write_json(path, value):
    # Same-filesystem hard link publishes complete bytes atomically without replacement.
    # Retain owned staging evidence (hard-linked inode after success); no destructive cleanup.
    import os, uuid
    path=Path(path)
    payload=canonical(value)
    staging=path.with_name('.'+path.name+'.pending-'+uuid.uuid4().hex)
    with staging.open('xb') as f:
        f.write(payload);f.flush();os.fsync(f.fileno())
    os.link(staging,path)
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def read_json(path):
    def unique(pairs):
        out = {}
        for k, v in pairs:
            if k in out:
                raise ValueError('duplicate JSON key')
            out[k] = v
        return out
    return json.loads(Path(path).read_text(), object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def validate_binding(binding):
    for name in ('source_sha256','checkpoint_sha256','catalog_sha256','idmap_sha256','query_sha256'):
        v = binding.get(name)
        if not isinstance(v, str) or len(v) != 64 or any(c not in '0123456789abcdef' for c in v):
            raise ValueError('missing scientific pin: '+name)


def export_vectors(checkpoint_context, output, binding, *, consumer_release=None):
    """Load caller-pinned own checkpoint on CPU without restoring optimizer/RNG."""
    consumer = consumer_identity(consumer_release=consumer_release)
    import torch
    from .retrieval_models import TwoTower
    from .retrieval_experiment import export_vectors as producer_export
    validate_binding(binding)
    ctx = checkpoint_context
    catalog = ctx['catalog']
    idmap = ctx['idmap']
    config = ctx['config']
    if (not catalog or any(type(i) is not int or i < 0 for i in catalog) or
            catalog != sorted(set(catalog)) or idmap != {item:i+1 for i,item in enumerate(catalog)} or
            config['seed'] != 42 or config['batch_size'] != 512):
        raise ValueError('catalog/idmap or frozen representative mismatch')
    queries = ctx['queries']
    ids = [q['query_id'] for q in queries]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('invalid query identity')
    if any(not q['history'] or len(q['history']) > 50 or any(i not in idmap for i in q['history']) for q in queries):
        raise ValueError('invalid eligible export history')
    state = torch.load(checked(ctx['checkpoint'], binding['checkpoint_sha256']), map_location='cpu', weights_only=True)
    if state.get('schema_version') != 1 or state['binding'] != ctx['checkpoint_binding']:
        raise ValueError('historical checkpoint binding mismatch')
    model = TwoTower(config['architecture'], ctx['content'], ctx['supported_indices'])
    model.load_state_dict(state['model'])
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('nonfinite checkpoint')
    users, items = producer_export(model, catalog, queries, idmap, config['batch_size'])
    if users.shape != (len(ids), 64) or items.shape != (len(catalog),64) or not np.isfinite(users).all() or not np.isfinite(items).all():
        raise ValueError('invalid exported vectors')
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    for name, rows in (('items.npy',items),('users.npy',users),('ids.npy',np.array(catalog,dtype=np.int64))):
        with (output/name).open('xb') as f:np.save(f,rows,allow_pickle=False)
    extra={}
    if 'graph' in ctx:
        from .retrieval_baselines import itemcf_rank, pop_rank
        from .baselines import history_centroids, blockwise_topk
        graph=ctx['graph'];popularity=pop_rank(graph)
        if graph['catalog']!=catalog:raise ValueError('graph/catalog mismatch')
        neighbors={int(i):row for i,row in graph['neighbors'].items()}
        histories=[q['history'] for q in queries]
        tt=blockwise_topk(users,items,np.array(catalog,dtype=np.int64),134).tolist()
        content_vectors=ctx['content'].numpy()[1:]
        content_users,zero_centroids=history_centroids(histories,catalog,content_vectors)
        content_ranks=blockwise_topk(content_users,content_vectors,np.array(catalog,dtype=np.int64),134).tolist()
        cf=[itemcf_rank(h,neighbors,popularity,134) for h in histories]
        routing={'query_ids':ids,'rankings':[[cf[i][0],tt[i],content_ranks[i],popularity[:134]] for i in range(len(ids))],
                 'itemcf_fill_top134':[r[1] for r in cf],
                 'itemcf_fill_top100':[itemcf_rank(h,neighbors,popularity,100)[1] for h in histories],
                 'popularity':popularity,'graph_sha256':ctx['graph_sha256'],
                 'query_records':queries,'scope':'one export computation; exact fixed float64 pair reduction; no seen masking'}
        write_json(output/'routes.json',routing);extra={'routes.json':file_hash(output/'routes.json')}
    manifest = {'consumer':consumer,'schema_version':1,'kind':'p4f_export','binding':binding,'query_ids':ids,'shape':list(items.shape),
                'query_records':queries,'files':{**{name:file_hash(output/name) for name in ('items.npy','users.npy','ids.npy')},**extra}}
    write_json(output/'manifest.json',manifest)
    return manifest


def read_export(path, sha256, binding, *, consumer=None):
    validate_binding(binding)
    path = checked(path,sha256); m = read_json(path)
    if consumer is not None:require_consumer(m,consumer)
    if m.get('kind') != 'p4f_export' or m['binding'] != binding or set(m['files']) not in ({'items.npy','users.npy','ids.npy'},{'items.npy','users.npy','ids.npy','routes.json'}):
        raise ValueError('export identity mismatch')
    arrays = {name:np.load(checked(path.parent/name,digest),allow_pickle=False) for name,digest in m['files'].items() if name.endswith('.npy')}
    ids, users, items = arrays['ids.npy'],arrays['users.npy'],arrays['items.npy']
    if (ids.dtype != np.int64 or ids.ndim != 1 or ids.tolist() != sorted(set(ids.tolist())) or
            items.dtype != np.float32 or users.dtype != np.float32 or items.shape != tuple(m['shape']) or
            items.shape != (len(ids),64) or users.shape != (len(m['query_ids']),64) or
            not np.isfinite(items).all() or not np.isfinite(users).all()):
        raise ValueError('invalid export content/shape')
    return ids, users, items


def _faiss():
    try:
        import faiss
    except ImportError as error:
        raise DependencyUnavailable('existing FAISS unavailable; real HNSW validation pending') from error
    return faiss


def build_index(export_manifest, output, binding, *, consumer_release=None):
    consumer = consumer_identity(consumer_release=consumer_release)
    ids, users, items = read_export(export_manifest['path'],export_manifest['sha256'],binding,consumer=consumer)
    faiss = _faiss(); faiss.omp_set_num_threads(1)
    rss_before=resident_bytes()
    start = time.perf_counter()
    index = faiss.IndexHNSWFlat(64,32,faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 200
    index.add(np.ascontiguousarray(items))
    elapsed = time.perf_counter()-start
    output = Path(output); output.mkdir(parents=True,exist_ok=False)
    faiss.write_index(index,str(output/'index.faiss'))
    m = {'schema_version':1,'kind':'p4f_hnsw','binding':binding,'export':export_manifest,
         'index_sha256':file_hash(output/'index.faiss'),'index_bytes':(output/'index.faiss').stat().st_size,
         'consumer':consumer,'rss_before_bytes':rss_before,'rss_after_build_bytes':resident_bytes(),'memory_scope':'visible process resident bytes; boundary delta includes build workspace; serialized index bytes separate','m':32,'ef_construction':200,'metric':'inner_product','threads':1,'build_seconds':elapsed}
    write_json(output/'manifest.json',m)
    return m


def audit_index(index_manifest, queries, exact_predictions, output, binding, *, consumer_release=None):
    consumer = consumer_identity(consumer_release=consumer_release)
    from .baselines import blockwise_topk
    m = read_json(checked(index_manifest['path'],index_manifest['sha256']))
    require_consumer(m,consumer)
    if m['kind'] != 'p4f_hnsw' or m['binding'] != binding or (m['m'],m['ef_construction'],m['threads']) != (32,200,1):
        raise ValueError('index identity mismatch')
    ids, all_users, items = read_export(m['export']['path'],m['export']['sha256'],binding,consumer=consumer)
    exported = read_json(m['export']['path'])
    qids = validate_audit_queries(queries,exported,binding)
    positions = {q:i for i,q in enumerate(exported['query_ids'])}
    if any(q not in positions for q in qids):raise ValueError('audit query missing from export')
    users = all_users[[positions[q] for q in qids]]
    if not np.array_equal(users,queries['vectors']) or len(queries['targets'])!=len(users):
        raise ValueError('exact/ANN vectors or query mismatch')
    exact = blockwise_topk(users,items,ids,50).tolist()
    exact_times=[]
    for user in users:
        begin=time.perf_counter();blockwise_topk(user.reshape(1,-1),items,ids,50);exact_times.append(time.perf_counter()-begin)
    if exact != exact_predictions:raise ValueError('exact authoritative audit mismatch')
    faiss = _faiss(); faiss.omp_set_num_threads(1)
    index = faiss.read_index(str(checked(Path(index_manifest['path']).parent/'index.faiss',m['index_sha256'])))
    if index.ntotal != len(ids) or index.d != 64:raise ValueError('index shape mismatch')
    def metric(predictions):
        import math
        recall = []; ndcg = []
        for row,target in zip(predictions,queries['targets']):
            hit = target in row;recall.append(float(hit));ndcg.append(1/math.log2(row.index(target)+2) if hit else 0.)
        return {'recall@50':float(np.mean(recall)),'ndcg@50':float(np.mean(ndcg))}
    reference = metric(exact); rows = []
    for ef in (64,128,256):
        index.hnsw.efSearch=ef
        index.search(np.ascontiguousarray(users[:1]),min(50,len(ids)))  # one unmeasured warmup
        predictions=[];retrieval=[];end_to_end=[]
        for user in users:
            begin=time.perf_counter();_,positions=index.search(np.ascontiguousarray(user.reshape(1,-1)),min(50,len(ids)));retrieval.append(time.perf_counter()-begin)
            pos=positions[0].tolist()
            if any(p<0 or p>=len(ids) for p in pos) or len(set(pos))!=len(pos):raise ValueError('invalid ANN positions')
            candidates=ids[pos];scores=np.einsum('qd,id->qi',user.reshape(1,-1),items[pos],dtype=np.float64,optimize=False)[0]
            predictions.append(candidates[np.lexsort((candidates,-scores))].tolist());end_to_end.append(time.perf_counter()-begin)
        overlap=float(np.mean([len(set(a)&set(b))/len(a) for a,b in zip(exact,predictions)]))
        ann=metric(predictions)
        rows.append({'ef_search':ef,'overlap@50':overlap,'model_delta':{k:ann[k]-reference[k] for k in reference},
                     'retrieval_p50_seconds':float(np.percentile(retrieval,50)),'retrieval_p95_seconds':float(np.percentile(retrieval,95)),
                     'end_to_end_p50_seconds':float(np.percentile(end_to_end,50)),'end_to_end_p95_seconds':float(np.percentile(end_to_end,95))})
    chosen=select_audit_setting(rows)
    result={'consumer':consumer,'kind':'p4f_ann_audit','binding':binding,'index':index_manifest,'query_ids':qids,'rows':rows,
            'gate_passed':chosen is not None,'selected_ef_search':chosen,'exact_retained':True,
            'scope':'fixed audit only; batch1; threads1; one warmup per ef; retrieval-only search; end-to-end search+mapping+rerank excludes input load',
            'index_bytes':m['index_bytes'],'build_seconds':m['build_seconds'],'rss_after_load_bytes':resident_bytes(),'exact_p50_seconds':float(np.percentile(exact_times,50)),'exact_p95_seconds':float(np.percentile(exact_times,95)),'memory_scope':m['memory_scope']}
    output=Path(output);output.mkdir(parents=True,exist_ok=False);write_json(output/'manifest.json',result)
    return result


def select_audit_setting(rows):
    if [r['ef_search'] for r in rows] != [64,128,256] or any(not np.isfinite(r['overlap@50']) or not 0 <= r['overlap@50'] <= 1 for r in rows):
        raise ValueError('fixed ef audit budget or finite overlap mismatch')
    return next((r['ef_search'] for r in rows if r['overlap@50'] >= .98), None)


# External release SHA is the coordinator trust anchor, never an implicit fallback.
_RELEASE_PARENTS = {
    'source': '7ff025e94b82ec11d731912c1da10fcef35def77146f860e2d34da32d32de31f',
    'acceptance': '2f9a6de259e8d5613bac940c18c4cbc06c730aa168717208700d64ec1316299e',
    'transport': 'c0bb16d835c4b5c105c420ee09077113be19e0ec02b0789340719d0395b1bfc9',
}
_RELEASE_CHANGES = {'src/vkgr/retrieval_ann.py', 'src/vkgr/retrieval_infer_run.py',
                    'tests/test_retrieval_ann.py', 'tests/test_retrieval_infer_run.py'}
_RELEASE_WHEELS = {
    'packaging': 'd7193f7c8e4e93f444fde0262bf90af30e16fa0ad0ad44cb553c87339b23cd1c',
    'faiss-cpu': 'f2c31b7f2f6647eb76829a5cfe3c398fb9346df9f26b1d4db35269c91eb58c33',
}
_RELEASE_VERSIONS = {'numpy':'1.26.4', 'torch':'2.11.0+cu128', 'pyarrow':'19.0.1',
                     'PyYAML':'6.0.2', 'packaging':'26.3', 'faiss-cpu':'1.15.1'}


def _fields(value, names):
    if type(value) is not dict or set(value) != set(names):
        raise ValueError('consumer release schema mismatch')


def _digest(value):
    if type(value) is not str or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('consumer release SHA required')
    return value


def _absolute_path(value):
    if type(value) is not str:
        raise ValueError('consumer release path must be string')
    path = Path(value)
    if not path.is_absolute() or str(path) != value or '..' in path.parts:
        raise ValueError('consumer release canonical absolute path required')
    for ancestor in (*reversed(path.parents),path):
        if ancestor.is_symlink():raise ValueError('consumer release symlink rejected')
    return path


def _regular(path):
    path = Path(path).absolute()
    for ancestor in (*reversed(path.parents), path):
        if ancestor.is_symlink():
            raise ValueError('consumer release symlink rejected')
    if not path.is_file():
        raise ValueError('consumer release file missing')
    return path


def _relative(root, name):
    import re
    if type(name) is not str or not re.fullmatch(r'[A-Za-z0-9_. /-]+',name) or ' ' in name:
        raise ValueError('consumer release unsafe member')
    if any(part in ('', '.', '..') for part in name.split('/')):
        raise ValueError('consumer release unsafe member')
    return _regular(root/name)


def _member(root, name, record):
    _fields(record, ('bytes','sha256'))
    if type(record['bytes']) is not int or record['bytes'] < 0:
        raise ValueError('consumer release invalid size')
    p = _relative(root,name)
    if p.stat().st_size != record['bytes'] or file_hash(p) != _digest(record['sha256']):
        raise ValueError('consumer release member drift')
    return p


def _runtime_identity():
    """Observe loaded modules and owning distribution metadata, with no Git access."""
    import importlib, importlib.metadata, platform, sys
    deps = {}
    for distribution, module in (('numpy','numpy'),('torch','torch'),('pyarrow','pyarrow'),
                                 ('PyYAML','yaml'),('packaging','packaging'),('faiss-cpu','faiss')):
        mod = importlib.import_module(module)
        dist = importlib.metadata.distribution(distribution)
        metadata = [f for f in (dist.files or ()) if str(f).endswith('.dist-info/METADATA')]
        if len(metadata) != 1:
            raise ValueError('consumer dependency metadata ambiguity')
        origin = _regular(mod.__file__)
        meta = _regular(dist.locate_file(metadata[0]))
        distribution_root = Path(dist.locate_file('')).absolute()
        if not origin.is_relative_to(distribution_root) or not meta.is_relative_to(distribution_root):
            raise ValueError('consumer dependency origin mismatch')
        deps[distribution] = {'version':dist.version,'module_version':mod.__version__,
            'origin':str(origin),'distribution_root':str(distribution_root),
            'origin_sha256':file_hash(origin),'metadata_sha256':file_hash(meta)}
    return {'python':platform.python_version(),
            'executable_sha256':file_hash(_regular(Path(sys.executable).resolve())), 'dependencies':deps}


def _dependency_snapshot(bundle, runtime):
    _fields(bundle, ('root','fixed_root','wheels','members'))
    if bundle['wheels'] != _RELEASE_WHEELS or type(bundle['members']) is not dict:
        raise ValueError('consumer dependency wheel identity mismatch')
    roots = ('packaging','packaging-26.3.dist-info','faiss','faiss_cpu.libs','faiss_cpu-1.15.1.dist-info')
    target = _absolute_path(bundle['root'])
    fixed = _absolute_path(bundle['fixed_root'])
    if not target.is_absolute() or not fixed.is_absolute() or target == fixed:
        raise ValueError('consumer dependency absolute isolated roots required')
    names = set(bundle['members'])
    if any(type(n) is not str for n in names) or len(names) != 68 or {n.split('/')[0] for n in names} != set(roots):
        raise ValueError('consumer dependency complete bundle required')
    for name, record in bundle['members'].items():
        _member(target,name,record)
    # No unrecorded code, native library, symlink or directory in the isolated target.
    actual = set()
    for path in target.rglob('*'):
        if path.is_symlink():raise ValueError('consumer dependency symlink rejected')
        if path.is_file():actual.add(path.relative_to(target).as_posix())
        elif not path.is_dir():raise ValueError('consumer dependency special member')
    if actual != names:
        raise ValueError('consumer dependency closure drift')
    _fields(runtime, ('python','executable_sha256','dependencies'))
    import re
    if type(runtime['python']) is not str or not re.fullmatch(r'3\.11\.[0-9]+',runtime['python']):
        raise ValueError('consumer runtime Python mismatch')
    _digest(runtime['executable_sha256'])
    _fields(runtime['dependencies'], _RELEASE_VERSIONS)
    for name, version in _RELEASE_VERSIONS.items():
        row=runtime['dependencies'][name]
        _fields(row, ('version','module_version','origin','distribution_root','origin_sha256','metadata_sha256'))
        if row['version'] != version or row['module_version'] != version:
            raise ValueError('consumer runtime dependency version mismatch')
        _digest(row['origin_sha256']);_digest(row['metadata_sha256'])
        expected = target if name in _RELEASE_WHEELS else fixed
        if row['distribution_root'] != str(expected) or not _absolute_path(row['origin']).is_relative_to(expected):
            raise ValueError('consumer runtime dependency source mismatch')
    observed = _runtime_identity()
    if observed != runtime:
        raise ValueError('consumer runtime/dependency drift')
    return hashlib.sha256(canonical(bundle)).hexdigest(), observed


def _release_identity(ref):
    from .manifest import source_fingerprint
    _fields(ref, ('path','sha256'))
    if type(ref['path']) is not str:raise ValueError('consumer release path must be string')
    path=_regular(ref['path'])
    if file_hash(path) != _digest(ref['sha256']):
        raise ValueError('consumer release external hash mismatch')
    release=read_json(path)
    if path.read_bytes() != canonical(release):
        raise ValueError('consumer release canonical JSON required')
    _fields(release, ('schema_version','kind','parents','declared_build_git','members','dependency_bundle','runtime'))
    if type(release['schema_version']) is not int or release['schema_version'] != 1 or release['kind'] != 'p4f_consumer_release':
        raise ValueError('consumer release unsupported schema')
    root=Path(__file__).absolute().parents[2]
    _fields(release['parents'], _RELEASE_PARENTS)
    parents={}
    for role, digest in _RELEASE_PARENTS.items():
        record=release['parents'][role];_fields(record, ('path','sha256'))
        p=_relative(root,record['path'])
        if record['sha256'] != digest or file_hash(p) != digest:
            raise ValueError('consumer accepted parent mismatch')
        parents[role]=read_json(p)
    source=parents['source'];acceptance=parents['acceptance'];base=dict(parents['transport']['members'])
    for name,digest in source['files'].items():
        base.setdefault(name,{'sha256':digest})
    if (acceptance['manifest_sha256'] != _RELEASE_PARENTS['source'] or
            acceptance['status'] != 'accepted_local_real_faiss_and_real_evaluation_pending' or
            source['scientific_source_sha256'] != '573c1bdc586f27ead094bf20d497ae16a447264faedbaf421302e230b147a508' or
            source['scientific_plan_sha256'] != '8d8e2a657824743f02cb9f7360dfeda9b120b7b26f81a9deec6ac49040257f3d'):
        raise ValueError('consumer accepted chain mismatch')
    members=release['members']
    if type(members) is not dict or set(members) != set(base):
        raise ValueError('consumer complete source/lock/config closure required')
    for name, record in members.items():
        _member(root,name,record)
        if name not in _RELEASE_CHANGES and record['sha256'] != base[name]['sha256']:
            raise ValueError('consumer protected source/lock/config changed')
    for name, digest in source['files'].items():
        if name not in _RELEASE_CHANGES and file_hash(_relative(root,name)) != digest:
            raise ValueError('consumer protected reviewed member drift')
    actual_sources={p.relative_to(root).as_posix() for p in (root/'src/vkgr').rglob('*')
                    if not p.is_dir() or p.is_symlink()}
    if actual_sources != {n for n in members if n.startswith('src/vkgr/')}:
        raise ValueError('consumer source closure drift')
    build=release['declared_build_git'];_fields(build, ('commit','dirty'))
    import re
    if type(build['commit']) is not str or not re.fullmatch('[0-9a-f]{40}',build['commit']) or type(build['dirty']) is not bool:
        raise ValueError('consumer declared build Git invalid')
    bundle_sha, runtime=_dependency_snapshot(release['dependency_bundle'],release['runtime'])
    return {'identity_kind':'verified_deployment_release','release_sha256':ref['sha256'],
            'accepted_parent_source_manifest_sha256':_RELEASE_PARENTS['source'],
            'accepted_parent_coordinator_sha256':_RELEASE_PARENTS['acceptance'],
            'actual_git_commit':None,'actual_git_dirty':None,'git_status':'not_queried_explicit_release',
            'declared_build_git':build,'source_fingerprint':source_fingerprint(),
            'lock_sha256':file_hash(root/'uv.lock'),'config_sha256':file_hash(root/'configs/traditional.yaml'),
            'dependency_bundle_sha256':bundle_sha,'runtime':runtime}


def require_consumer(record, identity):
    """Release artifacts may only be consumed under the same explicit identity."""
    recorded=record.get('consumer')
    if (identity.get('identity_kind') == 'verified_deployment_release' or
            isinstance(recorded,dict) and recorded.get('identity_kind') == 'verified_deployment_release'):
        if recorded != identity:
            raise ValueError('consumer release artifact identity mismatch')


def consumer_identity(*, consumer_release=None):
    if consumer_release is not None:
        return _release_identity(consumer_release)
    import platform
    from importlib.metadata import version, PackageNotFoundError
    from .manifest import git_context, source_fingerprint
    root=Path(__file__).resolve().parents[2]
    commit,dirty=git_context()
    deps={}
    for name in ('numpy','torch','faiss-cpu'):
        try:deps[name]=version(name)
        except PackageNotFoundError:deps[name]=None
    return {'git_commit':commit,'git_dirty':dirty,'source_fingerprint':source_fingerprint(),
            'lock_sha256':file_hash(root/'uv.lock'),'config_sha256':file_hash(root/'configs/traditional.yaml'),
            'python':platform.python_version(),'dependencies':deps}


def resident_bytes():
    import os, subprocess
    status=Path('/proc/self/status')
    if status.exists():
        for line in status.read_text().splitlines():
            if line.startswith('VmRSS:'):return int(line.split()[1])*1024
    return int(subprocess.run(['ps','-o','rss=','-p',str(os.getpid())],check=True,capture_output=True,text=True).stdout.strip())*1024


def validate_audit_queries(queries, exported, binding):
    from .data_audit import canonical_bytes
    ids=queries['query_ids']
    if not ids or len(ids)>512 or len(ids)!=len(set(ids)):
        raise ValueError('invalid audit query membership')
    records={q['query_id']:q for q in exported['query_records']}
    if queries.get('synthetic'):
        if ids!=exported['query_ids'] or queries['audit_sha256']!=binding['query_sha256']:
            raise ValueError('synthetic audit identity mismatch')
    else:
        root=Path(__file__).resolve().parents[2]
        accepted=root/'artifacts/p4-baselines-manifest.json'
        ref=queries['legacy_manifest']
        manifest=read_json(checked(ref['path'],ref['sha256']))
        if ref['sha256']!=file_hash(accepted) or manifest['audit_set_hashes']['validation']!=AUDIT_SHA:
            raise ValueError('legacy audit manifest mismatch')
        audit_ref=queries['legacy_validation_audit']
        old=read_json(checked(audit_ref['path'],manifest['artifacts']['validation-audit.json']))
        old_records=old['queries'];old_ids=[q['user_id'] for q in old_records]
        if ids!=old_ids or hashlib.sha256(canonical_bytes(ids)).hexdigest()!=AUDIT_SHA or queries['audit_sha256']!=AUDIT_SHA:
            raise ValueError('frozen legacy audit members mismatch')
        for original in old_records:
            current=records.get(original['user_id'])
            if current is None or any(current[k]!=original[k] for k in ('user_id','history','target','oov')):
                raise ValueError('legacy audit history/target mismatch')
    if any(i not in records for i in ids) or queries['targets']!=[records[i]['target'] for i in ids]:
        raise ValueError('audit target binding mismatch')
    return ids
