"""Bound original CPU/CUDA subset SGNS; no holdout reader or model training."""
import argparse
import json
from pathlib import Path
import sys
import time

from analysis.ur001_derive_cli import digest, encoded, secure

ROOT = Path(__file__).resolve().parents[1]
CATALOG_SHA = '80201c574fad26bd29f2e2425a5ad98db8fe7c2a9be91e07d2fc6aedc48d47f3'
CONTENT_SHA = '644b2d7a656999a8962355d7c32dae07134ad46e9557b4a8418f137d62f73581'


def progress(phase, **fields):
    print(json.dumps({'phase':phase, 'monotonic':time.monotonic(), **fields}, sort_keys=True),
          file=sys.stderr, flush=True)


def fixed_records():
    from analysis.a40_subset_prepare import inventory
    from analysis.new_instance_prepare import A_INVENTORY_SHA
    path = ROOT/'configs/ur001-preparation/hf-files.json'
    if digest(path) != A_INVENTORY_SHA:
        raise ValueError('full inventory drift')
    full = json.loads(path.read_bytes())
    records = inventory()['files'][:25]
    if [r['path'] for r in records] != [f'subsamples/ur0.01_ir0.01/train/week_{w:02d}.parquet' for w in range(25)]:
        raise ValueError('fixed train order required')
    embedding = [r for r in full['files'] if r['path'] == 'metadata/item_embeddings.npz']
    if len(embedding) != 1:
        raise ValueError('one official embedding required')
    return records + embedding


def checked_file(path, record):
    path = secure(path)
    if not path.is_file() or path.stat().st_size != record['size'] or digest(path) != record['sha256']:
        raise ValueError('fixed input SHA/size differs')
    return path


def snapshot(path, record):
    import hashlib
    raw = secure(path).read_bytes()
    if len(raw) != record['size'] or hashlib.sha256(raw).hexdigest() != record['sha256']:
        raise ValueError('consumed input SHA/size differs')
    return raw


def load_sequences(paths, records=None):
    """Same physical iteration as original sid_coll_job_entry.load_histories."""
    import pyarrow.parquet as pq
    import pyarrow as pa
    if records is not None and len(paths) != len(records):
        raise ValueError('one pin per train file required')
    histories = {}
    catalog = set()
    for index, path in enumerate(paths):
        progress('train_week_begin', index=index)
        raw = snapshot(path,records[index]) if records is not None else secure(path).read_bytes()
        for batch in pq.ParquetFile(pa.BufferReader(raw)).iter_batches(batch_size=65536, columns=['user_id', 'item_id']):
            for row in batch.to_pylist():
                user, item = row['user_id'], row['item_id']
                if any(type(x) is not int or not 0 <= x < 2**32 for x in (user, item)):
                    raise ValueError('raw user/item must be uint32')
                histories.setdefault(user, []).append(item)
                catalog.add(item)
        progress('train_week_end', index=index)
    return histories, sorted(catalog)


def inspect(data, embedding_root, lock):
    from analysis.new_instance_prepare import runtime
    from analysis.ur001_derive_cli import source_identity, guards
    data, embedding_root, lock = map(secure, (data, embedding_root, lock))
    source = source_identity()
    from vkgr import sid_coll_sgns, retrieval_data
    expected = {'src/vkgr/sid_coll_sgns.py':'2cc25d1997c12fe1e262e455e6644980186b5b9d0e00805014dee3148675876d',
                'src/vkgr/retrieval_data.py':'1fa39c85b98ba8aebfabfd0c23659c4a178a1f3f842a5985469656c9deadb867'}
    if any(source['files'].get(name) != sha for name, sha in expected.items()):
        raise ValueError('original scientific function bytes required')
    env = runtime(lock)
    records = fixed_records()
    # Reject a second file in any train week rather than choosing one silently.
    folder = data/'subsamples/ur0.01_ir0.01/train'
    if set(folder.glob('*.parquet')) != {data/r['path'] for r in records[:25]}:
        raise ValueError('one exact file per week required')
    guards(data, projected_memory=16*1024**3)
    paths = []
    for index, rec in enumerate(records):
        base = data if index < 25 else embedding_root
        path = checked_file(base/rec['path'], rec)
        paths.append(path)
        progress('input_hash_checked', index=index)
    return paths, {'schema':'sgns-subset-physical-input-v1','engineering':False,
                   'records':records,'source':source,'runtime':env,
                   'scientific_source':'079d6bd35ae74317b29408cd43d3ec35de77c1eb',
                   'histories':'all observed train events, original physical order, user ascending in fit'}


def prepare(data, embedding_root, lock, output, *, readonly):
    import numpy as np
    from vkgr.retrieval_data import content_join
    from vkgr.sid_coll_sgns import _sequence_digest
    output = secure(output)
    if output.exists():
        raise FileExistsError('new exclusive output required')
    if ROOT == output or ROOT in output.parents or output in ROOT.parents:
        raise ValueError('output must be outside deployment')
    paths, binding = inspect(data, embedding_root, lock)
    histories, catalog = load_sequences(paths[:25],records=binding['records'][:25])
    import hashlib
    if len(catalog) != 120415 or hashlib.sha256(encoded(catalog)).hexdigest() != CATALOG_SHA:
        raise ValueError('original catalog pin differs; STOP')
    progress('content_join_begin')
    import io
    feature_bytes = snapshot(paths[-1],binding['records'][-1])
    with np.load(io.BytesIO(feature_bytes), allow_pickle=False) as arrays:
        content = content_join(catalog, arrays['item_id'], arrays['embedding'])
    del feature_bytes
    buffer = io.BytesIO(); np.save(buffer, content, allow_pickle=False)
    if hashlib.sha256(buffer.getbuffer()).hexdigest() != CONTENT_SHA:
        raise ValueError('original content pin differs; STOP')
    # Rehash every source after use, before the final receipt can exist.
    for path, record in zip(paths, binding['records']): checked_file(path, record)
    from analysis.ur001_derive_cli import source_identity
    from analysis.new_instance_prepare import runtime
    if source_identity() != binding['source'] or runtime(secure(lock)) != binding['runtime']:
        raise ValueError('source/runtime changed during input preparation')
    receipt = {**binding, 'status':'verified' if readonly else 'prepared', 'training':False,
               'catalog_sha256':CATALOG_SHA, 'content_sha256':CONTENT_SHA,
               'train_sequence_sha256':_sequence_digest(histories),
               'users':len(histories), 'events':sum(map(len,histories.values())),
               'payload_written':not readonly}
    if readonly:
        return receipt
    output.mkdir(exist_ok=False)
    with (output/'catalog.json').open('xb') as f: f.write(encoded(catalog))
    with (output/'content.npy').open('xb') as f: f.write(buffer.getbuffer())
    with (output/'histories.jsonl').open('xb') as f:
        for user in sorted(histories): f.write(encoded([user, histories[user]]))
    receipt['artifacts'] = {name:digest(output/name) for name in ('catalog.json','content.npy','histories.jsonl')}
    with (output/'manifest.json').open('xb') as f: f.write(encoded(receipt))
    return receipt


def read_bundle(bundle, manifest_sha, source):
    import hashlib
    import io
    import numpy as np
    from vkgr.sid_coll_sgns import _sequence_digest
    bundle = secure(bundle)
    path = secure(bundle/'manifest.json')
    manifest_bytes = path.read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != manifest_sha:
        raise ValueError('input manifest drift')
    value = json.loads(manifest_bytes)
    if (value.get('schema') != 'sgns-subset-physical-input-v1' or value.get('status') != 'prepared'
            or value.get('engineering') is not False or value.get('source') != source):
        raise ValueError('input producer/source binding differs')
    names = ('catalog.json','content.npy','histories.jsonl')
    if set(value.get('artifacts',{})) != set(names) or value.get('records') != fixed_records():
        raise ValueError('fixed input DAG differs')
    for name in names:
        target = secure(bundle/name)
        if digest(target) != value['artifacts'][name]:
            raise ValueError('input payload drift')
    if value['artifacts']['catalog.json'] != CATALOG_SHA or value['artifacts']['content.npy'] != CONTENT_SHA:
        raise ValueError('original catalog/content pin differs')
    catalog_bytes = (bundle/'catalog.json').read_bytes()
    if hashlib.sha256(catalog_bytes).hexdigest() != CATALOG_SHA:
        raise ValueError('loaded catalog payload differs')
    catalog = json.loads(catalog_bytes)
    content = np.load(bundle/'content.npy',allow_pickle=False)
    # Bind the array actually passed into fit, including a swap-and-restore during load.
    content_bytes = io.BytesIO(); np.save(content_bytes,content,allow_pickle=False)
    if hashlib.sha256(content_bytes.getbuffer()).hexdigest() != CONTENT_SHA:
        raise ValueError('loaded content payload differs')
    if len(catalog) != 120415 or content.shape != (120416,64) or content.dtype != np.float32:
        raise ValueError('formal subset shape differs')
    histories = {}
    previous = -1
    consumed = hashlib.sha256()
    with (bundle/'histories.jsonl').open('rb') as f:
        for line in f:
            consumed.update(line)
            user, events = json.loads(line)
            if type(user) is not int or user <= previous or not 0 <= user < 2**32 or not isinstance(events,list):
                raise ValueError('unique ascending user sequence required')
            histories[user] = events
            previous = user
    if consumed.hexdigest() != value['artifacts']['histories.jsonl']:
        raise ValueError('loaded histories payload differs')
    if (_sequence_digest(histories) != value['train_sequence_sha256']
            or len(histories) != value['users'] or sum(map(len,histories.values())) != value['events']):
        raise ValueError('full observed sequence digest/count differs')
    return (catalog, histories, content[1:]), value


def original_fit(bundle, device):
    from vkgr.sid_coll_sgns import fit_sgns
    if device not in ('cpu','cuda'):
        raise ValueError('fixed CPU/CUDA device required')
    catalog, histories, content = bundle
    # All scientific defaults come directly from the original frozen function.
    return fit_sgns(catalog, histories, content, device=device, engineering=False)


def verify_original_fit(bundle, vectors, audit):
    from vkgr.sgns_subset_verify import verify_c1_audit
    catalog, histories, content = bundle
    return verify_c1_audit(catalog,histories,content,vectors,audit,engineering=False)


def fit(bundle, manifest_sha, device, lock, output):
    import hashlib
    import io
    import os
    import numpy as np
    import torch
    from analysis.new_instance_prepare import runtime
    from analysis.ur001_derive_cli import source_identity, guards
    from vkgr.sid_coll_sgns import save_sgns_output
    output = secure(output)
    if output.exists():
        raise FileExistsError('exclusive SGNS output required')
    if ROOT == output or ROOT in output.parents or output in ROOT.parents:
        raise ValueError('output must be outside deployment')
    source = source_identity()
    env = runtime(secure(lock))
    inputs, producer = read_bundle(bundle,manifest_sha,source)
    guards(output,projected_memory=16*1024**3)
    progress('sgns_fit_begin',device=device)
    started = time.monotonic()
    vectors, audit = original_fit(inputs,device)
    elapsed = time.monotonic()-started
    progress('sgns_fit_end',device=device,seconds=elapsed)
    # The original fit already applies the original health gate; no fallback or retry.
    read_bundle(bundle,manifest_sha,source)
    if source_identity() != source:
        raise ValueError('source changed during SGNS')
    guards(output)
    save_sgns_output(output,vectors,audit)
    persisted_bytes = {name:secure(output/name).read_bytes() for name in ('vectors.npy','audit.json')}
    artifacts = {name:hashlib.sha256(raw).hexdigest() for name,raw in persisted_bytes.items()}
    persisted_audit = json.loads(persisted_bytes['audit.json'])
    if persisted_audit != audit:
        raise ValueError('persisted SGNS audit differs from producer')
    progress('sgns_verify_begin',device=device)
    proof = verify_original_fit(inputs,np.load(io.BytesIO(persisted_bytes['vectors.npy']),allow_pickle=False),persisted_audit)
    progress('sgns_verify_end',device=device)
    if source_identity() != source or runtime(secure(lock)) != env:
        raise ValueError('source/runtime changed during SGNS verification')
    if any(digest(secure(output/name)) != sha for name,sha in artifacts.items()):
        raise ValueError('output artifact changed after original verification')
    value = {'schema':'gpu-sgns-subset-stage-v1','status':'verified','verified':True,'engineering':False,
             'authority':'ad1c428','stage':'sgns','device':device,'source':source,'runtime':env,
             'input_manifest_sha256':manifest_sha,'input_producer':producer,'seconds':elapsed,
             'run_id':os.environ.get('VKGR_RUN_ID'),'cuda_version':torch.version.cuda,
             'cublas_workspace_config':os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
             'artifacts':artifacts,
             'old_vectors_pin_equal':artifacts['vectors.npy'] == 'bf840eb0ecb350a4044bbe20a45e4ed82d6402f54d364aad8f17f5f554d582c7',
             'original_audit_verification':proof,
             'subset_four_gate_acceptance':False,'validation_read':False,'test_read':False}
    with (output/'stage-receipt.json').open('xb') as f: f.write(encoded(value))
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    sub.add_parser('dry')
    for command in ('preflight','prepare'):
        p = sub.add_parser(command)
        for name in ('data','embedding-root','lock','output'):
            p.add_argument('--'+name, type=Path, required=True)
    p = sub.add_parser('fit')
    for name in ('bundle','lock','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--bundle-sha',required=True)
    p.add_argument('--device',choices=('cpu','cuda'),required=True)
    args = parser.parse_args(argv)
    if args.command == 'dry':
        value = {'status':'ready','files':len(fixed_records()),'training':False,'payload_read':False}
    elif args.command == 'fit':
        value = fit(args.bundle,args.bundle_sha,args.device,args.lock,args.output)
    else:
        value = prepare(args.data,args.embedding_root,args.lock,args.output,readonly=args.command=='preflight')
    print(json.dumps(value,sort_keys=True)); return 0


if __name__ == '__main__': raise SystemExit(main())
