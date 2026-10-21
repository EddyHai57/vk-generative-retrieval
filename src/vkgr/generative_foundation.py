"""CPU synthetic compatibility adapter; not a real-data training protocol."""
import numpy as np
from .baselines import metrics
from .data_audit import build_queries
from .retrieval_data import prefix_examples, catalog_map, validate_example, content_join
from .sid import SIDCatalog
from .tiger import SIDTokens, TigerModel, decode_history

def prepare_fixture(value):
    _validate_fixture(value)
    ids = value['catalog']
    catalog = SIDCatalog(np.array(ids, dtype=np.uint32), np.array(value['sids'], dtype=np.uint32), 3, 3, [])
    partitions = [(p['week'],p['rows']) for p in value['train_partitions']]
    examples = list(prefix_examples(partitions))
    if not examples: raise ValueError('synthetic training examples required')
    for index,row in enumerate(examples): validate_example(row,catalog_map(ids),index)
    validation = value['validation']
    queries = build_queries(partitions, (validation['week'],validation['rows']), {'min_history':1,'max_history':50}, set(ids))['queries']
    features = content_join(ids,np.array(ids,dtype=np.uint32),np.array(value['features'],dtype=np.float16))[1:]
    if not queries: raise ValueError('synthetic validation queries required')
    for item in ids:
        if catalog.decode(catalog.encode(item)) != item: raise ValueError('SID identity mismatch')
    return {'codec':SIDTokens(catalog),'examples':examples,'queries':queries,'features':features}


def _validate_rows(rows):
    if type(rows) is not list or not 1 <= len(rows) <= 4096:
        raise ValueError('bounded synthetic rows required')
    for row in rows:
        if type(row) is not dict or set(row) != {'user_id','item_id','like'}:
            raise ValueError('invalid synthetic row schema')
        if any(type(row[k]) is not int or not 0 <= row[k] < 2**32 for k in ('user_id','item_id')) or type(row['like']) is not bool:
            raise ValueError('invalid synthetic interaction')


def _validate_fixture(value):
    expected = {'schema_version','synthetic','protocol','catalog','sids','features','train_partitions','validation'}
    if type(value) is not dict or set(value) != expected or type(value['schema_version']) is not int or value['schema_version'] != 1 or value['synthetic'] is not True or value['protocol'] != 'traditional-like-v1':
        raise ValueError('synthetic-only schema; test gate closed')
    ids = value['catalog']
    if type(ids) is not list or not 1 <= len(ids) <= 64:
        raise ValueError('bounded synthetic catalog required')
    catalog_map(ids)
    sids = value['sids']
    if type(sids) is not list or len(sids) != len(ids) or any(type(s) is not list or len(s) != 4 or any(type(x) is not int or x < 0 for x in s) or any(x >= 3 for x in s[:3]) or s[3] >= len(ids) for s in sids):
        raise ValueError('invalid synthetic SID')
    leaves = {}
    for sid in sids:
        leaf = tuple(sid[:3]); expected_suffix = leaves.get(leaf,0)
        if sid[3] != expected_suffix: raise ValueError('SID collision suffix identity mismatch')
        leaves[leaf] = expected_suffix+1
    features = value['features']
    if type(features) is not list or len(features) != len(ids) or any(type(r) is not list or len(r) != 64 or any(type(x) not in (int,float) or not np.isfinite(x) or abs(x)>65504 for x in r) for r in features):
        raise ValueError('invalid synthetic content features')
    partitions = value['train_partitions']
    if type(partitions) is not list or not 1 <= len(partitions) <= 25:
        raise ValueError('invalid synthetic train partitions')
    previous = -1; exposures = set(); count = 0
    for part in partitions:
        if type(part) is not dict or set(part) != {'week','rows'} or type(part['week']) is not int or not previous < part['week'] <= 24:
            raise ValueError('future/train ordering conflict')
        previous = part['week']; _validate_rows(part['rows']);count += len(part['rows'])
        exposures.update(row['item_id'] for row in part['rows'])
    if count > 4096 or exposures != set(ids): raise ValueError('catalog/train exposure identity mismatch')
    validation = value['validation']
    if type(validation) is not dict or set(validation) != {'week','rows'} or type(validation['week']) is not int or validation['week'] != 25:
        raise ValueError('validation-only; test gate closed')
    _validate_rows(validation['rows'])
    if len(validation['rows']) > 64: raise ValueError('bounded synthetic query population required')


def evaluate_items(predictions, queries, catalog):
    """Collision-corrected item IDs only, retaining OOV in the denominator."""
    return metrics(predictions, queries, catalog, (10,20,50))


def _parameter_hash(model):
    import hashlib
    digest=hashlib.sha256()
    for name,parameter in model.named_parameters():
        digest.update(name.encode());digest.update(parameter.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def run_fixture(value):
    """One bounded synthetic epoch; no old runner, cursor, deadline or test."""
    import torch
    prepared=prepare_fixture(value);codec=prepared['codec'];examples=prepared['examples']
    if len(examples)>4: raise ValueError('CPU foundation limited to two batches')
    # Fork only CPU RNG; do not inspect or initialize CUDA.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model=TigerModel(codec.vocab_size,16,4,1,1,32).cpu()
        initial=_parameter_hash(model)
        optimizer=torch.optim.AdamW(model.parameters(),lr=0.001,weight_decay=0.0001)
        order=torch.randperm(len(examples),generator=torch.Generator().manual_seed(42))
        losses=[];sizes=[]
        for indices in order.split(2):
            batch=[examples[i] for i in indices.tolist()]
            source,decoder,labels=codec.batch([r['history'] for r in batch],[r['target'] for r in batch],device='cpu')
            model.train();optimizer.zero_grad(set_to_none=True)
            loss=torch.nn.functional.cross_entropy(model(source,decoder).flatten(0,1),labels.flatten())
            if not torch.isfinite(loss): raise ValueError('nonfinite synthetic loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            losses.append(float(loss.detach()));sizes.append(len(batch))
        final=_parameter_hash(model)
        if final==initial: raise ValueError('no actual parameter update')
        predictions=[];audits=[]
        for query in prepared['queries']:
            predicted,audit=decode_history(model,codec,query['history'],beam=50)
            predictions.append(predicted);audits.append(audit)
    return {'device':'cpu','seed':42,'drawn_example_ids':order.tolist(),'batch_sizes':sizes,'losses':losses,
            'initial_parameters_sha256':initial,'final_parameters_sha256':final,
            'predictions':predictions,'decode_audits':audits,
            'metrics':evaluate_items(predictions,prepared['queries'],value['catalog'])}


def build_manifest():
    """Bind actual consumer source and deterministic synthetic computation."""
    import hashlib
    import platform
    import subprocess
    import torch
    import pyarrow
    import yaml
    from pathlib import Path
    from .manifest import read_json
    from .data_audit import canonical_bytes, file_sha256
    from .retrieval_data import validate_contract
    root=Path(__file__).resolve().parents[2]
    validate_contract(root)
    fixture=root/'tests/fixtures/g0_synthetic.json'
    if fixture.is_symlink(): raise ValueError('synthetic fixture symlink forbidden')
    data=read_json(fixture);prepared=prepare_fixture(data);result=run_fixture(data)
    def digest(value):return hashlib.sha256(canonical_bytes(value)).hexdigest()
    source_paths=sorted([p.relative_to(root).as_posix() for p in (root/'src/vkgr').glob('*.py')]+
                        ['pyproject.toml','uv.lock','configs/traditional.yaml','docs/data_contract.md',
                         'artifacts/p3d-retrieval-contract-manifest.json','tests/test_generative_foundation.py'])
    source_files={path:file_sha256(root/path) for path in source_paths}
    return {'schema_version':1,'task_id':'G0','boundary':'CPU synthetic engineering foundation; no model quality or matched-budget experiment',
            'test_gate':'closed','git_base':subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
            'consumer_source':{'files':source_files,'sha256':digest(source_files),'dirty_overlay':True},
            'environment':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,
                           'pyarrow':pyarrow.__version__,'PyYAML':yaml.__version__,'device':'cpu','torch_threads':torch.get_num_threads()},
            'config':{'seed':42,'d_model':16,'nhead':4,'encoder_layers':1,'decoder_layers':1,'ff':32,'dropout':0,
                      'lr':0.001,'weight_decay':0.0001,'gradient_clip':1,'batch_size':2,'max_steps':2,
                      'epoch_tail':'no_fill','beam':50,'ks':[10,20,50],'split':'synthetic_validation'},
            'input_hashes':{'fixture':file_sha256(fixture),'catalog':digest(data['catalog']),
                            'sid_map':digest(data['sids']),'features_float32':hashlib.sha256(prepared['features'].tobytes()).hexdigest(),
                            'examples':digest(prepared['examples']),'targets':digest([r['target'] for r in prepared['examples']]),
                            'queries':digest(prepared['queries'])},
            'computation_hashes':{'predictions':digest(result['predictions']),'result':digest(result),
                                  'initial_parameters':result['initial_parameters_sha256'],'final_parameters':result['final_parameters_sha256']},
            'summary':{'training_example_count':len(prepared['examples']),'query_count':len(prepared['queries']),
                       'catalog_count':len(data['catalog']),'actual_steps':len(result['losses']),
                       'batch_sizes':result['batch_sizes'],'metrics':result['metrics'],'decode_audits':result['decode_audits']}}


def write_manifest(path):
    from pathlib import Path
    from .data_audit import canonical_bytes
    path=Path(path)
    if path.exists() or path.is_symlink(): raise FileExistsError('exclusive G0 manifest already exists')
    value=build_manifest()
    with path.open('xb') as stream:stream.write(canonical_bytes(value))
    return value


def verify_manifest(path,expected_sha256):
    from pathlib import Path
    from .data_audit import file_sha256, canonical_bytes
    from .manifest import read_json
    path=Path(path)
    if path.is_symlink() or file_sha256(path)!=expected_sha256: raise ValueError('G0 manifest hash mismatch')
    value=read_json(path)
    if canonical_bytes(value)!=canonical_bytes(build_manifest()): raise ValueError('G0 source/input/computation replay mismatch')
    return value


def main(argv=None):
    import argparse
    from .data_audit import file_sha256
    parser=argparse.ArgumentParser(description='G0 CPU synthetic foundation; test gate closed')
    commands=parser.add_subparsers(dest='command',required=True)
    run=commands.add_parser('run',help='run bundled synthetic fixture on CPU')
    run.add_argument('--output',required=True)
    verify=commands.add_parser('verify',help='verify hash and replay bundled synthetic run')
    verify.add_argument('--manifest',required=True);verify.add_argument('--sha256',required=True)
    args=parser.parse_args(argv)
    try:
        if args.command=='run':
            write_manifest(args.output);print('G0 CPU synthetic manifest SHA256 '+file_sha256(args.output))
        else:
            verify_manifest(args.manifest,args.sha256);print('G0 source/input/CPU computation/item metrics replay verified; test gate closed')
    except (OSError,ValueError):
        parser.exit(1,'G0 synthetic input/source/output gate failed\n')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
