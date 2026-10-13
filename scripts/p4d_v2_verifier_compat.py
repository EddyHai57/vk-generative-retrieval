"""Explicit v2 producer plus narrowly versioned stats verifier compatibility.

The original producer files and module remain untouched. A separate namespace
uses identical function code objects except the two declared stats comparisons.
Every real CLI invocation attests the original complete transport and adapter.
"""
import ast
import hashlib
import inspect
import json
import re
import sys
import types
from pathlib import Path

PRODUCER_SOURCE = '2b8658b06ece14ca703f1b33cf84d8f24a096b9748d1f91ff91a3ce640bd2848'
EXPERIMENT_SHA = '4b29d02710582a933c2aed8fc908f0113ee70a7111f09da112c6f5a935f7caa7'
ENTRY_SHA = '87567fe8c2807cac752237b5da090314a02c012c8ec7c3fc2e4b853f803a7e1e'
LEDGER_SHA = '10a25620214bef7e94243ca242ac413ea743db0d7319a9a1197a7520e093995d'
REPLACEMENTS = {
    'verify_session': ("state['stats']", "value['stats']"),
    'verify_draw_provenance': ("value['stats']", 'actual'),
}

def digest(data):
    return hashlib.sha256(data).hexdigest()

def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()

def stats_bytes(value):
    """Only legal homogeneous int or canonical JSON epoch keys are normalized."""
    from vkgr.data_audit import canonical_bytes
    if type(value) is not dict or set(value) != {'drawn', 'skipped', 'epochs'}:
        raise ValueError('invalid stats structure')
    if any(type(value[k]) is not int or value[k] < 0 for k in ('drawn', 'skipped')) or value['skipped'] > value['drawn']:
        raise ValueError('invalid stats counts')
    epochs = value['epochs']
    if type(epochs) is not dict or len({type(k) for k in epochs}) > 1:
        raise ValueError('mixed or invalid epoch key types')
    normalized = {}
    for key, counts in epochs.items():
        if type(key) is int:
            if not 0 <= key < 30:
                raise ValueError('invalid integer epoch key')
            name = str(key)
        elif type(key) is str and re.fullmatch(r'0|[1-9][0-9]*', key) and int(key) < 30:
            name = key
        else:
            raise ValueError('invalid JSON epoch key')
        if name in normalized or type(counts) is not list or len(counts) != 2 or any(type(c) is not int for c in counts) or not 0 <= counts[1] <= counts[0]:
            raise ValueError('invalid epoch counts or key collision')
        normalized[name] = counts
    return canonical_bytes({**value, 'epochs': normalized})

def compatible_module(producer):
    """Return a separate namespace and exact two-function AST change proof."""
    source_path = Path(producer.__file__).resolve()
    if digest(source_path.read_bytes()) != EXPERIMENT_SHA:
        raise ValueError('only reviewed v2 experiment source is compatible')
    namespace = dict(producer.__dict__)
    namespace['_p4d_v2_stats_bytes'] = stats_bytes
    unchanged = {}
    # Original code objects, defaults and closure are retained; only globals point
    # at this explicit compatibility namespace, so resume/select use its validators.
    for name, value in producer.__dict__.items():
        if isinstance(value, types.FunctionType) and value.__module__ == producer.__name__:
            clone = types.FunctionType(value.__code__, namespace, name, value.__defaults__, value.__closure__)
            clone.__kwdefaults__ = value.__kwdefaults__
            clone.__module__ = value.__module__
            namespace[name] = clone
            if name not in REPLACEMENTS:
                unchanged[name] = digest(ast.dump(ast.parse(inspect.getsource(value)), include_attributes=False).encode())
    changes = {}
    for name, expressions in REPLACEMENTS.items():
        tree = ast.parse(inspect.getsource(getattr(producer, name)))
        original = ast.dump(tree, include_attributes=False)
        targets = {ast.dump(ast.parse(text, mode='eval').body, include_attributes=False) for text in expressions}
        count = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'canonical_bytes' and len(node.args) == 1 and not node.keywords and ast.dump(node.args[0], include_attributes=False) in targets:
                node.func.id = '_p4d_v2_stats_bytes'
                count += 1
        if count != 2:
            raise ValueError('v2 stats replacement proof mismatch')
        changed = ast.dump(tree, include_attributes=False)
        # Reverting only the declared names must recover the entire original AST.
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == '_p4d_v2_stats_bytes':
                node.id = 'canonical_bytes'
        if ast.dump(tree, include_attributes=False) != original:
            raise ValueError('undeclared validator AST change')
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'canonical_bytes' and len(node.args) == 1 and not node.keywords and ast.dump(node.args[0], include_attributes=False) in targets:
                node.func.id = '_p4d_v2_stats_bytes'
        exec(compile(tree, str(Path(__file__).resolve()) + ':' + name, 'exec'), namespace)
        changes[name] = {'original_ast_sha256': digest(original.encode()), 'compatible_ast_sha256': digest(changed.encode()), 'stats_calls_replaced': count, 'expressions': list(expressions)}
    for name in unchanged:
        if namespace[name].__code__ is not getattr(producer, name).__code__:
            raise ValueError('producer algorithm code object changed')
    module = types.ModuleType('p4d_explicit_v2_compatibility')
    module.__dict__.update(namespace)
    proof = {'schema_version': 1, 'task_id': 'P4D', 'trainer_source': PRODUCER_SOURCE, 'experiment_file_sha256': EXPERIMENT_SHA, 'adapter_sha256': digest(Path(__file__).read_bytes()), 'changed_validators': changes, 'unchanged_function_ast_sha256': unchanged, 'unchanged_function_code_objects': True, 'scope': 'strict stats epoch-key representation only; original model/training/export/rank/metric code unchanged'}
    proof['proof_sha256'] = digest(encoded(proof))
    return module, proof

def check_producer(root, ledger_path):
    """Pin the entire reviewed 105-file producer, never an arbitrary source."""
    from vkgr.manifest import read_json, source_fingerprint, ROOT
    root = Path(root).resolve()
    if ROOT.resolve() != root:
        raise ValueError('imported producer root mismatch')
    ledger = read_json(Path(ledger_path))
    entries = ledger['public_files']
    if ledger['source_fingerprint'] != PRODUCER_SOURCE or len(entries) != 105 or digest(encoded(entries)) != LEDGER_SHA:
        raise ValueError('reviewed v2 complete transport ledger required')
    for entry in entries:
        relative = Path(entry['path'])
        if relative.is_absolute() or '..' in relative.parts or not relative.parts:
            raise ValueError('unsafe producer ledger path')
        path = root / relative
        if any(parent.is_symlink() for parent in [path, *path.parents] if parent != root.parent) or not path.is_file() or path.stat().st_size != entry['bytes'] or digest(path.read_bytes()) != entry['sha256']:
            raise ValueError('producer transport mismatch')
    if source_fingerprint() != PRODUCER_SOURCE:
        raise ValueError('producer source mismatch')
    origin_path = root / 'cache/p4d-source-entry.json'
    if origin_path.is_symlink() or digest(origin_path.read_bytes()) != ENTRY_SHA:
        raise ValueError('reviewed v2 transport origin bytes mismatch')
    origin = read_json(origin_path)
    if origin['source_fingerprint'] != PRODUCER_SOURCE:
        raise ValueError('producer transport origin mismatch')
    return {'trainer_source': PRODUCER_SOURCE, 'complete_transport_ledger_sha256': LEDGER_SHA, 'complete_transport_files': len(entries), 'transport_origin_sha256': ENTRY_SHA, 'complete_transport_members': len(entries) + 1}

def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--producer-root', required=True)
    parser.add_argument('--ledger', required=True)
    parser.add_argument('--invocation', required=True)
    parser.add_argument('producer_arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    arguments = args.producer_arguments
    if arguments and arguments[0] == '--':
        arguments = arguments[1:]
    if not arguments or arguments[0] not in ('train', 'verify', 'select'):
        raise ValueError('compatibility applies only to train/resume/verify/select')
    root = Path(args.producer_root).resolve()
    sys.path.insert(0, str(root / 'src'))
    from vkgr import retrieval_experiment as producer
    transport = check_producer(root, args.ledger)
    module, proof = compatible_module(producer)
    invocation = Path(args.invocation)
    result_path = invocation.with_name(invocation.name + '.result.json')
    if not invocation.parent.is_dir() or invocation.exists() or invocation.is_symlink() or result_path.exists() or result_path.is_symlink():
        raise FileExistsError('exclusive invocation/result sidecars required before producer entry')
    record = {'schema_version': 1, 'task_id': 'P4D', 'kind': 'explicit_v2_stats_compatibility_invocation', **transport, 'proof': proof, 'producer_arguments': arguments, 'command': [sys.executable, str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)], 'test_gate': 'closed'}
    with invocation.open('xb') as stream:
        stream.write(encoded(record))
    # This sidecar is durable before any model, optimizer or producer command.
    exit_code = module.main(arguments)
    receipt = {'schema_version': 1, 'task_id': 'P4D', 'kind': 'explicit_v2_stats_compatibility_result', 'invocation_sha256': digest(invocation.read_bytes()), 'trainer_source': PRODUCER_SOURCE, 'adapter_sha256': proof['adapter_sha256'], 'proof_sha256': proof['proof_sha256'], 'producer_exit': exit_code, 'status': 'pass' if exit_code == 0 else 'failed_stop'}
    with result_path.open('xb') as stream:
        stream.write(encoded(receipt))
    return exit_code

if __name__ == '__main__':
    raise SystemExit(main())
