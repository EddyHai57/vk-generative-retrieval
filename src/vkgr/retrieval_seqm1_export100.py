"""G5 P-side top-100 candidate export from an already verified SEQ-M1 best checkpoint (read-only inference, new file only).

The accepted P runs archived only top-50 predictions. The registered fusion protocol needs 100 candidates per query, so this
module re-runs the *same* validation inference as ``retrieval_seqm1_run.verify_run`` (same data, same model construction, same
``export_vectors_side2`` + float64 ``blockwise_topk``) with depth 100. It is not training: no optimizer, no parameter update.

Boundaries (decision_log 2026-10-18 "G5 freeze"): the single best checkpoint of the named run is read with ``weights_only=True``
and only to run forward inference; ``last.pt`` is never opened; user vectors and item embeddings are never written; the existing
run directory is never modified (output goes to a new directory); test (week 26) has no code path. Correctness gate: the first 50
candidates of every query must equal the run's archived ``predictions-epoch-<best>.json`` bit for bit, otherwise the job stops.
Tie-breaking is item id ascending and scores do not depend on block shape (see ``baselines``), so an unchanged vector set must
reproduce the archived top-50 exactly.
"""
import argparse
import json
import sys
from pathlib import Path
from . import retrieval_b1_run as b1
from . import retrieval_feat02_run as fr
from . import retrieval_seqm1_run as sm
from .data_audit import file_sha256
from .retrieval_ablation_run import read_pinned, strict_json
from .retrieval_php01 import FEATURE

TASK = 'P-SEQ-M1-EXPORT100'
SCHEMA = 'p-seqm1-export100-v1'
KIND = 'p_seqm1_export100'
DEPTH = 100
GATE_DEPTH = 50
PREDICTIONS_NAME = 'top100.json'
EXPORT_MANIFEST_NAME = 'export-manifest.json'
EXPORT_FIELDS = {'schema', 'kind', 'status', 'run_id', 'seed', 'arm', 'depth', 'queries', 'best_epoch', 'train_manifest_sha256',
                 'train_binding_sha256', 'best_checkpoint_sha256', 'original_predictions_ref', 'original_predictions_sha256',
                 'top100_sha256', 'gate', 'candidate_stats', 'script_sha256', 'user_vectors_exported', 'item_embeddings_exported',
                 'checkpoints_opened', 'holdout_split_opened', 'train_source_commit', 'export_source_commit', 'pinned_delta'}
SOURCE_STRICT = ('python', 'dependencies', 'lock_sha')


def candidates(model, data, queries, config, check=None, depth=DEPTH):
    """Top-``depth`` catalog items per validation query; identical pipeline to ``evaluate_vectors`` with a larger k."""
    from .baselines import blockwise_topk
    users, items, _ = fr.export_vectors_side2(model, data['catalog'], queries, data['idmap'], config['batch_size'],
                                              50 if config['history_len'] <= 50 else config['history_len'], config['feature'], check)
    out = []
    for start in range(0, len(users), 32):
        if check: check()
        out.extend(blockwise_topk(users[start:start + 32], items, data['catalog'], depth).tolist())
    return out


def first50_gate(top100, original):
    """Counts only (never candidate content): queries whose first 50 candidates differ from the archived top-50."""
    if len(top100) != len(original):
        raise ValueError('first-50 gate failed: query count mismatch')
    mismatched = sum(1 for new, old in zip(top100, original) if new[:GATE_DEPTH] != old)
    if mismatched:
        raise ValueError(f'first-50 gate failed: {mismatched}/{len(original)} queries differ from the archived top-50')
    return {'first50_equal': True, 'queries': len(original), 'mismatched_queries': 0, 'compared_depth': GATE_DEPTH}


def candidate_stats(top100, catalog, depth):
    allowed = set(catalog)
    lengths = {len(row) for row in top100}
    if lengths != {depth}:
        raise ValueError(f'candidate depth mismatch: {sorted(lengths)} != {depth}')
    if any(len(set(row)) != depth for row in top100):
        raise ValueError('duplicate candidates inside a query')
    if any(item not in allowed for row in top100 for item in row):
        raise ValueError('candidate outside the catalog')
    return {'depth': depth, 'min_unique': depth, 'all_in_catalog': True}


def check_binding(stored, current, delta):
    """Pinned-delta source identity (decision_log 2026-10-18, fail-closed).

    Every binding field except ``source`` must be strictly equal (inputs, config, init, side, data hashes, execution...). For ``source``:
    python, dependencies and lock_sha are strictly equal, and the actual file ledger must equal the training ledger plus exactly the
    pinned new paths with exactly their pinned sha256. Any changed pre-existing file, any unpinned new path, any pinned path that
    already existed in the training ledger fails. Commits differ by construction and are only reported, never a pass condition.
    """
    if type(delta) is not dict or not delta or any(type(k) is not str or type(v) is not str or len(v) != 64 for k, v in delta.items()):
        raise ValueError('pinned delta required')
    if set(stored) != set(current):
        raise ValueError('binding fields differ')
    for key in stored:
        if key != 'source' and stored[key] != current[key]:
            raise ValueError(f'run input/init/side binding mismatch: {key}')
    old, new = stored['source'], current['source']
    if set(old) != set(new):
        raise ValueError('source identity fields differ')
    for key in SOURCE_STRICT:
        if old[key] != new[key]:
            raise ValueError(f'source identity mismatch: {key}')
    if set(old['files']) & set(delta):
        raise ValueError('pinned path already present in the training source ledger')
    if new['files'] != {**old['files'], **delta}:
        raise ValueError('source ledger differs from training ledger plus pinned delta')
    return {'train_source_commit': old['git_commit'], 'export_source_commit': new['git_commit']}


def _paths(manifest_path, output):
    run = Path(manifest_path).parent.resolve(); out = Path(output).resolve()
    if out == run or run in out.parents or out in run.parents:
        raise ValueError('export output must be a new directory outside the existing run directory')
    return run, out


def _inference(manifest_path, manifest_sha, data, config, device, check, inputs_binding, infer=True, delta=None):
    """Shared by export and verify: pinned manifest -> best checkpoint (weights_only) -> top-100 plus the archived top-50."""
    import torch
    value = read_pinned(manifest_path, manifest_sha); run = Path(manifest_path).parent
    if set(value) != sm.MANIFEST_FIELDS or value['schema'] != sm.MANIFEST_SCHEMA or value['kind'] != sm.KIND or value['status'] != 'complete':
        raise ValueError('complete P-SEQ-M1 manifest required')
    synthetic = bool(data.get('synthetic'))
    if value['config'] != config:
        raise ValueError('config mismatch')
    sm.validate_config(config, synthetic)
    if not synthetic and device != 'cuda':
        raise ValueError('real export requires producer CUDA')
    b1.execution_context(device)
    rows, queries = fr._view(data, config['history_len'])
    model = sm._model(data, config, device)
    provenance = check_binding(value['binding'], sm.binding_for4(data, config, rows, queries, model, device, inputs_binding), delta)
    best = value['best']
    if type(best) is not dict or best.get('path') != 'best.pt' or best.get('epoch') != value['selection']['best_epoch']:
        raise ValueError('best checkpoint identity mismatch')
    if file_sha256(run / 'best.pt') != best['sha256']:
        raise ValueError('best checkpoint hash mismatch')
    state = torch.load(run / 'best.pt', map_location='cpu', weights_only=True)
    if state.get('binding_sha256') != b1._digest(value['binding']) or state['step'] != best['step'] or state['epoch'] != best['epoch']:
        raise ValueError('best checkpoint hash/identity mismatch')
    model.load_state_dict(state['model'])
    evaluation = value['evaluations'][best['epoch'] - 1]
    if evaluation['epoch'] != best['epoch']:
        raise ValueError('best evaluation epoch mismatch')
    if Path(evaluation['predictions_ref']).name != evaluation['predictions_ref']:
        raise ValueError('predictions path escape')
    original = read_pinned(run / evaluation['predictions_ref'], evaluation['predictions_sha256'])
    top100 = candidates(model, data, queries, config, check) if infer else None
    return value, best, evaluation, original, top100, provenance


def preflight(manifest_path, manifest_sha, data, config, device='cpu', check=None, inputs_binding=None, delta=None):
    """Real-path read-only preflight: every identity check of export/verify, but no forward pass and nothing written."""
    value, best, evaluation, original, _, provenance = _inference(manifest_path, manifest_sha, data, config, device, check, inputs_binding, infer=False, delta=delta)
    depths = {len(row) for row in original}
    return {'status': 'preflight_ok', 'manifest_sha256': manifest_sha, 'best_epoch': best['epoch'], 'best_checkpoint_sha256': best['sha256'],
            'queries': len(original), 'original_depths': sorted(depths), 'catalog': len(data['catalog']),
            'original_predictions_sha256': evaluation['predictions_sha256'], 'forward_pass_run': False, 'written': False,
            'source_identity': 'pinned_delta_ok', **provenance, 'pinned_delta': dict(sorted(delta.items()))}


def export_run(manifest_path, manifest_sha, data, config, output, device='cpu', check=None, inputs_binding=None, run_id=None, delta=None):
    _, out = _paths(manifest_path, output)
    value, best, evaluation, original, top100, provenance = _inference(manifest_path, manifest_sha, data, config, device, check, inputs_binding, delta=delta)
    depth = min(DEPTH, len(data['catalog']))
    gate = first50_gate(top100, original)
    stats = candidate_stats(top100, data['catalog'], depth)
    out.mkdir(parents=True, exist_ok=False)
    top_sha = b1._write_json_atomic(out / PREDICTIONS_NAME, top100)
    record = {'schema': SCHEMA, 'kind': KIND, 'status': 'complete', 'run_id': run_id or Path(manifest_path).parent.parent.name,
              'seed': config['seed'], 'arm': config['arm'], 'depth': depth, 'queries': len(top100), 'best_epoch': best['epoch'],
              'train_manifest_sha256': manifest_sha, 'train_binding_sha256': b1._digest(value['binding']),
              'best_checkpoint_sha256': best['sha256'], 'original_predictions_ref': evaluation['predictions_ref'],
              'original_predictions_sha256': evaluation['predictions_sha256'], 'top100_sha256': top_sha, 'gate': gate,
              'candidate_stats': stats, 'script_sha256': file_sha256(Path(__file__)), 'user_vectors_exported': False,
              'item_embeddings_exported': False, 'checkpoints_opened': ['best.pt'], 'holdout_split_opened': False, **provenance,
              'pinned_delta': dict(sorted(delta.items()))}
    sha = b1._write_json_atomic(out / EXPORT_MANIFEST_NAME, record)
    return {**record, 'export_manifest_sha256': sha}


def verify_export(export_manifest_path, export_sha, manifest_path, manifest_sha, data, config, device='cpu', check=None, inputs_binding=None, delta=None):
    """Independent recomputation of the whole export from the best checkpoint; compares every stored byte-level identity."""
    record = read_pinned(export_manifest_path, export_sha); out = Path(export_manifest_path).parent
    if set(record) != EXPORT_FIELDS or record['schema'] != SCHEMA or record['kind'] != KIND or record['status'] != 'complete':
        raise ValueError('complete export manifest required')
    if sorted(p.name for p in out.iterdir()) != sorted([PREDICTIONS_NAME, EXPORT_MANIFEST_NAME]):
        raise ValueError('unexpected files in export directory')
    value, best, evaluation, original, top100, provenance = _inference(manifest_path, manifest_sha, data, config, device, check, inputs_binding, delta=delta)
    stored = read_pinned(out / PREDICTIONS_NAME, record['top100_sha256'])
    if stored != top100:
        raise ValueError('stored top-100 differs from independent recomputation')
    depth = min(DEPTH, len(data['catalog']))
    expected = {'train_manifest_sha256': manifest_sha, 'train_binding_sha256': b1._digest(value['binding']), 'best_epoch': best['epoch'],
                'best_checkpoint_sha256': best['sha256'], 'original_predictions_ref': evaluation['predictions_ref'],
                'original_predictions_sha256': evaluation['predictions_sha256'], 'seed': config['seed'], 'arm': config['arm'],
                'depth': depth, 'queries': len(top100), 'gate': first50_gate(top100, original),
                'candidate_stats': candidate_stats(top100, data['catalog'], depth), 'user_vectors_exported': False,
                'item_embeddings_exported': False, 'checkpoints_opened': ['best.pt'], 'holdout_split_opened': False, **provenance,
                'pinned_delta': dict(sorted(delta.items()))}
    for key, wanted in expected.items():
        if record[key] != wanted:
            raise ValueError(f'export manifest field mismatch: {key}')
    return {'status': 'verified', 'run_id': record['run_id'], 'manifest_sha256': manifest_sha, 'export_manifest_sha256': export_sha,
            'top100_sha256': record['top100_sha256'], 'best_epoch': best['epoch'], 'queries': len(top100), 'depth': depth,
            'first50_equal': True}


def parse_delta(items):
    out = {}
    for item in items or []:
        path, _, digest = item.partition('=')
        if not path or path in out or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest) or path.startswith('/') or '..' in Path(path).parts:
            raise ValueError('invalid --pin-new-path')
        out[path] = digest
    if not out:
        raise ValueError('pinned delta required')
    return out


def peak_mib():
    """Measured CUDA peak (allocated and reserved, MiB) of this process; None without CUDA. Reported on stdout only, never in manifests."""
    import torch
    if not torch.cuda.is_available():
        return None
    return {'allocated': round(torch.cuda.max_memory_allocated() / 2 ** 20, 1), 'reserved': round(torch.cuda.max_memory_reserved() / 2 ** 20, 1)}


def main(argv=None):
    parser = argparse.ArgumentParser(description='G5 P-side top-100 export (read-only inference; new version, runner untouched)')
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('export', 'verify', 'preflight'):
        p = sub.add_parser(name); p.add_argument('--inputs', required=True); p.add_argument('--inputs-sha', required=True)
        p.add_argument('--config', required=True); p.add_argument('--config-sha', required=True)
        p.add_argument('--manifest', required=True); p.add_argument('--manifest-sha', required=True)
        p.add_argument('--run-id', required=True); p.add_argument('--pin-new-path', action='append', required=True, metavar='PATH=SHA256')
        if name == 'preflight':
            pass
        elif name == 'export':
            p.add_argument('--output', required=True)
        else:
            p.add_argument('--export-manifest', required=True); p.add_argument('--export-manifest-sha', required=True)
    args = parser.parse_args(argv)
    try:
        config = read_pinned(args.config, args.config_sha); sm.validate_config(config)
        spec = fr.validate_inputs(read_pinned(args.inputs, args.inputs_sha), FEATURE)
        import torch
        if not torch.cuda.is_available():
            raise ValueError('actual CUDA environment required')
        delta = parse_delta(args.pin_new_path)
        if args.run_id != Path(args.manifest).parent.parent.name:
            raise ValueError('run id does not match the run directory')
        if args.command == 'export':
            _paths(args.manifest, args.output)
            if Path(args.output).exists():
                raise ValueError('export output already exists')
        target = Path(args.output) if args.command == 'export' else Path(args.export_manifest).parent if args.command == 'verify' else Path(args.manifest).parent
        check = b1.resource_guard(target, spec['disk_cap_bytes']); check()
        data, _, binding = b1.real_inputs(spec)
        data, extra_binding = fr.apply_feat02(data, spec, config['history_len'])
        inputs_binding = {**binding, **extra_binding, 'inputs_sha256': args.inputs_sha}
        if args.command == 'preflight':
            print(json.dumps({**preflight(args.manifest, args.manifest_sha, data, config, 'cuda', check, inputs_binding, delta), 'run_id': args.run_id, 'cuda_peak_mib': peak_mib()})); return 0
        if args.command == 'export':
            value = export_run(args.manifest, args.manifest_sha, data, config, args.output, 'cuda', check, inputs_binding, args.run_id, delta)
            print(json.dumps({'status': value['status'], 'run_id': value['run_id'], 'manifest_sha256': args.manifest_sha,
                              'export_manifest_sha256': value['export_manifest_sha256'], 'top100_sha256': value['top100_sha256'],
                              'first50_equal': value['gate']['first50_equal'], 'cuda_peak_mib': peak_mib()})); return 0
        result = verify_export(args.export_manifest, args.export_manifest_sha, args.manifest, args.manifest_sha, data, config, 'cuda', check, inputs_binding, delta)
        if result['run_id'] != args.run_id:
            raise ValueError('run id mismatch')
        print(json.dumps({**result, 'cuda_peak_mib': peak_mib()})); return 0
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
        print(f'ERROR P-SEQ-M1-EXPORT100 gate failed: {type(exc).__name__}: {exc}', file=sys.stderr); return 2


if __name__ == '__main__':
    raise SystemExit(main())
