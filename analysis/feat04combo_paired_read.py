"""One-shot validation-only C+D+E paired reader; aggregate stdout, no writes.

Run as a file or python -B - < source. Dependencies are read by fixed SHA before
execution; no __file__, model imports, checkpoint reads or queue submissions.
"""
import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
import types

PHASE2 = Path('/workspace/vkgr-successor-ops-v1/code/analysis/phase2_validation_read.py')
PHASE2_SHA = '548d8373f1a5008c2a83f29f729930892b1793b01e46e4dbf21942c76adb38d8'
COMBO_SOURCE = Path('/workspace/vkgr-feat04combo-p-v1/scripts/feat04combo_ops.py')
COMBO_SOURCE_SHA = 'b0c5d8a043a04b7adbce12c4b1da6f6258beea6e0776f3c43fe09a9cabe2ac70'
OPS = Path('/workspace/vkgr-feat04combo-ops-v1')
SEEDS = (42, 2024, 2025)
STEMS = ('p-feat04combo-cde', 'p-php01-h1-t010', 'p-feat04b-e-q2-behavior')


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def source_bytes(path, pin, reads):
    path = Path(path)
    if path.is_symlink() or path.stat().st_size > 1024 * 1024:
        raise ValueError('unsafe source')
    payload = path.read_bytes()
    sha = hashlib.sha256(payload).hexdigest()
    if sha != pin:
        raise ValueError('source SHA mismatch')
    reads[str(path)] = sha
    return payload


def pinned_module(path, pin, reads):
    payload = source_bytes(path, pin, reads)
    module = types.ModuleType('combo_pinned_phase2')
    exec(compile(payload, str(path), 'exec'), module.__dict__)
    return module


def plain_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def mechanical(pairs):
    if len(pairs) != 3:
        return {'classification': 'UNKNOWN', 'promotion_statistical_condition': False}
    deltas = [p['delta_hits'] for p in pairs]
    mean = sum(deltas) / 3
    count = sum(v >= 6 for v in deltas)
    effective = mean >= 6 and count >= 2
    flag = 'effective' if effective else 'harmful' if mean <= -6 else 'inconclusive' if mean < 6 else 'UNCLASSIFIED_PENDING_KIRO'
    sig = sum(p['mcnemar_exact_two_sided'] < .05 for p in pairs)
    return {'classification': flag, 'deltas': deltas, 'mean_delta_hits': mean,
            'seeds_ge6': count, 'exact_significant_seeds': sig,
            'effective_condition': effective, 'promotion_statistical_condition': effective and sig >= 2,
            'independent_leakage_review': 'NOT_ASSESSED', 'promotion_verdict': 'NOT_ISSUED'}


def check_combo_identity(manifest, entry, config, deployment):
    binding = manifest['binding']
    if manifest['config'] != config or binding['source']['git_commit'] != deployment['commit']:
        raise ValueError('combo config/source identity mismatch')
    if plain_digest(binding) != entry['expected']['expected_binding_sha256']:
        raise ValueError('combo preflight binding mismatch')
    for key, pin in entry['expected']['data_hashes'].items():
        if binding[key] != pin:
            raise ValueError('combo preflight data hash mismatch')
    if entry['job']['seed'] != config['seed'] or entry['job']['run_id'] != f"p-feat04combo-cde-s{config['seed']}":
        raise ValueError('combo job/config identity mismatch')


def candidate_check(predictions, catalog, support_sha, expected_support):
    if support_sha != expected_support:
        raise ValueError('shared support identity mismatch')
    if any(any(type(item) is not int or item not in catalog for item in row) for row in predictions):
        raise ValueError('predicted item outside frozen catalog')


def accepted_archive(r, rid, state, queries, hist, catalog, support_sha):
    # Reuse all original verifier, prediction SHA, role/path, history/query,
    # ordered population, best-epoch selection and R/N recomputation checks.
    value = r.load_archive(rid, queries, hist, state)
    manifest, _ = r.read(r.ROOT / 'runs' / rid / 'out/manifest.json', state['manifest_sha256'])
    if manifest['config']['seed'] != int(rid.rsplit('-s', 1)[1]):
        raise ValueError('run/seed identity mismatch')
    if manifest['binding'].get('synthetic') is not False or manifest['binding'].get('test_gate') != 'closed_hash_only':
        raise ValueError('real validation closed test gate required')
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in manifest['selection']['best_metric']):
        raise ValueError('nonfinite best metrics')
    epoch = manifest['selection']['best_epoch']
    evaluation = next(x for x in manifest['evaluations'] if x['epoch'] == epoch)
    predictions, _ = r.read(r.ROOT / 'runs' / rid / 'out' / evaluation['predictions_ref'], evaluation['predictions_sha256'])
    candidate_check(predictions, catalog, manifest['binding']['support_sha256'], support_sha)
    # Cross-check stored aggregate slices, including OOV structural zeros.
    metrics = evaluation['result']['metrics']
    for key, count in (('all', 2520), ('oov', 623), ('train_observed', 1897)):
        part = metrics[key]
        if any(type(part[name]) not in (int, float) or not math.isfinite(part[name]) for name in ('recall@50', 'ndcg@50')):
            raise ValueError('nonfinite stored metrics')
        hits = 0 if key == 'oov' else sum(value['hits'])
        ndcg = 0 if key == 'oov' else sum(value['ndcg'])
        if part['count'] != count or abs(part['recall@50'] - hits / count) > 1e-9 or abs(part['ndcg@50'] - ndcg / count) > 1e-9:
            raise ValueError('stored validation slice metrics mismatch')
    return value, manifest


def main():
    started = now()
    reads = {}
    r = pinned_module(PHASE2, PHASE2_SHA, reads)
    source_bytes(COMBO_SOURCE, COMBO_SOURCE_SHA, reads)  # Evidence only: never import model ops.
    r.READS.update(reads)
    plan, _ = r.read(OPS / 'plan.json')
    deployment, _ = r.read(OPS / 'deployment.json')
    expected_ids = {f'{STEMS[0]}-s{s}' for s in SEEDS}
    entries = {x['job']['run_id']: x for x in plan['entries']}
    if len(plan['entries']) != 3 or set(entries) != expected_ids:
        raise ValueError('exact registered combo seeds required')
    hdir = r.ROOT / 'runs/h2-prepare/out'
    hist, _ = r.read(hdir / 'manifest.json', r.HISTORY_PIN)
    queries, _ = r.read(hdir / 'validation-l50.json', hist['artifacts']['validation-l50.json'])
    catalog, catalog_sha = r.read(r.ROOT.parent / 'vkgr-p4b-20261007t-pipeline-03/prepared-v3/catalog.json')
    if len(queries) != 2520 or sum(q['oov'] for q in queries) != 623:
        raise ValueError('frozen population mismatch')
    if not isinstance(catalog, list) or len(catalog) != 120415 or len(set(catalog)) != len(catalog) or r.plain_digest(catalog) != hist['catalog_sha256'] or catalog_sha != hist['binding']['catalog_sha']:
        raise ValueError('frozen catalog identity mismatch')
    catalog_set = set(catalog)
    if any(type(q['oov']) is not bool or q['oov'] != (q['target'] not in catalog_set) for q in queries):
        raise ValueError('query target/OOV mismatch')
    hist['_base_queries_canonical_digest'] = hist['binding']['query_sha']
    states = {}; data = {}; summaries = {}; support_sha = None
    catalog = catalog_set
    for seed in SEEDS:
        for stem in STEMS:
            rid = f'{stem}-s{seed}'
            state = r.inspect_run(rid)
            if not state['verified'] or state['terminal']['exit_code'] != 0:
                raise ValueError('train/raw science verifier not accepted: ' + rid)
            # Include raw stdout hashes in the global path->SHA evidence map.
            for job in state['verify']['jobs']:
                for evidence in job['evidence']:
                    r.READS[evidence['path']] = evidence['sha256']
            manifest, _ = r.read(r.ROOT / 'runs' / rid / 'out/manifest.json', state['manifest_sha256'])
            if stem == STEMS[0]:
                entry = entries[rid]
                train_jobs = sorted((r.ROOT / 'queue/claimed').glob('*-' + rid + '.json'))
                if len(train_jobs) != 1 or r.read(train_jobs[0])[0] != entry['job']:
                    raise ValueError('combo claimed train job/plan mismatch')
                config, config_sha = r.read(entry['job']['config'], entry['expected']['config_sha256'])
                check_combo_identity(manifest, entry, config, deployment)
                if state['manifest_path'] != str(r.ROOT / 'runs' / rid / 'out/manifest.json'):
                    raise ValueError('combo manifest location mismatch')
            support_sha = support_sha or manifest['binding']['support_sha256']
            data[rid], _ = accepted_archive(r, rid, state, queries, hist, catalog, support_sha)
            summaries[rid] = data[rid]['summary']
            states[rid] = state
    paired = {}
    for label, control in (('vs_f4_registered_primary', STEMS[1]), ('vs_e_descriptive_only', STEMS[2])):
        paired[label] = {}
        for seed in SEEDS:
            pair = r.paired(data[f'{STEMS[0]}-s{seed}'], data[f'{control}-s{seed}'])
            pair['neither'] = 2520 - pair['both'] - pair['only_new'] - pair['only_control']
            paired[label][str(seed)] = pair
    result = {'read_at_utc': started, 'finished_at_utc': now(), 'states': states,
              'population': {'n': 2520, 'oov': 623, 'non_oov': 1897, 'history_sha256': r.HISTORY_PIN, 'ordered_core_queries_sha256': r.digest(queries), 'support_sha256': support_sha, 'catalog_sha256': hist['catalog_sha256']},
              'archive_summaries': summaries, 'paired': paired,
              'mechanical_vs_f4': mechanical(list(paired['vs_f4_registered_primary'].values())),
              'read_artifacts_sha256': r.READS, 'test_read': False, 'checkpoint_read': False, 'embedding_read': False, 'new_submit': False,
              'limitations': ['Registered primary comparator remains F4-T010; no retrospective switch to M1-mean.',
                'E comparison is descriptive and adds no gate; no pooled p-value or new multiple-comparison rule.',
                'Three seeds evaluate the same 2520 queries; per-seed query bootstrap does not estimate seed or population uncertainty.',
                'D-1: C/D full-train statistics contain training examples own target and later rows; gains can contain own-target information; no claim of strictly temporally isolated item features.',
                'Audit correction: sum_likes 105700 vs 86785 training examples; likes_i equals target_freq_i for 93.0% of items; D like-rate overlaps logQ target frequency.',
                'C9 is a preset report reminder, not a promotion gate; audited p_like=0.2115 <=0.30, reminder not triggered; 0.30 lacks stronger justification.',
                'No leakage-review, promotion, anchor or latency verdict; no test/checkpoint/embedding read and no new submit.']}
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
