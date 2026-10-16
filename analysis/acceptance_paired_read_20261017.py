"""Read existing validation archives once; emit aggregate JSON only.

No writes, model imports, checkpoint loads, queue changes or test reads.
The frozen phase2 reader supplies raw-verifier/ordered archive checks and the
unchanged 10000-query-bootstrap algorithm. p_pool remains the registered gate for multi-interest; supplemental for combination.
"""
import datetime
import hashlib
import json
import sys
import math
from pathlib import Path
import types

PHASE2 = None
PHASE2_SHA = '548d8373f1a5008c2a83f29f729930892b1793b01e46e4dbf21942c76adb38d8'
COMBO_OPS = None
SEEDS = (42, 2024, 2025)
COMPARISONS = (
    ('combo_vs_f4_registered', 'p-feat04combo-cde', 'p-php01-h1-t010'),
    ('combo_vs_m1_descriptive', 'p-feat04combo-cde', 'p-seqm1-mean'),
    ('m2_vs_m1_registered', 'p-seqm2-k4', 'p-seqm1-mean'))


# Previously accepted immutable manifests: also pins config/source/science identity.
MANIFEST_PINS = {'p-feat04combo-cde-s2024': '1b3616ad8c76fcd038c9ec359a9918926d22229656324511820f1dced1a1ed41', 'p-feat04combo-cde-s2025': 'cdb3499571b6dda295207f10fbd9c860a83140b00307d3c17efe8fcf7d2dce87', 'p-feat04combo-cde-s42': '9557cca6f9a8dc5e0bb077fa23727f2a351643142c20978a1378a7da937fdb92', 'p-php01-h1-t010-s2024': 'b2a5684687cb7a28af0ebb8adefc9f5678b2fb64550488c44192dbb506c7a8a3', 'p-php01-h1-t010-s2025': '6fbebcea816866a94f71f2a3c0c8f31ede848d3727b596509d0ce6b271795976', 'p-php01-h1-t010-s42': '1fcdd6ba4e058a2ce7ba2f0fcb361b88e191152bc47e235b6b731e94a131c2b5', 'p-seqm1-mean-s2024': '8387b60206f116a3c4958e7a0a8ac597b829077b2013f8f0e0e8136a404ba9d8', 'p-seqm1-mean-s2025': '783bbd9e101476383a711f62df6de54e139c156a903ccdc6643e57eacbc16ccb', 'p-seqm1-mean-s42': 'bdb0e39d6abb721ac8de78dbdd11ebf2da2614b0e25292c3154ca096ec8cdc6e', 'p-seqm2-k4-s42': 'a07faf79e00983a450750cde78c22934ea0dbb46516beeb4006b925755a144d8', 'p-seqm2-k4-s2024': '31ac1df62299e4543b167a46b4596797f1b127cf05219c83ab10ebecf7d4d0d9', 'p-seqm2-k4-s2025': 'b41e32a451a64030eff0fdc8da0714d92c3118ad4f0aab0a489b3638eb21fea5'}

def check_manifest_pin(rid, actual):
    if rid not in MANIFEST_PINS or actual != MANIFEST_PINS[rid]:
        raise ValueError("accepted archive identity mismatch")


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def pinned_module(path, pin, reads):
    path = Path(path)
    if path.is_symlink() or path.stat().st_size > 1024 * 1024:
        raise ValueError('unsafe source')
    payload = path.read_bytes()
    sha = hashlib.sha256(payload).hexdigest()
    if sha != pin:
        raise ValueError('source SHA mismatch')
    reads[str(path)] = sha
    module = types.ModuleType('acceptance_pinned_phase2')
    exec(compile(payload, str(path), 'exec'), module.__dict__)
    return module


def mcnemar(b, c):
    if any(type(v) is not int or v < 0 for v in (b, c)):
        raise ValueError('discordants must be nonnegative integers')
    n = b + c
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)


def pool_summary(pairs, registered_gate=False):
    if set(pairs) != {str(s) for s in SEEDS}:
        raise ValueError('exactly three registered seeds required')
    for p in pairs.values():
        mcnemar(p['only_new'], p['only_control'])
        if p['delta_hits'] != p['only_new'] - p['only_control']:
            raise ValueError('inconsistent discordant counts')
    b = sum(p['only_new'] for p in pairs.values())
    c = sum(p['only_control'] for p in pairs.values())
    return {'only_new': b, 'only_control': c, 'p_pool': mcnemar(b, c),
            'definition': 'two-sided exact binomial on sum(b_s), sum(c_s)',
            'independent_observations': False, 'promotion_gate': registered_gate,
            'mean_delta_hits': sum(p['delta_hits'] for p in pairs.values()) / 3,
            'seeds_delta_ge6': sum(p['delta_hits'] >= 6 for p in pairs.values()),
            'seeds_exact_p_lt_005': sum(p['mcnemar_exact_two_sided'] < .05 for p in pairs.values())}


def check_manifest(rid, manifest):
    binding = manifest['binding']
    if manifest['config']['seed'] != int(rid.rsplit('-s', 1)[1]):
        raise ValueError('run/seed identity mismatch')
    if binding.get('synthetic') is not False or binding.get('test_gate') != 'closed_hash_only':
        raise ValueError('real validation closed test gate required')
    if any(type(v) not in (int, float) or not math.isfinite(v)
           for v in manifest['selection']['best_metric']):
        raise ValueError('nonfinite best metrics')


def accepted_archive(r, rid, state, queries, hist, catalog, support_sha):
    # load_archive checks raw verifier, ordered query/history/catalog binding,
    # best selection, predictions SHA/path/shape/uniqueness, OOV and R/N replay.
    value = r.load_archive(rid, queries, hist, state)
    manifest, _ = r.read(r.ROOT / 'runs' / rid / 'out/manifest.json', state['manifest_sha256'])
    check_manifest(rid, manifest)
    if manifest['binding']['support_sha256'] != support_sha:
        raise ValueError('shared support identity mismatch')
    epoch = manifest['selection']['best_epoch']
    evaluation = next(x for x in manifest['evaluations'] if x['epoch'] == epoch)
    predictions, _ = r.read(r.ROOT / 'runs' / rid / 'out' / evaluation['predictions_ref'], evaluation['predictions_sha256'])
    if any(any(type(item) is not int or item not in catalog for item in row) for row in predictions):
        raise ValueError('predicted item outside frozen catalog')
    for key, count in (('all', 2520), ('oov', 623), ('train_observed', 1897)):
        part = evaluation['result']['metrics'][key]
        hits = 0 if key == 'oov' else sum(value['hits'])
        ndcg = 0 if key == 'oov' else sum(value['ndcg'])
        if (part['count'] != count or
            any(type(part[name]) not in (int, float) or not math.isfinite(part[name]) for name in ('recall@50', 'ndcg@50')) or
            abs(part['recall@50'] - hits / count) > 1e-9 or abs(part['ndcg@50'] - ndcg / count) > 1e-9):
            raise ValueError('stored validation slice metrics mismatch')
    return value, manifest


def check_combo(r, rid, manifest, entries, deployment):
    entry = entries[rid]
    jobs = sorted((r.ROOT / 'queue/claimed').glob('*-' + rid + '.json'))
    if len(jobs) != 1 or r.read(jobs[0])[0] != entry['job']:
        raise ValueError('combo claimed train job/plan mismatch')
    config, _ = r.read(entry['job']['config'], entry['expected']['config_sha256'])
    binding = manifest['binding']
    if manifest['config'] != config or binding['source']['git_commit'] != deployment['commit']:
        raise ValueError('combo config/source mismatch')
    if r.plain_digest(binding) != entry['expected']['expected_binding_sha256']:
        raise ValueError('combo preflight binding mismatch')
    if any(binding[k] != v for k, v in entry['expected']['data_hashes'].items()):
        raise ValueError('combo data identity mismatch')



def configure_paths(paths):
    global PHASE2, COMBO_OPS
    if not isinstance(paths, dict) or set(paths) != {'combo_ops', 'phase2'}:
        raise ValueError("exact private path keys required")
    if any(not isinstance(v, str) or not Path(v).is_absolute() or ".." in Path(v).parts for v in paths.values()):
        raise ValueError("absolute private locations required")
    PHASE2 = Path(paths['phase2'])
    COMBO_OPS = Path(paths['combo_ops'])


def main():
    configure_paths(json.loads(sys.stdin.read(65537)))
    started = utc()
    source_reads = {}
    r = pinned_module(PHASE2, PHASE2_SHA, source_reads)
    r.READS.update(source_reads)
    plan, _ = r.read(COMBO_OPS / 'plan.json')
    deployment, _ = r.read(COMBO_OPS / 'deployment.json')
    entries = {x['job']['run_id']: x for x in plan['entries']}
    if len(plan['entries']) != 3 or set(entries) != {f'p-feat04combo-cde-s{s}' for s in SEEDS}:
        raise ValueError('exact registered combo seeds required')
    hdir = r.ROOT / 'runs/h2-prepare/out'
    hist, _ = r.read(hdir / 'manifest.json', r.HISTORY_PIN)
    queries, _ = r.read(hdir / 'validation-l50.json', hist['artifacts']['validation-l50.json'])
    catalog, catalog_sha = r.read(r.ROOT.parent / 'vkgr-p4b-20261007t-pipeline-03/prepared-v3/catalog.json')
    if len(queries) != 2520 or sum(q['oov'] for q in queries) != 623:
        raise ValueError('frozen population mismatch')
    if (not isinstance(catalog, list) or len(catalog) != 120415 or len(set(catalog)) != len(catalog)
        or r.plain_digest(catalog) != hist['catalog_sha256'] or catalog_sha != hist['binding']['catalog_sha']):
        raise ValueError('frozen catalog identity mismatch')
    catalog = set(catalog)
    if any(type(q['oov']) is not bool or q['oov'] != (q['target'] not in catalog) for q in queries):
        raise ValueError('query target/OOV mismatch')
    hist['_base_queries_canonical_digest'] = hist['binding']['query_sha']
    stems = sorted({stem for _, new, ctl in COMPARISONS for stem in (new, ctl)})
    states, data, identities = {}, {}, {}
    support_sha = None
    for seed in SEEDS:
        for stem in stems:
            rid = f'{stem}-s{seed}'
            state = r.inspect_run(rid)
            check_manifest_pin(rid, state.get('manifest_sha256'))
            if not state['verified'] or (state['terminal'] or {}).get('exit_code') != 0:
                raise ValueError('train/raw science verifier not accepted: ' + rid)
            for job in state['verify']['jobs']:
                for evidence in job['evidence']:
                    r.READS[evidence['path']] = evidence['sha256']
            manifest, _ = r.read(r.ROOT / 'runs' / rid / 'out/manifest.json', state['manifest_sha256'])
            support_sha = support_sha or manifest['binding']['support_sha256']
            data[rid], manifest = accepted_archive(r, rid, state, queries, hist, catalog, support_sha)
            if stem == 'p-feat04combo-cde':
                check_combo(r, rid, manifest, entries, deployment)
            identities[rid] = {'config': manifest['config'], 'binding': manifest['binding'],
                               'manifest_sha256': state['manifest_sha256'], 'summary': data[rid]['summary']}
            states[rid] = state
    comparisons = {}
    for label, new, ctl in COMPARISONS:
        pairs = {}
        for seed in SEEDS:
            p = r.paired(data[f'{new}-s{seed}'], data[f'{ctl}-s{seed}'])
            if abs(mcnemar(p['only_new'], p['only_control']) - p['mcnemar_exact_two_sided']) > 1e-12:
                raise ValueError('independent exact McNemar cross-check failed')
            p['neither'] = 2520 - p['both'] - p['only_new'] - p['only_control']
            pairs[str(seed)] = p
        comparisons[label] = {'per_seed': pairs, 'pooled': pool_summary(pairs, label == 'm2_vs_m1_registered')}
    print(json.dumps({'read_at_utc': started, 'finished_at_utc': utc(), 'states': states,
        'population': {'n': 2520, 'oov': 623, 'catalog': 120415, 'history_sha256': r.HISTORY_PIN,
                       'ordered_core_queries_sha256': r.digest(queries), 'support_sha256': support_sha},
        'identities': identities, 'comparisons': comparisons, 'read_artifacts_sha256': r.READS,
        'test_read': False, 'checkpoint_read': False, 'embedding_read': False, 'new_submit': False,
        'limitations': [
            'FEAT combination registered primary stays F4; M1 comparison is descriptive only.',
            'p_pool sums seed discordants; repeated queries/users across seeds are correlated, so it is optimistic and not independent evidence.',
            'For FEAT-04 p_pool is a supplemental report requested 2026-10-17, not a gate. For M2 it retains the preregistered M1 section8.1 definition and threshold.',
            'Bootstrap reuses pinned phase2: default_rng(0), 10x1000 paired query resamples, shared R/N indices, percentile 95% CI; not seed/user/population uncertainty.',
            'D-1 full-train own-target and later-train information retained; C9 remains a report item.',
            'No leakage-review, promotion, anchor or latency verdict.']}, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
