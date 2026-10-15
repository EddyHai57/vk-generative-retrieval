"""One-shot, read-only validation receipt analysis; no model or tensor imports.

Default mode computes paired statistics and P/G hit unions from verified archives.
--metadata inspects only fixed run metadata and verifier receipts.
The source can be sent through the approved transport on stdin; stdout is aggregate
JSON only. No writes, queue submissions, checkpoint reads or embedding reads.
"""
import datetime
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path('/workspace/vkgr-infra01-root')
HISTORY_PIN = '8b929d10c41cebbac3976398ad6f48f74f4d4441675d6b054fd19ec0c95c8d43'
SEEDS = (42, 2024, 2025)
N = 2520
READS = {}


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def plain_digest(value):
    return hashlib.sha256(canonical(value)[:-1]).hexdigest()


def archive_role(value, expected):
    if value != expected:
        raise ValueError('archive role mismatch')


def read(path, pin=None):
    path = Path(path)
    # Fail closed even if a published reference points at a prohibited artifact.
    if path.suffix != '.json' or any(x in path.name.lower() for x in ('test', 'checkpoint', 'embedding')):
        raise ValueError('only registered JSON metadata/validation archives permitted')
    if path.is_symlink() or path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError('symlink or oversized JSON')
    payload = path.read_bytes()
    sha = hashlib.sha256(payload).hexdigest()
    if pin is not None and sha != pin:
        raise ValueError('artifact SHA mismatch: ' + str(path))
    READS[str(path)] = sha
    return json.loads(payload), sha


def verified_records(records, manifest_sha, job_bound=False):
    for x in records:
        if not isinstance(x, dict) or x.get('status') != 'verified':
            continue
        pins = [x[k] for k in ('manifest_sha256', 'run_manifest_sha256') if k in x]
        if (pins or job_bound) and all(p == manifest_sha for p in pins):
            return True
    return False


def verifier(rid, manifest_sha):
    found = []
    for suffix in ('-verify', '-v1-verify'):
        v = ROOT / 'runs' / (rid + suffix)
        if not v.exists():
            continue
        e = {'run_id': v.name, 'terminal': None, 'records': [], 'evidence': []}
        t = v / 'terminal.json'
        if t.exists():
            term, _ = read(t)
            e['terminal'] = {k: term.get(k) for k in ('status', 'exit_code', 'wall_seconds', 'ended_at')}
        log = v / 'logs/stdout.log'
        if log.exists():
            if log.stat().st_size > 1024 * 1024:
                raise ValueError('oversized verifier stdout')
            payload = log.read_bytes()
            e['evidence'].append({'path': str(log), 'sha256': hashlib.sha256(payload).hexdigest()})
            for line in payload.decode().splitlines():
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        e['records'].append(obj)
                except ValueError:
                    pass
        for name in ('verify.json', 'manifest.json'):
            p = v / 'out' / name
            if p.exists():
                obj, sha = read(p)
                e['records'].append(obj)
                e['evidence'].append({'path': str(p), 'sha256': sha})
        jobs = sorted((ROOT / 'queue/claimed').glob('*-' + v.name + '.json'))
        job_bound = False
        if len(jobs) == 1:
            job, jsha = read(jobs[0])
            argv = job.get('argv', [])
            if '--manifest' in argv and '--manifest-sha' in argv:
                target = argv[argv.index('--manifest') + 1]
                pin = argv[argv.index('--manifest-sha') + 1]
                job_bound = job.get('run_id') == v.name and target == str(ROOT / 'runs' / rid / 'out/manifest.json') and pin == manifest_sha
            e['job_receipt'] = {'path': str(jobs[0]), 'sha256': jsha, 'target_binding_matches': job_bound}
        e['verified'] = job_bound and (e['terminal'] or {}).get('status') == 'succeeded' and (e['terminal'] or {}).get('exit_code') == 0 and verified_records(e['records'], manifest_sha, job_bound)
        # Emit verifier verdicts, not possibly large nested replay metadata.
        e['records'] = [{k: x.get(k) for k in ('status', 'manifest_sha256', 'run_manifest_sha256', 'checks', 'replayed_epochs') if k in x} for x in e['records'] if isinstance(x, dict)]
        found.append(e)
    return {'verified': any(x['verified'] for x in found), 'jobs': found}


def queue(rid):
    out = {}
    for name in ('pending', 'claimed', 'rejected'):
        fs = sorted((ROOT / 'queue' / name).glob('*-' + rid + '.json'))
        out[name] = [p.name for p in fs]
        if name == 'rejected':
            out['reasons'] = {}
            for p in fs:
                for q in (p.with_suffix('.reason'), Path(str(p) + '.reason')):
                    if q.exists(): out['reasons'][q.name] = q.read_text()[:2000]
    return out


def inspect_run(rid):
    rd = ROOT / 'runs' / rid
    result = {'run_id': rid, 'queue': queue(rid), 'terminal': None, 'verified': False}
    t = rd / 'terminal.json'
    if t.exists():
        obj, _ = read(t)
        result['terminal'] = {k: obj.get(k) for k in ('status', 'exit_code', 'wall_seconds', 'ended_at')}
    mp = rd / 'out/manifest.json'
    if mp.exists() and (result['terminal'] or {}).get('status') == 'succeeded':
        m, sha = read(mp)
        result.update(manifest_path=str(mp), manifest_sha256=sha, status=m.get('status'), selection=m.get('selection'), hit_max_epochs=m.get('hit_max_epochs'))
        result['verify'] = verifier(rid, sha)
        result['verified'] = result['verify']['verified']
        result['binding_keys'] = sorted(m.get('binding', {}))
    return result


def classify(deltas, feature=False):
    if len(deltas) != 3:
        return {'classification': 'UNKNOWN', 'temperature_adopt_condition': False}
    mean = sum(deltas) / 3
    count = sum(x >= 6 for x in deltas)
    flag = 'effective' if mean >= 6 and count >= 2 else 'harmful' if mean <= -6 else 'inconclusive'
    if feature and mean >= 6 and count < 2:
        flag = 'UNCLASSIFIED_PENDING_KIRO'
    return {'deltas': deltas, 'mean_delta_hits': mean, 'seeds_ge6': count,
            'classification': flag,
            'temperature_adopt_condition': all(x >= 6 for x in deltas)}


def mcnemar(b, c):
    n = b + c
    return min(1.0, float(2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / (2 ** n))) if n else 1.0


def check_g_rows(rows, queries):
    if len(rows) != len(queries): raise ValueError('query coverage mismatch')
    for i, (row, q) in enumerate(zip(rows, queries)):
        if row['ordinal'] != i or row['query_sha256'] != digest(q):
            raise ValueError('full ordered query identity mismatch')


def overlap(p, g, oov):
    if len(p) != len(g) or len(p) != len(oov): raise ValueError('hit-vector shape mismatch')
    if any((a or b) and o for a, b, o in zip(p, g, oov)): raise ValueError('OOV structural zero violated')
    out = {}
    for label, idx in [('all', list(range(len(p)))), ('oov', [i for i, x in enumerate(oov) if x]), ('non_oov', [i for i, x in enumerate(oov) if not x])]:
        both = sum(bool(p[i] and g[i]) for i in idx)
        hp, hg = sum(p[i] for i in idx), sum(g[i] for i in idx)
        union = hp + hg - both
        out[label] = {'queries': len(idx), 'p_hits': hp, 'g_hits': hg, 'both': both, 'only_p': hp-both, 'only_g': hg-both,
                      'union': union, 'neither': len(idx)-union, 'jaccard': both/union if union else None,
                      'oracle_recall': union/len(idx) if idx else None}
    return out


def load_archive(rid, queries, hist, state):
    if not state['verified'] or (state['terminal'] or {}).get('status') != 'succeeded':
        raise ValueError('train/verifier not accepted')
    mp = ROOT / 'runs' / rid / 'out/manifest.json'
    m, _ = read(mp, state['manifest_sha256'])
    be = m['selection']['best_epoch']
    bm = m['selection']['best_metric']
    if m['status'] != 'complete' or m['best']['epoch'] != be or list(m['best']['metric']) != bm:
        raise ValueError('best selection mismatch')
    binding = m['binding']
    if 'evaluations' in m:  # P list of ordered top-50 item IDs.
        # P's queries_sha256 also hashes feature-side fields and uses no newline.
        # Its original query-file pin plus the shared immutable v2 history pin
        # bind the complete ordered core queries, without dropping to target-only.
        if binding['inputs']['history_manifest_sha256'] != HISTORY_PIN or binding['inputs']['query_sha'] != hist['binding']['query_sha']:
            raise ValueError('P query/history binding mismatch')
        if binding['catalog_sha256'] != hist['catalog_sha256'] or binding['inputs']['catalog_sha'] != hist['binding']['catalog_sha']:
            raise ValueError('P catalog binding mismatch')
        recs = [x for x in m['evaluations'] if x['epoch'] == be]
        if len(recs) != 1: raise ValueError('P best evaluation cardinality')
        rec = recs[0]
        archive_role(rec['predictions_ref'], f'predictions-epoch-{be:03d}.json')
        p = (mp.parent / rec['predictions_ref']).resolve()
        if not p.is_relative_to(mp.parent.resolve()): raise ValueError('P archive escaped run')
        predictions, _ = read(p, rec['predictions_sha256'])
        if len(predictions) != len(queries) or any(len(x) != 50 for x in predictions): raise ValueError('P predictions shape')
        schema = 'P predictions-epoch-NNN.json'
    else:  # G evaluation manifest and pinned ordered shards.
        epochs = [x for x in m['epochs'] if x['epoch'] == be]
        if len(epochs) != 1: raise ValueError('G best epoch cardinality')
        entry = epochs[0]['evaluation']
        archive_role(entry['path'], f'evaluation-{be:03d}')
        p = (mp.parent / entry['path'] / 'manifest.json').resolve()
        if not p.is_relative_to(mp.parent.resolve()): raise ValueError('G evaluation escaped run')
        ev, _ = read(p, entry['manifest_sha256'])
        ctx = ev['binding']['context']
        if ctx['queries_sha256'] != digest(queries) or ctx['b1']['history_manifest_sha256'] != HISTORY_PIN:
            raise ValueError('G query/history binding mismatch')
        if ctx['catalog_sha256'] != hist['_g_catalog_digest'] or ctx['b1']['base_queries_sha256'] != hist['_base_queries_canonical_digest']:
            raise ValueError('G catalog binding mismatch')
        if not ev['complete'] or ev['completed_queries'] != len(queries): raise ValueError('G incomplete evaluation')
        rows = []
        for part in ev['shards']:
            sp = Path(part['path']).resolve()
            if not sp.is_relative_to(p.parent): raise ValueError('G shard escaped evaluation')
            archive_role(sp.name, f"shard-{part['start']:06d}-{part['end']:06d}.json")
            if sp.parent != p.parent: raise ValueError('G shard directory mismatch')
            obj, _ = read(sp, part['sha256'])
            if part['start'] != len(rows) or obj['start'] != part['start'] or obj['end'] != part['end'] or len(obj['rows']) != part['end'] - part['start']:
                raise ValueError('G shard gap/overlap')
            if obj['binding'] != ev['binding']: raise ValueError('G shard binding mismatch')
            rows.extend(obj['rows'])
        check_g_rows(rows, queries)
        if digest({'rows': [{k: v for k, v in row.items() if k != 'latency_seconds'} for row in rows], 'aggregate': ev['aggregate']}) != ev['computation_sha256']:
            raise ValueError('G computation hash mismatch')
        predictions = [row['items'][:50] for row in rows]
        schema = 'G evaluation-NNN/manifest.json and shards'
        if any(len(row['items']) < 50 or any(row['audit'].get(k) != 0 for k in ('invalid', 'deadend', 'collision')) for row in rows):
            raise ValueError('G candidate/audit mismatch')
    hits, ndcg = [], []
    for pred, q in zip(predictions, queries):
        if len(pred) != len(set(pred)): raise ValueError('duplicate predicted item')
        found = q['target'] in pred
        hits.append(int(found))
        ndcg.append(1/math.log2(pred.index(q['target'])+2) if found else 0)
    if any(h and q['oov'] for h, q in zip(hits, queries)): raise ValueError('OOV hit')
    if abs(sum(hits)/N-bm[0]) > 1e-9 or abs(sum(ndcg)/N-bm[1]) > 1e-9:
        raise ValueError('recomputed item metrics mismatch')
    return {'hits': hits, 'ndcg': ndcg, 'summary': {'hits': sum(hits), 'recall': bm[0], 'ndcg': bm[1], 'best_epoch': be, 'stop_epoch': m['selection']['epoch'], 'format': schema, 'archive_selfcheck': 'PASS', 'query_sha256': digest(queries)}}


def paired(new, ctl):
    import numpy as np
    hn, hc = np.array(new['hits']), np.array(ctl['hits'])
    d = hn-hc
    dn = np.array(new['ndcg'])-np.array(ctl['ndcg'])
    b, c = int(((hn == 1) & (hc == 0)).sum()), int(((hn == 0) & (hc == 1)).sum())
    rng = np.random.default_rng(0)
    mr, mn = [], []
    for _ in range(10):
        indices = rng.integers(0, len(hn), size=(1000, len(hn)))
        mr.extend(d[indices].mean(axis=1).tolist()); mn.extend(dn[indices].mean(axis=1).tolist())
    return {'new_hits': int(hn.sum()), 'control_hits': int(hc.sum()), 'delta_hits': int(d.sum()), 'only_new': b, 'only_control': c,
            'both': int(((hn == 1) & (hc == 1)).sum()), 'mcnemar_exact_two_sided': mcnemar(b, c),
            'delta_recall': float(d.mean()), 'recall_ci95': np.percentile(mr, [2.5, 97.5]).tolist(),
            'delta_ndcg': float(dn.mean()), 'ndcg_ci95': np.percentile(mn, [2.5, 97.5]).tolist(), 'bootstrap': {'replicates': 10000, 'seed': 0, 'unit': 'query'}}


def run_ids():
    return [f'{stem}-s{s}' for stem in ('p-php01-h1-t010', 'p-php02-t005', 'p-php02-t007', 'p-feat04a-a-q2t010', 'p-feat04a-b-f6hist', 'p-feat04b-c-i4-popfresh', 'p-feat04b-d-i5-quality', 'g-gtok01b-t3-b256d3', 'g-tok-t1r-rqvae-b256', 'g-tok-t3-bhkm-b256', 'g-garch01-a1-lr3e-4') for s in SEEDS]


def inspect_fits():
    out = {}
    for variant in ('b1024d3', 'b512d3', 'b128d3', 'b64d3', 'b32d3', 'b256d2'):
        rid = 'g-tok2-' + variant + '-fit'
        mp = ROOT / 'runs' / rid / 'out/run_manifest.json'
        m, pin = read(mp)
        audit, audit_sha = read(mp.parent/'audit.json', m['artifacts']['audit.json'])
        vr = ROOT / 'runs' / (rid+'-verify')
        proof, proof_sha = read(vr/'out/verify.json')
        t, _ = read(vr/'terminal.json')
        if proof.get('status') != 'verified' or proof.get('manifest_sha256') != pin or t.get('status') != 'succeeded':
            raise ValueError('fit verifier not accepted')
        out[rid] = {'manifest_sha256': pin, 'audit_sha256': audit_sha, 'audit': audit,
                    'verify_path': str(vr/'out/verify.json'), 'verify_sha256': proof_sha, 'verify_status': proof['status'],
                    'engineering': m['engineering'], 'gates': m['gates']}
    return out


def main():
    out = {'read_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'test_read': False, 'checkpoint_read': False, 'new_submit': False}
    states = {rid: inspect_run(rid) for rid in run_ids()}
    out['runs'] = states
    out['fits'] = inspect_fits()
    out['kiro_tok2_queue'] = {f'g-tok2-{v}-s42': queue(f'g-tok2-{v}-s42') for v in ('b1024d3', 'b512d3', 'b128d3', 'b64d3', 'b32d3', 'b256d2')}
    if '--metadata' not in sys.argv:
        hdir = ROOT / 'runs/h2-prepare/out'
        hist, _ = read(hdir/'manifest.json', HISTORY_PIN)
        queries, _ = read(hdir/'validation-l50.json', hist['artifacts']['validation-l50.json'])
        catalog, catalog_file_sha = read(ROOT.parent / 'vkgr-p4b-20261007t-pipeline-03/prepared-v3/catalog.json')
        if not isinstance(catalog, list) or len(catalog) != 120415 or len(set(catalog)) != len(catalog) or plain_digest(catalog) != hist['catalog_sha256'] or catalog_file_sha != hist['binding']['catalog_sha']:
            raise ValueError('frozen catalog identity mismatch')
        # The original query file pin is already part of the immutable history
        # binding. G stores the canonical-with-newline digest of the same list.
        # Its expected base digest equals the pinned original canonical JSON file.
        hist['_base_queries_canonical_digest'] = hist['binding']['query_sha']
        hist['_g_catalog_digest'] = digest(catalog)
        hist['_catalog_file_sha'] = catalog_file_sha
        if len(queries) != N or sum(q['oov'] for q in queries) != 623: raise ValueError('frozen population mismatch')
        out['population'] = {'n': N, 'oov': 623, 'non_oov': 1897, 'history_pin': HISTORY_PIN, 'queries_sha256': digest(queries), 'catalog_sha256': hist['catalog_sha256']}
        data = {}
        out['archive_summaries'] = {}
        for rid, state in states.items():
            try:
                data[rid] = load_archive(rid, queries, hist, state)
                out['archive_summaries'][rid] = data[rid]['summary']
            except (ValueError, KeyError, OSError) as e:
                out['archive_summaries'][rid] = {'status': 'UNKNOWN', 'reason': str(e)}
        out['paired'] = {}; out['mechanical'] = {}; out['g5_precheck'] = {}
        for stem in ('p-php02-t005', 'p-php02-t007', 'p-feat04a-a-q2t010', 'p-feat04a-b-f6hist', 'p-feat04b-c-i4-popfresh', 'p-feat04b-d-i5-quality'):
            pairs = {}
            for s in SEEDS:
                new, ctl = f'{stem}-s{s}', f'p-php01-h1-t010-s{s}'
                if new in data and ctl in data: pairs[str(s)] = paired(data[new], data[ctl])
            out['paired'][stem] = pairs
            out['mechanical'][stem] = classify([pairs[str(s)]['delta_hits'] for s in SEEDS if str(s) in pairs], feature=stem.startswith('p-feat04'))
        for stem in ('g-tok-t3-bhkm-b256', 'g-garch01-a1-lr3e-4'):
            out['g5_precheck'][stem] = {}
            for s in SEEDS:
                p, g = f'p-php01-h1-t010-s{s}', f'{stem}-s{s}'
                if p in data and g in data:
                    out['g5_precheck'][stem][str(s)] = overlap(data[p]['hits'], data[g]['hits'], [q['oov'] for q in queries])
        out['oracle_boundary'] = 'precheck only: union is an oracle upper bound, not achievable recall or a fusion result'
    out['read_artifacts_sha256'] = READS
    out['finished_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    print(json.dumps(out, ensure_ascii=False, sort_keys=True, allow_nan=False))


if __name__ == '__main__': main()
