"""FEAT-04b runner tests: config registry, the equivalences E1/E2/E3 (+ E4 in its own class), temperature plumbing, candidate-side seam,
leakage L1-L7, train + verify round trips, CLI loop on CPU, preflight, real-path wiring.

``guard_*`` functions are plain assertion functions (raise AssertionError, survive ``python -O``); tests/test_feat04b_mutations.py
re-runs them under deliberately wrong implementations.

Equivalence comparison (E1-E4) reads the manifest.json of both runs and compares every top-level key except EXACTLY these:
``schema``, ``kind``, ``binding``, ``training_seconds``, ``evaluation_seconds`` and, inside ``config``, ``schema`` and ``task_id``
(the design's exclusion list). Two consequences are not extensions of that list but follow from it:
(a) a checkpoint file embeds the digest of the excluded ``binding``, so ``best.sha256`` / ``last.sha256`` cannot be compared and the model state
dicts are compared byte for byte instead; (b) php01's config has the knob/dim schema, so ``config`` is compared on the training hyper-parameters of
``b1.default_config()`` (and, for E2/E3/E4, also on ``feature`` and ``side_version``).
"""
import contextlib
import copy
import inspect
import io
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_feat02_model as tm  # noqa: E402
import test_feat03_model as t3  # noqa: E402
import test_feat04b_model as tmb  # noqa: E402
from test_feat02_run import flip_non_history_fields  # noqa: E402
from test_feat03_run import craft_row  # noqa: E402
from test_feat04b_untouched import check_pinned, load_pins  # noqa: E402

ensure = tm.ensure
CONTROL, A, E = 'f4_control', 'f4_recency_pool', 'f4_recency_behavior'
C, D = 'f4_item_popularity', 'f4_item_quality'
CD = (C, D)
REAL_FEATURE = 'f4_item_popularity'                    # a real variant of this runner (implemented in a later feature; config-level checks only here)
ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_TOP = ('schema', 'kind', 'binding', 'training_seconds', 'evaluation_seconds')
EXCLUDED_CONFIG = ('schema', 'task_id')
CUDA_MESSAGE = 'ERROR P-FEAT04B gate failed: ValueError: actual CUDA environment required'


def r4():
    from vkgr import retrieval_feat04b_run
    return retrieval_feat04b_run


def fr2():
    from vkgr import retrieval_feat02_run
    return retrieval_feat02_run


def syn():
    """The synthetic world of every variant: FEAT-02 synthetic data plus seeded synthetic ITEM_STATS_V1 (a superset: A/E/control ignore the extra keys)."""
    return r4().synthetic_data4b()


def config4(**overrides):
    value = r4().default_config(); value.update(logq=True, temperature=0.1, batch_size=16, max_epochs=8, min_epochs=2, feature=A); value.update(overrides)
    return value


def real_config(**overrides):
    value = r4().default_config(); value.update(logq=True, temperature=0.1, feature=REAL_FEATURE); value.update(overrides)
    return value


def model4(data, feature, logits=False):
    """Runner-built model on CPU in eval mode; A and E optionally get fixed non-zero logits (E also a non-zero beta) so the weighted branch is exercised."""
    model = r4()._model4b(data, config4(feature=feature), 'cpu'); model.eval()
    if feature in (A, E) and logits:
        t3.set_logits(model, seed=5, scale=0.5)
    if feature == E and logits:
        tmb.set_beta(model, seed=6, scale=0.6)
    return model


def width_of(batch):
    return max(len(r['history']) for r in batch)


def rejects(fn):
    try:
        fn()
    except ValueError:
        return True
    return False


def strip_timing(value):
    return {k: v for k, v in value.items() if k not in ('training_seconds', 'evaluation_seconds')}


# ---------------------------------------------------------------- shared guards (reused by the mutation tests)

def guard_s_config_registry():
    """Real path: T=0.1 (float), seed in {42,2024,2025}, real features only (never f4_control / f4_recency_pool). Synthetic: T in {0.1, 0.15}, control and A allowed."""
    m = r4()
    for seed in (42, 2024, 2025):
        m.validate_config(real_config(seed=seed))
    for bad in ({'temperature': 0.15}, {'temperature': 0.07}, {'temperature': 1}, {'feature': CONTROL}, {'feature': A}, {'learning_rate': 0.0005}, {'learning_rate': 0.0003},
                {'seed': 7}, {'logq': False}, {'extra_uniform': 128}, {'init': 'legacy'}, {'history_len': 20}, {'max_epochs': 30}, {'schema': 'p-feat04a-run-v1'},
                {'task_id': 'FEAT-04A'}, {'unknown': 1}):
        ensure(rejects(lambda bad=bad: m.validate_config(real_config(**bad))), f'real config {bad} must be rejected')
    ensure(rejects(lambda: m.validate_config({k: v for k, v in real_config().items() if k != 'feature'})), 'missing field must be rejected')
    for temperature in (0.1, 0.15):
        for feature in (CONTROL, A):
            m.validate_config(config4(feature=feature, temperature=temperature), synthetic=True)
    ensure(rejects(lambda: m.validate_config(config4(temperature=0.2), synthetic=True)), 'synthetic T=0.2 must be rejected')
    ensure(rejects(lambda: m.validate_config(config4(feature='nope'), synthetic=True)), 'unknown feature must be rejected')


def guard_s_lr_fixed():
    """Real path: learning_rate is float 0.001 only (b1 allows 0.0003 too); the synthetic path keeps both b1 values for the equivalence fixtures."""
    m = r4()
    m.validate_config(real_config(learning_rate=0.001))
    for bad in (0.0003, 0.0005, 0.01, 1):
        ensure(rejects(lambda bad=bad: m.validate_config(real_config(learning_rate=bad))), f'real learning_rate {bad!r} must be rejected')
    m.validate_config(config4(learning_rate=0.0003), synthetic=True)


def guard_s_cli_default_device_cpu():
    parser = r4().build_parser()
    for argv in (['synthetic-train', '--config', 'c.json', '--output', 'o'],
                 ['synthetic-verify', '--config', 'c.json', '--manifest', 'm.json', '--manifest-sha', 'a' * 64]):
        ensure(parser.parse_args(argv).device == 'cpu', f'{argv[0]}: --device must default to cpu')


def _first_batch(config):
    from vkgr.retrieval_negatives import target_frequency
    from vkgr.retrieval_training import EpochCursor
    data = syn(); rows, _ = fr2()._view(data, 50)
    indices = EpochCursor(len(rows), config['seed']).next(config['batch_size'])
    return data, rows, [rows[i] for i in indices], target_frequency(rows)


def _independent_loss(config, temperature):
    from vkgr import retrieval_b1_run as b1
    from vkgr import retrieval_feat04b as m4
    from vkgr.retrieval_negatives import masked_ce_logq
    data, rows, batch, frequency = _first_batch(config)
    model = m4.make_feat04b_model(config['feature'], data['content'], data['supported_indices'], 50, config['seed']); model.train()
    columns, labels, mask, negatives, pool_size = b1._columns(batch, data, config, random.Random(0), None)
    histories, side = fr2()._tensors2(batch, data['idmap'], max(len(r['history']) for r in batch), 'f3_f4_recency')
    log_q = b1._log_q(batch, negatives, pool_size, config, frequency)
    loss, _ = masked_ce_logq(model.users(histories, side), model.items(columns), labels, mask, temperature, log_q)
    return float(loss.detach())


def guard_s_temperature_plumbing():
    """The first-step loss uses the temperature of the config (0.1 and 0.15), not a constant."""
    from vkgr.retrieval_training import SamplerStats
    for feature in (CONTROL, A):
        for temperature in (0.1, 0.15):
            config = config4(feature=feature, temperature=temperature); data, rows, batch, frequency = _first_batch(config)
            model = r4()._model4b(data, config, 'cpu')
            optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
            record = r4().train_step(model, optimizer, batch, {**data, 'rows': rows}, SamplerStats(), 0, config, random.Random(0), None, frequency)
            expected, other = _independent_loss(config, temperature), _independent_loss(config, 0.25 - temperature)
            ensure(abs(expected - other) > 1e-4, 'fixture cannot tell the temperatures apart')
            ensure(abs(record['loss'] - expected) < 1e-6, f'{feature}: first-step loss {record["loss"]} is not the T={temperature} loss {expected}')


def _stripped_manifest(path, config_keys):
    value = json.loads(Path(path).read_text())
    value = {k: v for k, v in value.items() if k not in EXCLUDED_TOP}
    value['config'] = {k: v for k, v in value['config'].items() if k in config_keys and k not in EXCLUDED_CONFIG}
    for name in ('best', 'last'):
        value[name] = {k: v for k, v in value[name].items() if k != 'sha256'}           # the checkpoint file embeds the excluded binding; state dicts are compared below
    return value


def compare_runs(dir_a, dir_b, config_keys):
    """Every top-level manifest key except the excluded ones is equal; both key sets are equal; state dicts equal byte for byte."""
    a, b = _stripped_manifest(Path(dir_a) / 'manifest.json', config_keys), _stripped_manifest(Path(dir_b) / 'manifest.json', config_keys)
    ensure(set(a) == set(b), f'manifest keys differ: {sorted(set(a) ^ set(b))}')
    ensure(len(a['records']) > 8, 'fixture too short')
    for key in sorted(a):
        if key in ('records', 'evaluations'):
            continue
        ensure(a[key] == b[key], f'manifest.{key} differs')
    for key in ('loss', 'gradient_norm', 'drawn', 'skipped', 'indices_sha256', 'negatives_sha256', 'step', 'epoch'):
        ensure([r[key] for r in a['records']] == [r[key] for r in b['records']], f'records.{key} differ')
    ensure(a['records'] == b['records'], 'records differ')
    ensure(a['evaluations'] == b['evaluations'], 'evaluations (result, predictions_sha256, diagnostics) differ')
    for name in ('best.pt', 'last.pt'):
        sa = torch.load(Path(dir_a) / name, map_location='cpu', weights_only=True)['model']
        sb = torch.load(Path(dir_b) / name, map_location='cpu', weights_only=True)['model']
        ensure(list(sa) == list(sb), f'{name}: state keys differ')
        for key in sa:
            ensure(sa[key].numpy().tobytes() == sb[key].numpy().tobytes(), f'{name}: {key} differs')


def _training_keys():
    from vkgr import retrieval_b1_run as b1
    return set(b1.default_config())


def guard_s_equiv_h1():
    """E1: f4_control at T0.10 == php01 H1 (T0.10 + no new feature), every epoch."""
    from vkgr import retrieval_php01_run as php
    data = fr2().synthetic_data()
    php_config = {**php.default_config(), 'knob': 'H1', 'temperature': 0.1, 'dim': 64, 'batch_size': 16, 'max_epochs': 8, 'min_epochs': 2}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        r4().run_session(tmp / 'a', data, config4(feature=CONTROL, temperature=0.1)); php.run_session(tmp / 'b', data, php_config)
        compare_runs(tmp / 'a', tmp / 'b', _training_keys())


def guard_s_equiv_feat03_q2():
    """E2: chain A at T0.15 == feat03 Q2 (logit trajectories included in the per-epoch diagnostics)."""
    from vkgr import retrieval_feat03_run as f3r
    data = fr2().synthetic_data()
    f3_config = {**f3r.default_config(), 'feature': A, 'logq': True, 'batch_size': 16, 'max_epochs': 8, 'min_epochs': 2}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        value = r4().run_session(tmp / 'a', data, config4(feature=A, temperature=0.15)); f3r.run_session(tmp / 'b', data, f3_config)
        ensure('recency_logits' in value['evaluations'][0]['diagnostics'], 'A must record the logit trajectory')
        compare_runs(tmp / 'a', tmp / 'b', _training_keys() | {'feature', 'side_version'})


def guard_s_equiv_f4_control_t015():
    """E3: f4_control at T0.15 == feat03 f4_control (catches temperature wiring errors)."""
    from vkgr import retrieval_feat03_run as f3r
    data = fr2().synthetic_data()
    f3_config = {**f3r.default_config(), 'feature': CONTROL, 'logq': True, 'batch_size': 16, 'max_epochs': 8, 'min_epochs': 2}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        r4().run_session(tmp / 'a', data, config4(feature=CONTROL, temperature=0.15)); f3r.run_session(tmp / 'b', data, f3_config)
        compare_runs(tmp / 'a', tmp / 'b', _training_keys() | {'feature', 'side_version'})


def guard_s_first_step_loss_equals_anchor():
    """A starts exactly at the anchor and E exactly at A: the loss of the first step is bit-equal to f4_control's (same seed, same batch)."""
    from vkgr.retrieval_training import SamplerStats
    losses = {}
    for feature in (CONTROL, A, E):
        config = config4(feature=feature); data, rows, batch, frequency = _first_batch(config)
        model = r4()._model4b(data, config, 'cpu'); optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
        losses[feature] = r4().train_step(model, optimizer, batch, {**data, 'rows': rows}, SamplerStats(), 0, config, random.Random(0), None, frequency)['loss']
    ensure(losses[CONTROL] == losses[A] == losses[E], f'first-step loss differs: {losses}')


def _bridge_runner():
    """The FEAT-04a runner (merged provisionally from feat04a/p); unavailable -> SkipTest, and the verification note records whether E4 really ran."""
    try:
        from vkgr import retrieval_feat04a_run
    except ImportError:
        raise unittest.SkipTest('vkgr.retrieval_feat04a_run unavailable (E4 did not run)')
    return retrieval_feat04a_run


def guard_s_bridge_e4(temperature=0.1):
    """E4: 04b chain A at T0.10 vs the FEAT-04a A, both produced by the synthetic-train CLI of their own runner: records, evaluations (incl. diagnostics.recency_logits),
    selection and every other manifest key outside the exclusion list are equal, and so are the model state dicts byte for byte."""
    runner_a = _bridge_runner()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp); config = config4(feature=A, temperature=0.1)
        (tmp / 'b.json').write_text(json.dumps(config))
        (tmp / 'a.json').write_text(json.dumps({**config, 'schema': runner_a.SCHEMA, 'task_id': runner_a.TASK, 'temperature': temperature}))
        with contextlib.redirect_stdout(io.StringIO()):
            ensure(r4().main(['synthetic-train', '--config', str(tmp / 'b.json'), '--output', str(tmp / 'out-b')]) == 0, '04b synthetic-train failed')
            ensure(runner_a.main(['synthetic-train', '--config', str(tmp / 'a.json'), '--output', str(tmp / 'out-a')]) == 0, '04a synthetic-train failed')
        ensure('recency_logits' in json.loads((tmp / 'out-b' / 'manifest.json').read_text())['evaluations'][0]['diagnostics'], 'A must record the logit trajectory')
        compare_runs(tmp / 'out-b', tmp / 'out-a', set(config))


def guard_s_untouched():
    check_pinned(ROOT, load_pins())


def guard_s_candidate_seam_wiring():
    """train_step's candidate column and the evaluation export's catalog vectors are read through ``model.candidate_items`` (C/D seam), not ``items``."""
    from vkgr import retrieval_feat04b as m4
    from vkgr.retrieval_training import SamplerStats
    config = config4(feature=A); data, rows, batch, frequency = _first_batch(config)
    model = r4()._model4b(data, config, 'cpu'); optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    original = m4.Feat04bIDMean.candidate_items; calls = []

    def spy(self, indices):
        calls.append(tuple(indices.shape))
        return original(self, indices)
    with mock.patch.object(m4.Feat04bIDMean, 'candidate_items', spy):
        r4().train_step(model, optimizer, batch, {**data, 'rows': rows}, SamplerStats(), 0, config, random.Random(0), None, frequency)
        ensure(len(calls) == 1, 'train_step must read the candidate column through candidate_items exactly once')
        calls.clear()
        queries = fr2()._view(data, 50)[1][:2]
        r4().export_vectors_side4b(model, data['catalog'], queries, data['idmap'], 16, 50, A)
        ensure(len(calls) == -(-len(data['catalog']) // 16), 'the evaluation export must read every catalog chunk through candidate_items')


# ---------------------------------------------------------------- C/D guards (guard_cd_*, L3 / L5 / L9 for the candidate side)

def _cd_loss_through(variant, method):
    """First-step loss of the batch computed independently, with the candidate column read through ``model.<method>`` (candidate_items or items)."""
    from vkgr import retrieval_b1_run as b1
    from vkgr.retrieval_negatives import masked_ce_logq
    config = config4(feature=variant); data, rows, batch, frequency = _first_batch(config)
    model = r4()._model4b(data, config, 'cpu'); model.train()
    columns, labels, mask, negatives, pool_size = b1._columns(batch, data, config, random.Random(0), None)
    histories, side = fr2()._tensors2(batch, data['idmap'], max(len(r['history']) for r in batch), 'f3_f4_recency')
    log_q = b1._log_q(batch, negatives, pool_size, config, frequency)
    loss, _ = masked_ce_logq(model.users(histories, side), getattr(model, method)(columns), labels, mask, config['temperature'], log_q)
    return float(loss.detach())


def guard_cd_train_step_uses_candidate_items():
    """C/D: train_step reads the candidate column through candidate_items (spy) and its first-step loss is the independent candidate_items loss, not the items loss."""
    from vkgr import retrieval_feat04b as m4
    from vkgr.retrieval_training import SamplerStats
    for variant in CD:
        config = config4(feature=variant); data, rows, batch, frequency = _first_batch(config)
        model = r4()._model4b(data, config, 'cpu'); optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
        original = m4.Feat04bIDMean.candidate_items; calls = []

        def spy(self, indices):
            calls.append(tuple(indices.shape)); return original(self, indices)
        with mock.patch.object(m4.Feat04bIDMean, 'candidate_items', spy):
            record = r4().train_step(model, optimizer, batch, {**data, 'rows': rows}, SamplerStats(), 0, config, random.Random(0), None, frequency)
        ensure(len(calls) == 1, f'{variant}: train_step must read the candidate column through candidate_items exactly once')
        through_candidates, through_items = _cd_loss_through(variant, 'candidate_items'), _cd_loss_through(variant, 'items')
        ensure(abs(through_candidates - through_items) > 1e-5, f'{variant}: fixture cannot tell candidate_items from items')
        ensure(abs(record['loss'] - through_candidates) < 1e-6, f'{variant}: train_step loss {record["loss"]} is not the candidate_items loss {through_candidates}')


def guard_cd_train_step_tables_receive_gradient():
    """C/D: after one real train_step the variant's two bucket tables hold finite, non-zero gradients (the loss reaches them through the candidate column) and the PAD rows exactly 0."""
    from vkgr import retrieval_feat04b as m4
    from vkgr.retrieval_training import SamplerStats
    for variant in CD:
        config = config4(feature=variant); data, rows, batch, frequency = _first_batch(config)
        model = r4()._model4b(data, config, 'cpu'); optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
        r4().train_step(model, optimizer, batch, {**data, 'rows': rows}, SamplerStats(), 0, config, random.Random(0), None, frequency)
        for table, _, _ in m4.STAT_LAYOUT[variant]:
            grad = getattr(model, table).weight.grad
            ensure(grad is not None and bool(torch.isfinite(grad).all()) and float(grad.abs().sum()) > 0, f'{variant}: {table} got no gradient from train_step')
            ensure(bool((grad[0] == 0).all()), f'{variant}: PAD row of {table} must get exactly 0')


def guard_cd_export_uses_candidate_items():
    """C/D: the evaluation export reads the catalog through candidate_items: spy count per chunk, and the raw norms returned are the norms of candidate_items."""
    from vkgr import retrieval_feat04b as m4
    for variant in CD:
        data = syn(); model = model4(data, variant); queries = fr2()._view(data, 50)[1][:4]
        original = m4.Feat04bIDMean.candidate_items; calls = []

        def spy(self, indices):
            calls.append(tuple(indices.shape)); return original(self, indices)
        with mock.patch.object(m4.Feat04bIDMean, 'candidate_items', spy):
            _, _, norms = r4().export_vectors_side4b(model, data['catalog'], queries, data['idmap'], 16, 50, variant)
        ensure(len(calls) == -(-len(data['catalog']) // 16), f'{variant}: the export must read every catalog chunk through candidate_items')
        ids = torch.tensor([data['idmap'][i] for i in data['catalog']])
        with torch.no_grad():
            expected = np.linalg.norm(model.candidate_items(ids).numpy().astype(np.float64), axis=1)
        ensure(np.array_equal(norms, expected), f'{variant}: exported item norms are not the candidate_items norms')


def guard_cd_binding_has_cutoff(variant=None):
    """C/D binding: stat_cutoff_week == 24, the sidecar digest and the digest of the variant's own bucket-id vectors; a different cutoff / bucket ids change it; the others carry none of these."""
    from vkgr import item_stats
    for feature in (variant,) if variant else CD:
        data = syn(); config = config4(feature=feature); rows, queries = fr2()._view(data, 50); model = r4()._model4b(data, config, 'cpu')
        binding = r4().binding_for4b(data, config, rows, queries, model, 'cpu', None)
        ensure('stat_cutoff_week' in binding and binding['stat_cutoff_week'] == item_stats.STAT_CUTOFF_WEEK == 24, f'{feature}: binding must carry stat_cutoff_week == 24')
        ensure(binding.get('stats_sha256') == data['stats_sha256'] and len(binding.get('stat_bucket_ids_sha256', '')) == 64, f'{feature}: binding must carry the sidecar and bucket-id digests')
        later = r4().binding_for4b({**data, 'stat_cutoff_week': 25}, config, rows, queries, model, 'cpu', None)
        ensure(later != binding, f'{feature}: a different cutoff week must change the binding')
    if variant is None:
        for feature in (CONTROL, A, E):
            data = syn(); config = config4(feature=feature); rows, queries = fr2()._view(data, 50); model = r4()._model4b(data, config, 'cpu')
            binding = r4().binding_for4b(data, config, rows, queries, model, 'cpu', None)
            ensure(not any(k in binding for k in ('stat_cutoff_week', 'stats_sha256', 'stat_bucket_ids_sha256')), f'{feature}: binding must not carry stat fields')


def guard_l3_candidate_reverse_control(variant):
    """L3 for C/D: changing one item's bucket id changes that item's candidate vector and nothing else (users are untouched); restoring it restores the vectors."""
    from vkgr import retrieval_feat04b as m4
    data = syn(); model = model4(data, variant); table, buffer, edges = m4.STAT_LAYOUT[variant][0]
    idx = data['idmap'][data['catalog'][3]]; ids = torch.tensor([data['idmap'][i] for i in data['catalog']])
    rows, _ = fr2()._view(data, 50); width = width_of(rows[:8]); histories, side = r4()._tensors4b(rows[:8], data['idmap'], width, variant)
    base, base_users = model.candidate_items(ids), model.users(histories, side)
    old = int(getattr(model, buffer)[idx])
    getattr(model, buffer)[idx] = old % (len(getattr(m4, edges)) + 1) + 1
    moved = model.candidate_items(ids)
    changed = (moved != base).any(-1).nonzero().flatten().tolist()
    ensure(changed == [data['catalog'].index(data['catalog'][3])], f'{variant}: only the edited item may change, got rows {changed}')
    ensure(torch.equal(model.users(histories, side), base_users), f'{variant}: the user vector must not depend on the candidate-side bucket ids')
    getattr(model, buffer)[idx] = old
    ensure(torch.equal(model.candidate_items(ids), base), f'{variant}: restoring the bucket id must restore the vectors')


def guard_l5_candidate_side_depends_on_item_id_only(variant):
    """L5 for C/D: the candidate vector of an item id is bit-equal in different batches, positions and with different companions, and equals the catalog vector."""
    from vkgr import retrieval_b1_run as b1
    data = syn(); model = model4(data, variant); rows, _ = fr2()._view(data, 50); config = config4(feature=variant)
    ids = torch.tensor([data['idmap'][i] for i in data['catalog']]); catalog_vectors = model.candidate_items(ids)
    seen = {}
    for batch in (rows[:20], list(reversed(rows[10:30])), rows[5:15] + rows[40:50]):
        columns = b1._columns(batch, data, config, random.Random(1), None)[0].flatten()
        vectors = model.candidate_items(columns)
        for position, index in enumerate(columns.tolist()):
            ensure(torch.equal(vectors[position], catalog_vectors[index - 1]), f'{variant}: candidate vector of index {index} differs from the catalog vector')
            if index in seen:
                ensure(torch.equal(seen[index], vectors[position]), f'{variant}: candidate vector of index {index} differs between batches')
            seen[index] = vectors[position]
    ensure(len(seen) >= 10, 'fixture: too few candidate columns')


def guard_l9_train_and_export_vectors_equal(variant):
    """L9 for C/D: the candidate vector train_step feeds the loss and the one the evaluation export writes for the same item are bit-equal (same candidate_items, same weights)."""
    from vkgr import retrieval_feat04b as m4
    from vkgr.retrieval_training import SamplerStats
    seen = []
    original = m4.Feat04bIDMean.candidate_items

    def spy(self, indices):
        out = original(self, indices); seen.append((indices.clone(), out.detach().clone())); return out
    config = config4(feature=variant); data, rows, batch, frequency = _first_batch(config)
    with mock.patch.object(m4.Feat04bIDMean, 'candidate_items', spy):
        trained = r4()._model4b(data, config, 'cpu'); optimizer = torch.optim.AdamW(trained.parameters(), lr=0.001, weight_decay=0.0001)
        r4().train_step(trained, optimizer, batch, {**data, 'rows': rows}, SamplerStats(), 0, config, random.Random(0), None, frequency)
        train_calls, seen[:] = list(seen), []
        exporter = r4()._model4b(data, config, 'cpu'); exporter.eval()
        r4().export_vectors_side4b(exporter, data['catalog'], fr2()._view(data, 50)[1][:2], data['idmap'], 64, 50, variant)
    ensure(len(train_calls) == 1 and len(seen) > 0, f'{variant}: both paths must go through candidate_items')
    by_index = {}
    for indices, vectors in seen:
        for k, index in enumerate(indices.tolist()):
            by_index[index] = vectors[k]
    columns, vectors = train_calls[0]
    ensure(all(torch.equal(by_index[index], vectors[k]) for k, index in enumerate(columns.tolist())), f'{variant}: training and export see different vectors for the same item')


# ---------------------------------------------------------------- leakage guards (L1-L7)

def guard_target_flip_leaves_feature_unchanged(variant):
    """L1: changing the target week/position (and every other non-history field) moves neither the batch tensors nor the user vectors."""
    data = syn(); rows, queries = fr2()._view(data, 50)
    model = model4(data, variant, logits=True)
    picks = (lambda r: r['history_positions'][-1][0], lambda r: 24, lambda r: 26, lambda r: 30)
    for batch in (rows[:24], queries[:24]):
        width = width_of(batch)
        base = r4()._tensors4b(batch, data['idmap'], width, variant); base_users = model.users(*base)
        for pick in picks:
            moved = []
            for r in batch:
                m = flip_non_history_fields(r); m['target_position'] = [pick(r), 5]
                moved.append(m)
            got = r4()._tensors4b(moved, data['idmap'], width, variant)
            ensure(torch.equal(got[0], base[0]) and torch.equal(got[1], base[1]), f'{variant}: batch tensors changed with the target week/position')
            ensure(torch.equal(model.users(*got), base_users), f'{variant}: user vectors changed with the target week/position')


def guard_l2_export_path_ignores_non_history_fields(variant):
    """L2: the evaluation export (user and item vectors after normalisation, raw item norms) is independent of the non-history fields."""
    data = syn(); rows, queries = fr2()._view(data, 50)
    model = model4(data, variant, logits=True)
    flipped = [flip_non_history_fields(q) for q in queries]
    a = r4().export_vectors_side4b(model, data['catalog'], queries, data['idmap'], 16, 50, variant)
    b = r4().export_vectors_side4b(model, data['catalog'], flipped, data['idmap'], 16, 50, variant)
    ensure(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) and np.array_equal(a[2], b[2]), f'{variant}: export changed with the non-history fields')


def guard_l4_reverse_controls(variant):
    """L4: changing a history event's own field changes the user vector; restoring it restores the vector."""
    data = syn(); model = model4(data, variant, logits=True)
    base_row = craft_row(data, [0, 4, 8, 12, 16]); width = 5

    def users(row):
        return model.users(*r4()._tensors4b([row], data['idmap'], width, variant))
    base = users(base_row)
    changes = [lambda r, slot=slot, new=new: {**r, 'history_positions': [[w if k != slot else new, o] for k, (w, o) in enumerate(r['history_positions'])]} for slot, new in ((1, 8), (3, 15), (4, 24))]
    changes.append(lambda r: {**r, 'history': [data['catalog'][40]] + r['history'][1:]})
    if variant == E:                                                    # E additionally reads each event's own feedback bits
        for slot, bit in ((0, 0), (2, 3), (4, 5)):
            changes.append(lambda r, slot=slot, bit=bit: {**r, 'side': [[m ^ (1 << bit) if k == slot else m, t, d] for k, (m, t, d) in enumerate(r['side'])]})
    for change in changes:
        ensure(not torch.equal(users(change(base_row)), base), f'{variant}: a history-event change did not move the user vector')
    ensure(torch.equal(users(copy.deepcopy(base_row)), base), f'{variant}: restoring the row did not restore the vector')


def guard_l5_week_shift_invariance(variant):
    """L5: adding a constant to every history week (staying <= 24) leaves the user vector bit-equal (no absolute calendar channel)."""
    data = syn(); rows, queries = fr2()._view(data, 50)
    shiftable = [r for r in rows + queries if r['history_positions'][-1][0] <= 23][:24]
    ensure(len(shiftable) >= 12, 'fixture: not enough shiftable rows')
    shifted = [{**r, 'history_positions': [[w + 1, o] for w, o in r['history_positions']]} for r in shiftable]
    model = model4(data, variant, logits=True); width = width_of(shiftable)
    a = r4()._tensors4b(shiftable, data['idmap'], width, variant); b = r4()._tensors4b(shifted, data['idmap'], width, variant)
    ensure(not torch.equal(a[1], b[1]), 'the week column did not move')
    ensure(torch.equal(model.users(*a), model.users(*b)), f'{variant}: user vectors depend on the absolute week')


def guard_l6_week_25_rejected(variant):
    data = syn(); model = model4(data, variant, logits=True)
    histories, side = r4()._tensors4b([craft_row(data, [0, 4, 8, 12, 25])], data['idmap'], 5, variant)
    ensure(rejects(lambda: model.users(histories, side)), f'{variant}: week 25 in a history must be rejected')


FORBIDDEN_TEXT = ('target_position', 'past_likes', "['target']", '["target"]', "['oov']", '["oov"]')
FORBIDDEN_DIAGNOSTICS = ('evaluate_model', 'support_slices', 'top50_unsupported_share', 'history_support_slices', 'side_visibility', 'data_diagnostics',
                         'item_norm_stats', 'top1_is_last_history_fraction', 'zero_user_vectors', 'beta_bit_train_counts', 'target_bucket_hist', 'stats_self_inclusion',
                         'logq_redundancy', 'top50_stat_profile', 'feature_data_diagnostics')


def feature_path_callables(variant=None):
    """Callables on the feature path, resolved at call time so patched versions are scanned too."""
    from vkgr import retrieval_feat03 as f3
    from vkgr import retrieval_feat04b as m4
    return [fr2()._tensors2, r4()._tensors4b, r4().train_step, r4().export_vectors_side4b, f3.event_vectors, m4.event_vectors4, m4.behavior_logits, m4.candidate_stat_terms,
            m4.stat_bucket_ids, m4.expo_bucket, m4.first_bucket, m4.ratio_bucket4, m4.like_bucket,
            m4.Feat04bIDMean.users, m4.Feat04bIDMean.history_items, m4.Feat04bIDMean.candidate_items]


def guard_leak_l7_source_scan(fn_or_source):
    """L7: no target/past_likes/oov access and no diagnostic helper in the source of a feature-assembly function (a function object or a source string)."""
    source = fn_or_source if isinstance(fn_or_source, str) else inspect.getsource(fn_or_source)
    body = '\n'.join(line.split('#')[0] for line in source.splitlines())          # comments are not code
    for text in FORBIDDEN_TEXT:
        ensure(text not in body.replace('history_positions', ''), f'forbidden access {text}')
    for name in FORBIDDEN_DIAGNOSTICS:
        ensure(name not in body, f'diagnostic function {name} on the feature path')


def guard_l7_source_has_no_target_access():
    """L7 over every callable of the feature-assembly path (resolved at call time, so patched versions are scanned too)."""
    for fn in feature_path_callables():
        try:
            guard_leak_l7_source_scan(fn)
        except AssertionError as error:
            raise AssertionError(f'{fn.__qualname__}: {error}') from error


# ---------------------------------------------------------------- CLI helpers

def run_cli(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = r4().main(argv)
    return code, out.getvalue(), err.getvalue()


GOOD_SPEC = {'schema': 'p-feat04b-inputs-v1', 'prepared': '/a', 'holdout': '/b', 'raw_root': '/c', 'history_manifest': '/d/manifest.json',
             'history_sha256': 'a' * 64, 'side_manifest': '/e/manifest.json', 'side_sha256': 'b' * 64, 'items_metadata_sha256': 'c' * 64,
             'ctx_manifest': None, 'ctx_sha256': None, 'attr_manifest': None, 'attr_sha256': None, 'stats_manifest': None, 'stats_sha256': None,
             'disk_cap_bytes': 1 << 30, 'test_gate': 'closed_hash_only'}
CD_SPEC = {**GOOD_SPEC, 'stats_manifest': '/s/manifest.json', 'stats_sha256': 'e' * 64}
BASE_SPEC = {k: v for k, v in GOOD_SPEC.items() if k not in ('stats_manifest', 'stats_sha256')} | {'schema': 'p-feat02-inputs-v1'}      # the spec of range-check / prepare-stats / verify-stats
RUNNABLE = (CONTROL, A, E, C, D)


class ConfigAndInputsTests(unittest.TestCase):
    def test_constants(self):
        m = r4()
        self.assertEqual((m.TASK, m.SCHEMA, m.MANIFEST_SCHEMA, m.CHECKPOINT_SCHEMA, m.KIND),
                         ('FEAT-04B', 'p-feat04b-run-v1', 'p-feat04b-train-manifest-v1', 'p-feat04b-checkpoint-v1', 'p_feat04b_train'))
        self.assertEqual(m.DEFAULT_DEVICE, 'cpu')

    def test_config_registry(self):
        guard_s_config_registry()

    def test_lr_fixed(self):
        guard_s_lr_fixed()

    def test_default_config_fields_equal_the_feat03_field_set(self):
        from vkgr import retrieval_feat03_run as f3r
        self.assertEqual(set(r4().default_config()), set(f3r.default_config()))

    def test_inputs_pin_rules(self):
        m = r4()
        m.validate_inputs4b(dict(GOOD_SPEC), A)
        m.validate_inputs4b(dict(GOOD_SPEC), CONTROL)
        for bad_pin in ({'ctx_manifest': '/f/manifest.json', 'ctx_sha256': 'd' * 64}, {'attr_manifest': '/f/manifest.json', 'attr_sha256': 'd' * 64}):
            with self.assertRaises(ValueError):
                m.validate_inputs4b({**GOOD_SPEC, **bad_pin}, A)
        with self.assertRaises(ValueError):
            m.validate_inputs4b({**GOOD_SPEC, 'schema': 'p-feat01-inputs-v1'}, A)
        with self.assertRaises(ValueError):
            m.validate_inputs4b({**GOOD_SPEC, 'attr_manifest': '/f/manifest.json'}, A)              # half pin
        with self.assertRaises(ValueError):
            m.validate_inputs4b(dict(GOOD_SPEC), 'nope')

    def test_inputs_stats_pin_rules(self):
        """A pin that is used must be set, a pin that is not used must be null: C/D need the stats pair, control / A / E refuse it; ctx / attr stay null for everyone."""
        m = r4()
        for feature in CD:
            m.validate_inputs4b(dict(CD_SPEC), feature)
            with self.assertRaises(ValueError):
                m.validate_inputs4b(dict(GOOD_SPEC), feature)                                       # used pin missing
            for bad in ({'stats_sha256': None}, {'stats_manifest': None}, {'stats_sha256': 'xyz'}, {'stats_manifest': 7},
                        {'attr_manifest': '/f/manifest.json', 'attr_sha256': 'd' * 64}, {'ctx_manifest': '/f/manifest.json', 'ctx_sha256': 'd' * 64}):
                with self.assertRaises(ValueError, msg=str(bad)):
                    m.validate_inputs4b({**CD_SPEC, **bad}, feature)
        for feature in (CONTROL, A, E):
            m.validate_inputs4b(dict(GOOD_SPEC), feature)
            with self.assertRaises(ValueError):
                m.validate_inputs4b(dict(CD_SPEC), feature)                                         # unused pin must be null
        with self.assertRaises(ValueError):
            m.validate_inputs4b({**CD_SPEC, 'schema': 'p-feat02-inputs-v1'}, C)
        with self.assertRaises(ValueError):
            m.validate_inputs4b({k: v for k, v in CD_SPEC.items() if k != 'stats_manifest'}, C)
        with self.assertRaises(ValueError):
            m.validate_inputs4b(dict(BASE_SPEC), A)                                                 # the old schema is for the stats commands only

    def test_apply_feat04b_loads_the_stats_for_c_d_only(self):
        """C/D: item_stats.load_item_stats gets the pinned manifest / sha / metadata / history sha / catalog / support / audited digest; the data gains the bucket ids, the raw
        stats, their digest and the cutoff week; the extra binding names the sidecar. E / A / control never touch the sidecar."""
        from vkgr import item_stats, retrieval_feat04b as m4
        data = syn(); stats = data['stats']; calls = []
        loaded = {'stats': {'stats_version': 'ITEM_STATS_V1', 'cutoff_week': 24, 'rows_total': sum(stats['n']), **{k: stats[k] for k in ('n', 'first_week', 'likes', 'ratio_pm_sum')}},
                  'stats_sha256': item_stats.stats_digest(stats), 'manifest_sha256': 'e' * 64}
        base = {k: v for k, v in data.items() if k not in ('stats', 'stat_buckets', 'stats_sha256', 'stat_cutoff_week')}

        def fake_load(*args, **kwargs):
            calls.append((args, kwargs)); return loaded
        with mock.patch.object(r4().fr, 'apply_feat02', lambda d, spec, length: (d, {'history_manifest_sha256': 'h'})), mock.patch.object(item_stats, 'load_item_stats', fake_load):
            for feature in CD:
                calls.clear(); out, binding = r4().apply_feat04b(dict(base), dict(CD_SPEC), 50, feature)
                self.assertEqual(len(calls), 1); args, kwargs = calls[0]
                self.assertEqual(args, ('/s/manifest.json', 'e' * 64, 'c' * 64, 'a' * 64, base['catalog'], base['support']))
                self.assertEqual(kwargs, {'expected_sha256': m4.STATS_SHA256})
                self.assertEqual(out['stats_sha256'], loaded['stats_sha256']); self.assertEqual(out['stat_cutoff_week'], 24)
                self.assertEqual(out['stats'], loaded['stats'])
                ref = m4.stat_bucket_ids(loaded['stats'], base['catalog'], base['idmap'], len(base['content']))
                self.assertTrue(all(torch.equal(out['stat_buckets'][k], ref[k]) for k in ref))
                self.assertEqual(binding, {'history_manifest_sha256': 'h', 'stats_manifest_sha256': 'e' * 64})
            calls.clear()
            for feature in (CONTROL, A, E):
                out, binding = r4().apply_feat04b(dict(base), dict(GOOD_SPEC), 50, feature)
                self.assertEqual((calls, binding), ([], {'history_manifest_sha256': 'h'})); self.assertNotIn('stat_buckets', out)

    def test_apply_feat04b_refuses_pins_for_a(self):
        with self.assertRaises(ValueError):
            r4().apply_feat04b({}, {**GOOD_SPEC, 'attr_manifest': '/f/manifest.json', 'attr_sha256': 'd' * 64}, 50, A)
        with self.assertRaises(ValueError):
            r4().apply_feat04b({}, dict(CD_SPEC), 50, A)


class TensorTests(unittest.TestCase):
    def test_tensors_equal_the_feat02_f4_tensors_and_fill_the_week_column(self):
        data = fr2().synthetic_data(); rows, queries = fr2()._view(data, 50)
        batch = rows[:20] + queries[:20]; width = width_of(batch)
        for feature in RUNNABLE:
            histories, side = r4()._tensors4b(batch, data['idmap'], width, feature)
            h2, s2 = fr2()._tensors2(batch, data['idmap'], width, 'f3_f4_recency')
            self.assertTrue(torch.equal(histories, h2) and torch.equal(side, s2))
            self.assertTrue(any(int(side[j, 0, 3]) > 0 or int(side[j, -1, 3]) > 0 for j in range(len(batch))))
            self.assertTrue(bool(side[..., 4:].eq(0).all()))
        with self.assertRaises(ValueError):
            r4()._tensors4b(batch, data['idmap'], width, 'nope')


class LeakageTests(unittest.TestCase):
    def test_l1_target_fields_do_not_reach_tensors_or_users(self):
        for variant in RUNNABLE:
            with self.subTest(variant=variant):
                guard_target_flip_leaves_feature_unchanged(variant)

    def test_l2_export_path(self):
        for variant in RUNNABLE:
            with self.subTest(variant=variant):
                guard_l2_export_path_ignores_non_history_fields(variant)

    def test_l4_reverse_controls(self):
        for variant in RUNNABLE:
            with self.subTest(variant=variant):
                guard_l4_reverse_controls(variant)

    def test_l5_week_shift_invariance(self):
        for variant in RUNNABLE:
            with self.subTest(variant=variant):
                guard_l5_week_shift_invariance(variant)

    def test_l6_week_25_rejected(self):
        for variant in RUNNABLE:
            with self.subTest(variant=variant):
                guard_l6_week_25_rejected(variant)

    def test_l7_source_guard(self):
        guard_l7_source_has_no_target_access()

    def test_l7_guard_bites_on_a_leaky_function(self):
        original = fr2()._tensors2

        def leaky(batch, idmap, width, feature):
            for r in batch:
                r['target_position']
            return original(batch, idmap, width, feature)
        with mock.patch.object(fr2(), '_tensors2', leaky):
            with self.assertRaises(AssertionError):
                guard_l7_source_has_no_target_access()


class EquivalenceTests(unittest.TestCase):
    def test_e1_control_t010_equals_php01_h1(self):
        guard_s_equiv_h1()

    def test_e2_a_t015_equals_feat03_q2(self):
        guard_s_equiv_feat03_q2()

    def test_e3_control_t015_equals_feat03_control(self):
        guard_s_equiv_f4_control_t015()

    def test_first_step_loss_equals_anchor(self):
        guard_s_first_step_loss_equals_anchor()

    def test_temperature_plumbing(self):
        guard_s_temperature_plumbing()

    def test_candidate_seam_wiring(self):
        guard_s_candidate_seam_wiring()

    def test_comparison_bites_on_a_wrong_temperature(self):
        """The E-comparison is not vacuous: the same pair of runs at different temperatures must fail it."""
        from vkgr import retrieval_feat03_run as f3r
        data = fr2().synthetic_data()
        f3_config = {**f3r.default_config(), 'feature': CONTROL, 'logq': True, 'batch_size': 16, 'max_epochs': 8, 'min_epochs': 2}
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            r4().run_session(tmp / 'a', data, config4(feature=CONTROL, temperature=0.1)); f3r.run_session(tmp / 'b', data, f3_config)
            with self.assertRaises(AssertionError):
                compare_runs(tmp / 'a', tmp / 'b', _training_keys() | {'feature', 'side_version'})

    def test_untouched(self):
        guard_s_untouched()


class BridgeE4Tests(unittest.TestCase):
    def test_e4_chain_a_t010_equals_feat04a_a(self):
        guard_s_bridge_e4()

    def test_e4_comparison_bites_on_a_different_temperature(self):
        _bridge_runner()
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            tmp = Path(tmp); runner_a = _bridge_runner(); config = config4(feature=A, temperature=0.1)
            (tmp / 'b.json').write_text(json.dumps(config))
            (tmp / 'a.json').write_text(json.dumps({**config, 'schema': runner_a.SCHEMA, 'task_id': runner_a.TASK, 'temperature': 0.15}))
            self.assertEqual(r4().main(['synthetic-train', '--config', str(tmp / 'b.json'), '--output', str(tmp / 'out-b')]), 0)
            self.assertEqual(runner_a.main(['synthetic-train', '--config', str(tmp / 'a.json'), '--output', str(tmp / 'out-a')]), 0)
            with self.assertRaises(AssertionError):
                compare_runs(tmp / 'out-b', tmp / 'out-a', set(config))


class RunnerTests(unittest.TestCase):
    def test_each_variant_trains_and_verifies(self):
        from vkgr import retrieval_feat04b as m4
        for feature in RUNNABLE:
            with self.subTest(feature=feature), tempfile.TemporaryDirectory() as tmp:
                data = syn(); config = config4(feature=feature); run = Path(tmp) / 'run'
                value = r4().run_session(run, data, config)
                self.assertEqual(sorted(p.name for p in run.glob('*.pt')), ['best.pt', 'last.pt'])
                self.assertEqual((value['schema'], value['kind'], value['status']), ('p-feat04b-train-manifest-v1', 'p_feat04b_train', 'complete'))
                binding = value['binding']
                self.assertEqual((binding['task'], binding['feature'], binding['temperature'], binding['side_version']), ('P-FEAT04B', feature, 0.1, 'HIST_V2_SIDE_V1'))
                self.assertEqual(binding['bucket_spec_sha256'], m4.bucket_spec_sha256(feature))
                self.assertEqual(binding['param_report'], m4.expected_param_counts4b(feature))
                self.assertIn('weeks_sha256', binding); self.assertIn('side_sha256', binding)
                self.assertNotIn('attr_sha256', binding); self.assertNotIn('ctx_sha256', binding)
                if feature in CD:
                    self.assertEqual((binding['stat_cutoff_week'], binding['stats_sha256']), (24, data['stats_sha256'])); self.assertEqual(len(binding['stat_bucket_ids_sha256']), 64)
                    shifted = {k: v.clone() for k, v in data['stat_buckets'].items()}
                    for v in shifted.values():
                        v[1:] = v[1:].roll(1)
                    with self.assertRaisesRegex(ValueError, 'binding mismatch'):
                        r4().verify_run(run / 'manifest.json', value['manifest_sha256'], {**data, 'stat_buckets': shifted}, config)
                    with self.assertRaisesRegex(ValueError, 'binding mismatch'):
                        r4().verify_run(run / 'manifest.json', value['manifest_sha256'], {**data, 'stat_cutoff_week': 25}, config)
                    with self.assertRaisesRegex(ValueError, 'binding mismatch'):
                        r4().verify_run(run / 'manifest.json', value['manifest_sha256'], {**data, 'stats_sha256': 'f' * 64}, config)
                else:
                    self.assertFalse(any(k in binding for k in ('stat_cutoff_week', 'stats_sha256', 'stat_bucket_ids_sha256')))
                self.assertEqual(binding['execution']['device'], 'cpu')
                self.assertEqual(r4().verify_run(run / 'manifest.json', value['manifest_sha256'], data, config)['status'], 'verified')
                other = CONTROL if feature != CONTROL else A
                with self.assertRaises(ValueError):
                    r4().verify_run(run / 'manifest.json', value['manifest_sha256'], data, {**config, 'feature': other})
                with self.assertRaises(ValueError):
                    r4().verify_run(run / 'manifest.json', value['manifest_sha256'], data, {**config, 'temperature': 0.15})
                side = {**data, 'rows': [{**r, 'side': [[mask ^ 1, t, d] for mask, t, d in r['side']]} for r in data['rows']]}
                with self.assertRaisesRegex(ValueError, 'binding mismatch'):
                    r4().verify_run(run / 'manifest.json', value['manifest_sha256'], side, config)
                weeks = {**data, 'rows': [{**r, 'history_positions': [[(w + 1) % 25 if k == len(r['history_positions']) - 1 else w, o] for k, (w, o) in enumerate(r['history_positions'])]}
                                          for r in data['rows']]}
                with self.assertRaisesRegex(ValueError, 'binding mismatch'):
                    r4().verify_run(run / 'manifest.json', value['manifest_sha256'], weeks, config)

    def test_binding_hash_is_per_variant(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = syn(); seen = {}
            for feature in RUNNABLE:
                seen[feature] = r4().run_session(Path(tmp) / feature, data, config4(feature=feature))['binding']['bucket_spec_sha256']
            self.assertEqual(len(set(seen.values())), len(RUNNABLE))

    def test_verify_requires_cuda_for_real_data_and_pin(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = fr2().synthetic_data(); config = config4(); value = r4().run_session(Path(tmp) / 'run', data, config); run = Path(tmp) / 'run'
            with self.assertRaises(ValueError):
                r4().verify_run(run / 'manifest.json', value['manifest_sha256'], {**data, 'synthetic': False}, config)
            with self.assertRaises(ValueError):
                r4().verify_run(run / 'manifest.json', '0' * 64, data, config)

    def test_a_logit_diagnostics_per_epoch_and_recomputed_for_the_best_checkpoint(self):
        from vkgr.data_audit import file_sha256
        from vkgr.retrieval_ablation_run import strict_json
        with tempfile.TemporaryDirectory() as tmp:
            data = fr2().synthetic_data(); config = config4(feature=A); value = r4().run_session(Path(tmp) / 'run', data, config); run = Path(tmp) / 'run'
            for evaluation in value['evaluations']:
                logits = evaluation['diagnostics']['recency_logits']
                self.assertEqual((len(logits['position']), len(logits['weekgap'])), (7, 6))
            last = value['evaluations'][-1]['diagnostics']['recency_logits']
            self.assertTrue(any(v != 0.0 for v in last['position'] + last['weekgap']), 'training must move the logits')
            state = torch.load(run / 'last.pt', map_location='cpu', weights_only=True)['model']
            self.assertEqual(last, {'position': state['recency_logit_pos'].tolist(), 'weekgap': state['recency_logit_gap'].tolist()})
            forged = strict_json((run / 'manifest.json').read_text()); forged['evaluations'][value['best']['epoch'] - 1]['diagnostics']['recency_logits']['position'][0] += 0.25
            (run / 'forged.json').write_text(json.dumps(forged))
            with self.assertRaisesRegex(ValueError, 'diagnostics'):
                r4().verify_run(run / 'forged.json', file_sha256(run / 'forged.json'), data, config)

    def test_control_run_has_no_logit_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            value = r4().run_session(Path(tmp) / 'run', fr2().synthetic_data(), config4(feature=CONTROL))
            self.assertNotIn('recency_logits', value['evaluations'][0]['diagnostics'])
            self.assertNotIn('behavior_beta', value['evaluations'][0]['diagnostics']); self.assertNotIn('beta_bit_train_counts', value['data_diagnostics'])

    def test_a_run_has_logits_but_no_beta_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            value = r4().run_session(Path(tmp) / 'run', fr2().synthetic_data(), config4(feature=A))
            self.assertIn('recency_logits', value['evaluations'][0]['diagnostics'])
            self.assertNotIn('behavior_beta', value['evaluations'][0]['diagnostics']); self.assertNotIn('beta_bit_train_counts', value['data_diagnostics'])

    def test_e_beta_diagnostics_per_epoch_bit_counts_and_verify_recomputation(self):
        from vkgr.data_audit import file_sha256
        from vkgr.history_side import FEEDBACK_BITS
        from vkgr.retrieval_ablation_run import strict_json
        with tempfile.TemporaryDirectory() as tmp:
            data = fr2().synthetic_data(); config = config4(feature=E); value = r4().run_session(Path(tmp) / 'run', data, config); run = Path(tmp) / 'run'
            for evaluation in value['evaluations']:
                diag = evaluation['diagnostics']
                self.assertEqual(list(diag['behavior_beta']), list(FEEDBACK_BITS)); self.assertEqual(len(diag['recency_logits']['position']), 7)
            last = value['evaluations'][-1]['diagnostics']['behavior_beta']
            self.assertTrue(any(v != 0.0 for v in last.values()), 'training must move beta')
            state = torch.load(run / 'last.pt', map_location='cpu', weights_only=True)['model']
            self.assertEqual(last, dict(zip(FEEDBACK_BITS, state['behavior_beta'].tolist())))
            counts = value['data_diagnostics']['beta_bit_train_counts']
            rows = fr2()._view(data, 50)[0]
            self.assertEqual(counts, {name: sum((m >> b) & 1 for r in rows for m, _, _ in r['side']) for b, name in enumerate(FEEDBACK_BITS)})
            self.assertTrue(all(v > 0 for v in counts.values()))
            forged = strict_json((run / 'manifest.json').read_text()); forged['evaluations'][value['best']['epoch'] - 1]['diagnostics']['behavior_beta']['like'] += 0.25
            (run / 'forged.json').write_text(json.dumps(forged))
            with self.assertRaisesRegex(ValueError, 'diagnostics'):
                r4().verify_run(run / 'forged.json', file_sha256(run / 'forged.json'), data, config)
            forged = strict_json((run / 'manifest.json').read_text()); forged['data_diagnostics']['beta_bit_train_counts']['like'] += 1
            (run / 'forged2.json').write_text(json.dumps(forged))
            with self.assertRaisesRegex(ValueError, 'beta_bit_train_counts'):
                r4().verify_run(run / 'forged2.json', file_sha256(run / 'forged2.json'), data, config)

    def test_nonfinite_gradient_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch('torch.nn.utils.clip_grad_norm_', lambda *a, **k: torch.tensor(float('nan'))):
                with self.assertRaises(ValueError):
                    r4().run_session(Path(tmp) / 'run', fr2().synthetic_data(), config4())

    def test_real_run_requires_cuda_and_inputs_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = {**fr2().synthetic_data(), 'synthetic': False}
            with self.assertRaises(ValueError):
                r4().run_session(Path(tmp) / 'run', data, real_config(), 'cpu')


class CDRunnerTests(unittest.TestCase):
    def test_train_step_uses_candidate_items(self):
        guard_cd_train_step_uses_candidate_items()
        guard_cd_train_step_tables_receive_gradient()

    def test_export_uses_candidate_items(self):
        guard_cd_export_uses_candidate_items()

    def test_binding_has_cutoff(self):
        guard_cd_binding_has_cutoff()

    def test_candidate_side_leakage_controls(self):
        for variant in CD:
            with self.subTest(variant=variant):
                guard_l3_candidate_reverse_control(variant)
                guard_l5_candidate_side_depends_on_item_id_only(variant)
                guard_l9_train_and_export_vectors_equal(variant)

    def test_synthetic_world_is_a_superset_of_the_feat02_world(self):
        data, base = syn(), fr2().synthetic_data()
        self.assertTrue(all(k in data and (data[k] == base[k] if not isinstance(base[k], torch.Tensor) else torch.equal(data[k], base[k])) for k in base))
        self.assertEqual(data['stat_cutoff_week'], 24); self.assertEqual(len(data['stats']['n']), len(data['catalog']))
        for name, ids in data['stat_buckets'].items():
            self.assertEqual((ids.dtype, tuple(ids.shape), int(ids[0])), (torch.long, (len(data['content']),), 0))
            self.assertGreaterEqual(len({int(v) for v in ids[1:]}), 3, name)                    # the synthetic stats spread over several buckets
        from vkgr import item_stats
        self.assertEqual(data['stats_sha256'], item_stats.stats_digest(data['stats']))
        self.assertEqual(syn()['stats'], data['stats'])                                          # seeded: deterministic

    def test_top50_stat_profile_per_epoch_and_recomputed_in_verify(self):
        from vkgr.data_audit import file_sha256
        from vkgr.retrieval_ablation_run import strict_json
        for variant in CD:
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as tmp:
                data = syn(); config = config4(feature=variant); run = Path(tmp) / 'run'; value = r4().run_session(run, data, config)
                names = {C: ('xb', 'fb'), D: ('rb', 'lb')}[variant]; labels = {'xb': 'expo', 'fb': 'first', 'rb': 'ratio', 'lb': 'like'}
                for evaluation in value['evaluations']:
                    predictions = json.loads((run / evaluation['predictions_ref']).read_text()); profile = evaluation['diagnostics']['top50_stat_profile']
                    expected = {'slots': sum(len(p[:50]) for p in predictions)}
                    for name in names:
                        counts = {}
                        for p in predictions:
                            for item in p[:50]:
                                bucket = int(data['stat_buckets'][name][data['idmap'][item]]); counts[str(bucket)] = counts.get(str(bucket), 0) + 1
                        expected[labels[name]] = dict(sorted(counts.items(), key=lambda kv: int(kv[0])))
                    self.assertEqual(profile, expected)
                    self.assertEqual(sum(profile[labels[names[0]]].values()), profile['slots'])
                forged = strict_json((run / 'manifest.json').read_text()); forged['evaluations'][value['best']['epoch'] - 1]['diagnostics']['top50_stat_profile']['slots'] += 1
                (run / 'forged.json').write_text(json.dumps(forged))
                with self.assertRaisesRegex(ValueError, 'top50_stat_profile|diagnostics'):
                    r4().verify_run(run / 'forged.json', file_sha256(run / 'forged.json'), data, config)
        for variant in (CONTROL, A, E):
            with tempfile.TemporaryDirectory() as tmp:
                value = r4().run_session(Path(tmp) / 'run', syn(), config4(feature=variant))
                self.assertNotIn('top50_stat_profile', value['evaluations'][0]['diagnostics']); self.assertNotIn('target_bucket_hist', value['data_diagnostics'])

    def test_data_diagnostics_hooks_are_recomputed_in_verify(self):
        from vkgr import item_stats, retrieval_feat04b as m4
        from vkgr.data_audit import file_sha256
        from vkgr.retrieval_negatives import target_frequency
        from vkgr.retrieval_ablation_run import strict_json
        fns = {'expo': m4.expo_bucket, 'first': m4.first_bucket, 'ratio': m4.ratio_bucket4, 'like': m4.like_bucket}
        for variant in CD:
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as tmp:
                data = syn(); config = config4(feature=variant); run = Path(tmp) / 'run'; value = r4().run_session(run, data, config)
                rows, queries = fr2()._view(data, 50)
                hist = item_stats.target_bucket_hist(rows, queries, data['catalog'], data['stats'], fns)
                red = item_stats.logq_redundancy(data['stats'], data['catalog'], data['support'], target_frequency(rows)['counts'], fns)
                self.assertEqual(value['data_diagnostics']['target_bucket_hist'], json.loads(json.dumps(hist)))
                self.assertEqual(value['data_diagnostics']['logq_redundancy'], json.loads(json.dumps(red)))
                self.assertEqual(hist['train']['examples'], len(rows)); self.assertEqual(hist['validation']['queries'], len(queries))
                for key, edit in (('target_bucket_hist', lambda d: d['train'].update(examples=d['train']['examples'] + 1)), ('logq_redundancy', lambda d: d.update(supported_items=d['supported_items'] + 1))):
                    forged = strict_json((run / 'manifest.json').read_text()); edit(forged['data_diagnostics'][key]); (run / 'forged.json').write_text(json.dumps(forged))
                    with self.assertRaisesRegex(ValueError, key):
                        r4().verify_run(run / 'forged.json', file_sha256(run / 'forged.json'), data, config)

    def test_tables_train_and_the_best_checkpoint_reloads_with_its_bucket_ids(self):
        for variant in CD:
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as tmp:
                data = syn(); config = config4(feature=variant); run = Path(tmp) / 'run'; r4().run_session(run, data, config)
                state = torch.load(run / 'last.pt', map_location='cpu', weights_only=True)['model']
                fresh = r4()._model4b(data, config, 'cpu').state_dict()
                from vkgr import retrieval_feat04b as m4
                for table, buffer, _ in m4.STAT_LAYOUT[variant]:
                    self.assertFalse(torch.equal(state[f'{table}.weight'], fresh[f'{table}.weight']), f'{table} did not train')
                    self.assertTrue(torch.equal(state[buffer], fresh[buffer]) and torch.equal(state[f'{table}.weight'][0], torch.zeros(64)))
                for key in ('embedding.weight',):
                    self.assertFalse(torch.equal(state[key], fresh[key]))


class CliTests(unittest.TestCase):
    def test_default_device_is_cpu(self):
        guard_s_cli_default_device_cpu()

    def test_device_choices(self):
        parser = r4().build_parser()
        for bad in ('tpu', 'cuda:0', ''):
            for argv in (['synthetic-train', '--config', 'c', '--output', 'o', '--device', bad], ['synthetic-verify', '--config', 'c', '--manifest', 'm', '--manifest-sha', 'a' * 64, '--device', bad]):
                with self.assertRaises(SystemExit) as cm, contextlib.redirect_stderr(io.StringIO()):
                    parser.parse_args(argv)
                self.assertEqual(cm.exception.code, 2)

    def test_cpu_round_trip(self):
        for feature in RUNNABLE:
            with self.subTest(feature=feature), tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp); (tmp / 'config.json').write_text(json.dumps(config4(feature=feature)))
                code, out, err = run_cli(['synthetic-train', '--config', str(tmp / 'config.json'), '--output', str(tmp / 'run')])
                self.assertEqual((code, err), (0, '')); result = json.loads(out); self.assertEqual(result['status'], 'complete')
                common = ['--config', str(tmp / 'config.json'), '--manifest', str(tmp / 'run' / 'manifest.json'), '--manifest-sha', result['manifest_sha256']]
                code, out, err = run_cli(['synthetic-verify', *common])
                self.assertEqual((code, err, json.loads(out)['status']), (0, '', 'verified'))
                code, out, _ = run_cli(['synthetic-verify', *common, '--device', 'cpu'])
                self.assertEqual((code, json.loads(out)['status']), (0, 'verified'))
                code, out, err = run_cli(['synthetic-verify', '--config', str(tmp / 'config.json'), '--manifest', str(tmp / 'run' / 'manifest.json'), '--manifest-sha', '0' * 64])
                self.assertEqual((code, out), (2, '')); self.assertIn('ERROR P-FEAT04B gate failed', err)

    def test_default_synthetic_train_equals_a_direct_run_session_except_timing(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp); config = config4(); (tmp / 'config.json').write_text(json.dumps(config))
            code, _, _ = run_cli(['synthetic-train', '--config', str(tmp / 'config.json'), '--output', str(tmp / 'cli')])
            self.assertEqual(code, 0)
            r4().run_session(tmp / 'direct', fr2().synthetic_data(), config)
            a = json.loads((tmp / 'cli' / 'manifest.json').read_text()); b = json.loads((tmp / 'direct' / 'manifest.json').read_text())
            self.assertEqual(a['binding']['execution']['device'], 'cpu')
            self.assertEqual(strip_timing(a), strip_timing(b))

    def test_cuda_without_cuda_exits_2_with_empty_stdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp); (tmp / 'config.json').write_text(json.dumps(config4()))
            with mock.patch('torch.cuda.is_available', return_value=False):
                code, out, err = run_cli(['synthetic-train', '--config', str(tmp / 'config.json'), '--output', str(tmp / 'run'), '--device', 'cuda'])
                self.assertEqual((code, out), (2, '')); self.assertEqual(err.strip(), CUDA_MESSAGE); self.assertFalse((tmp / 'run').exists())
                code, out, err = run_cli(['synthetic-verify', '--config', str(tmp / 'config.json'), '--manifest', str(tmp / 'm.json'), '--manifest-sha', 'a' * 64, '--device', 'cuda'])
                self.assertEqual((code, out), (2, '')); self.assertEqual(err.strip(), CUDA_MESSAGE)

    def test_cuda_without_deterministic_workspace_exits_2(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch('torch.cuda.is_available', return_value=True), mock.patch.dict('os.environ'):
            import os
            os.environ.pop('CUBLAS_WORKSPACE_CONFIG', None)
            tmp = Path(tmp); (tmp / 'config.json').write_text(json.dumps(config4()))
            code, out, err = run_cli(['synthetic-train', '--config', str(tmp / 'config.json'), '--output', str(tmp / 'run'), '--device', 'cuda'])
            self.assertEqual((code, out), (2, '')); self.assertIn('ERROR P-FEAT04B gate failed: ValueError', err)

    def _real_patches(self, m, fake_data):
        return [mock.patch('torch.cuda.is_available', return_value=True), mock.patch.object(m.b1, 'resource_guard', lambda *a, **k: (lambda: None)),
                mock.patch.object(m.b1, 'real_inputs', lambda s: (fake_data, None, {'prepared_sha': 'x'})),
                mock.patch.object(m, 'apply_feat04b', lambda d, s, n, f: (d, {'history_manifest_sha256': 'h'}))]

    def test_real_path_wiring_train_and_verify(self):
        from vkgr.data_audit import file_sha256
        m = r4()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp); (tmp / 'inputs.json').write_text(json.dumps(CD_SPEC)); (tmp / 'real.json').write_text(json.dumps(real_config()))
            common = ['--inputs', str(tmp / 'inputs.json'), '--inputs-sha', file_sha256(tmp / 'inputs.json'), '--config', str(tmp / 'real.json'), '--config-sha', file_sha256(tmp / 'real.json')]
            patches = self._real_patches(m, {'rows': [], 'queries': [], 'catalog': []}) + [
                mock.patch.object(m, 'run_session', lambda *a, **k: {'status': 'complete', 'manifest_sha256': 'm', 'selection': {'best_epoch': 3}}),
                mock.patch.object(m, 'verify_run', lambda *a, **k: {'status': 'verified'})]
            with contextlib.ExitStack() as stack:
                for p in patches:
                    stack.enter_context(p)
                code, out, _ = run_cli(['train', *common, '--output', str(tmp / 'real-run')])
                self.assertEqual((code, json.loads(out)['best_epoch']), (0, 3))
                code, out, _ = run_cli(['verify', *common, '--manifest', str(tmp / 'real-run' / 'manifest.json'), '--manifest-sha', 'a' * 64])
                self.assertEqual((code, json.loads(out)['status']), (0, 'verified'))
                for rejected in (CONTROL, A):                       # fixtures are never submitted as real runs of this runner
                    (tmp / 'fixture.json').write_text(json.dumps(real_config(feature=rejected)))
                    code, out, err = run_cli(['train', '--inputs', str(tmp / 'inputs.json'), '--inputs-sha', file_sha256(tmp / 'inputs.json'), '--config', str(tmp / 'fixture.json'),
                                              '--config-sha', file_sha256(tmp / 'fixture.json'), '--output', str(tmp / 'x')])
                    self.assertEqual((code, out), (2, '')); self.assertIn('P-FEAT04B', err); self.assertFalse((tmp / 'x').exists())
                (tmp / 'pinned.json').write_text(json.dumps({**CD_SPEC, 'ctx_manifest': '/f/manifest.json', 'ctx_sha256': 'd' * 64}))
                code, out, err = run_cli(['train', '--inputs', str(tmp / 'pinned.json'), '--inputs-sha', file_sha256(tmp / 'pinned.json'), '--config', str(tmp / 'real.json'),
                                          '--config-sha', file_sha256(tmp / 'real.json'), '--output', str(tmp / 'y')])
                self.assertEqual((code, out), (2, '')); self.assertIn('P-FEAT04B', err); self.assertFalse((tmp / 'y').exists())


class PreflightTests(unittest.TestCase):
    def test_preflight_check_passes_on_synthetic_data_for_a(self):
        data = fr2().synthetic_data(); report = r4().preflight_check(data, config4(feature=A), None)
        self.assertEqual(report['status'], 'preflight_ok'); self.assertEqual(report['feature'], A)
        self.assertEqual(report['supported_items'], len(data['support']))
        self.assertIn('binding_sha256', report); self.assertEqual(report['param_report']['by_table']['recency_logit_pos']['total'], 7)
        json.dumps(report)

    def test_preflight_check_passes_on_synthetic_data_for_e(self):
        data = fr2().synthetic_data(); report = r4().preflight_check(data, config4(feature=E), None)
        self.assertEqual((report['status'], report['feature']), ('preflight_ok', E))
        self.assertEqual(report['param_report']['by_table']['behavior_beta']['total'], 6); json.dumps(report)

    def test_preflight_check_passes_on_synthetic_data_for_c_d(self):
        from vkgr import retrieval_feat04b as m4
        for variant in CD:
            data = syn(); report = r4().preflight_check(data, config4(feature=variant), {**CD_SPEC}, None)
            self.assertEqual((report['status'], report['feature']), ('preflight_ok', variant))
            for table, buffer, _ in m4.STAT_LAYOUT[variant]:
                self.assertIn(table, report['param_report']['by_table'])
                shares = report['stat_buckets'][buffer]
                self.assertEqual(sum(shares['catalog'].values()), len(data['catalog'])); self.assertEqual(sum(shares['supported'].values()), len(data['support']))
            self.assertEqual(report['sidecar'], {'side_sha256': 'b' * 64, 'history_sha256': 'a' * 64, 'stats_sha256': 'e' * 64}); json.dumps(report)

    def test_preflight_check_fails_when_c_d_break_the_contract(self):
        from vkgr import retrieval_feat04b as m4
        data = syn()
        for variant in CD:
            config = config4(feature=variant)
            with mock.patch.object(m4, 'candidate_stat_terms', lambda model, indices: torch.zeros(*indices.shape, 64)):          # the statistics never reach the candidates
                with self.assertRaises(ValueError):
                    r4().preflight_check(data, config, None)
            with mock.patch.object(m4, 'candidate_stat_terms', lambda model, indices: torch.full((*indices.shape, 64), 0.01)):  # no supported mask
                with self.assertRaises(ValueError):
                    r4().preflight_check(data, config, None)
            original = m4.Feat04bIDMean.history_items
            with mock.patch.object(m4.Feat04bIDMean, 'history_items', lambda self, indices: original(self, indices) + (0.01 * (indices != 0).unsqueeze(-1) if self.variant in CD else 0.0)):   # user tower touched
                with self.assertRaises(ValueError):
                    r4().preflight_check(data, config, None)

    def test_preflight_check_fails_when_e_does_not_start_at_a(self):
        from vkgr import retrieval_feat04b as m4

        def noisy(model):
            model.behavior_beta = torch.nn.Parameter(torch.linspace(0.1, 0.6, 6))
        with mock.patch.object(m4, 'init_behavior_beta', noisy):
            with self.assertRaises(ValueError):
                r4().preflight_check(fr2().synthetic_data(), config4(feature=E), None)

    def test_preflight_check_fails_when_a_does_not_start_at_the_anchor(self):
        from vkgr import retrieval_feat03 as f3

        def noisy(model):
            model.recency_logit_pos = torch.nn.Parameter(torch.linspace(0.0, 1.0, 7)); model.recency_logit_gap = torch.nn.Parameter(torch.zeros(6))
        with mock.patch.object(f3, 'init_recency_logits', noisy):
            with self.assertRaises(ValueError):
                r4().preflight_check(fr2().synthetic_data(), config4(feature=A), None)

    def test_preflight_check_fails_when_the_user_vector_reads_the_target(self):
        original = fr2()._tensors2

        def leaky(batch, idmap, width, feature):
            histories, side = original(batch, idmap, width, feature)
            for j, r in enumerate(batch):
                side[j, len(r['history']) - 1, 3] = min(r['target_position'][0], 24)
            return histories, side
        with mock.patch.object(fr2(), '_tensors2', leaky):
            with self.assertRaises(ValueError):
                r4().preflight_check(fr2().synthetic_data(), config4(feature=A), None)

    def test_preflight_cli_wiring_and_exit_codes(self):
        from vkgr.data_audit import file_sha256
        m = r4()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp); (tmp / 'inputs.json').write_text(json.dumps(CD_SPEC)); (tmp / 'real.json').write_text(json.dumps(real_config()))
            common = ['--inputs', str(tmp / 'inputs.json'), '--inputs-sha', file_sha256(tmp / 'inputs.json'), '--config', str(tmp / 'real.json'), '--config-sha', file_sha256(tmp / 'real.json')]
            data = fr2().synthetic_data()
            with contextlib.ExitStack() as stack:
                stack.enter_context(mock.patch.object(m.b1, 'real_inputs', lambda s: (data, None, {'prepared_sha': 'x'})))
                stack.enter_context(mock.patch.object(m, 'apply_feat04b', lambda d, s, n, f: (d, {'history_manifest_sha256': 'h'})))
                stack.enter_context(mock.patch.object(m, 'preflight_check', lambda *a, **k: {'status': 'preflight_ok'}))       # C/D/E checks arrive in a later feature
                stack.enter_context(mock.patch('torch.cuda.is_available', return_value=False))          # preflight must not need CUDA
                code, out, err = run_cli(['preflight', *common])
                self.assertEqual((code, err), (0, '')); self.assertEqual(json.loads(out)['status'], 'preflight_ok')
                (tmp / 'bad.json').write_text(json.dumps(real_config(temperature=0.15)))
                code, out, err = run_cli(['preflight', '--inputs', str(tmp / 'inputs.json'), '--inputs-sha', file_sha256(tmp / 'inputs.json'),
                                          '--config', str(tmp / 'bad.json'), '--config-sha', file_sha256(tmp / 'bad.json')])
                self.assertEqual((code, out), (2, '')); self.assertIn('ERROR P-FEAT04B gate failed', err)
                for fixture in (CONTROL, A):
                    (tmp / 'control.json').write_text(json.dumps(real_config(feature=fixture)))
                    code, out, err = run_cli(['preflight', '--inputs', str(tmp / 'inputs.json'), '--inputs-sha', file_sha256(tmp / 'inputs.json'),
                                              '--config', str(tmp / 'control.json'), '--config-sha', file_sha256(tmp / 'control.json')])
                    self.assertEqual((code, out), (2, ''))


    def test_preflight_cli_runs_the_real_check_on_synthetic_data_for_c_d_e(self):
        """Item 16: the `preflight` command (config + inputs validation, real_inputs / apply seams mocked to the synthetic world, the real preflight_check) exits 0 with a JSON
        report for C, D and E, and 2 with one stderr line and empty stdout for a bad config; it never needs CUDA and writes nothing."""
        from vkgr.data_audit import file_sha256
        m = r4()
        for feature, spec in ((C, CD_SPEC), (D, CD_SPEC), (E, GOOD_SPEC)):
            with self.subTest(feature=feature), tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp); (tmp / 'inputs.json').write_text(json.dumps(spec)); (tmp / 'real.json').write_text(json.dumps(real_config(feature=feature)))
                (tmp / 'bad.json').write_text(json.dumps(real_config(feature=feature, temperature=0.15)))
                data = syn()

                def argv(config):
                    return ['preflight', '--inputs', str(tmp / 'inputs.json'), '--inputs-sha', file_sha256(tmp / 'inputs.json'), '--config', str(tmp / config), '--config-sha', file_sha256(tmp / config)]
                with contextlib.ExitStack() as stack:
                    stack.enter_context(mock.patch.object(m.b1, 'real_inputs', lambda s: (data, None, {'prepared_sha': 'x'})))
                    stack.enter_context(mock.patch.object(m, 'apply_feat04b', lambda d, s, n, f: (d, {'history_manifest_sha256': 'h'})))
                    stack.enter_context(mock.patch('torch.cuda.is_available', return_value=False))
                    before = sorted(p.name for p in tmp.iterdir())
                    code, out, err = run_cli(argv('real.json'))
                    self.assertEqual((code, err), (0, '')); report = json.loads(out)
                    self.assertEqual((report['status'], report['feature'], report['temperature']), ('preflight_ok', feature, 0.1)); self.assertEqual(len(report['binding_sha256']), 64)
                    self.assertEqual(sorted(p.name for p in tmp.iterdir()), before)
                    code, out, err = run_cli(argv('bad.json'))
                    self.assertEqual((code, out), (2, '')); self.assertEqual(len(err.strip().splitlines()), 1); self.assertTrue(err.startswith('ERROR P-FEAT04B gate failed: '))


class StatsCommandTests(unittest.TestCase):
    """range-check / prepare-stats / verify-stats on tiny synthetic raw partitions (mocked ``b1.real_inputs``, a real items_metadata parquet on disk)."""
    CATALOG, DURATIONS = [10, 20, 30], {10: 10, 20: 4, 30: 255}

    def _row(self, item, timespent, like=False):
        return {'user_id': 1, 'item_id': item, 'timespent': timespent, 'like': like}

    def _partitions(self):
        r = self._row
        return [(0, [r(10, 5, True), r(20, 4), r(10, 0)]), (1, [r(30, 255, True), r(10, 30)]), (3, [r(20, 8, True), r(30, 0)])]

    def _world(self, tmp, partitions=None, durations=None, extra_rows=()):
        import pyarrow as pa
        import pyarrow.parquet as pq
        from vkgr.data_audit import file_sha256
        tmp = Path(tmp); durations = dict(self.DURATIONS if durations is None else durations)
        (tmp / 'raw' / 'metadata').mkdir(parents=True)
        items = list(durations) + [99]; values = [durations[i] for i in durations] + [7]
        pq.write_table(pa.table({'item_id': pa.array(items, pa.uint32()), 'author_id': pa.array([1] * len(items), pa.uint32()), 'duration': pa.array(values, pa.uint8())}),
                       str(tmp / 'raw' / 'metadata' / 'items_metadata.parquet'))
        parts = self._partitions() if partitions is None else partitions
        if extra_rows:
            parts = [(w, rows + list(extra_rows) if w == 1 else rows) for w, rows in parts]
        spec = {**BASE_SPEC, 'raw_root': str(tmp / 'raw'), 'prepared': str(tmp / 'p'), 'holdout': str(tmp / 'h'),
                'items_metadata_sha256': file_sha256(tmp / 'raw' / 'metadata' / 'items_metadata.parquet')}
        (tmp / 'inputs.json').write_text(json.dumps(spec))
        data = {'catalog': list(self.CATALOG), 'support': {10, 30}, 'queries': [{'target': 10}]}
        binding = {'prepared_sha': 'p', 'query_sha': 'q' * 8, 'catalog_sha': 'c', 'test_gate': 'closed_hash_only', 'raw_records': [{'role': 'week_00.parquet', 'sha256': 'r'}]}
        fake = lambda s, with_partitions=False: (data, (lambda: iter([(w, list(rows)) for w, rows in parts])), dict(binding))
        return spec, fake, ['--inputs', str(tmp / 'inputs.json'), '--inputs-sha', file_sha256(tmp / 'inputs.json')]

    def _cli(self, fake, argv):
        m = r4()
        with mock.patch.object(m.b1, 'real_inputs', fake), mock.patch.object(m.b1, 'source_identity', lambda: {'source_sha': 'x'}):
            return run_cli(argv)

    def _one_error_line(self, err):
        lines = err.strip().splitlines()
        self.assertEqual(len(lines), 1, err); self.assertTrue(lines[0].startswith('ERROR P-FEAT04B gate failed: '), err)

    def test_range_check_passes_on_clean_partitions_and_is_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec, fake, common = self._world(tmp)
            before = sorted(str(p) for p in Path(tmp).rglob('*'))
            code, out, err = self._cli(fake, ['range-check', *common])
            self.assertEqual((code, err), (0, ''))
            report = json.loads(out)
            self.assertEqual((report['status'], report['rows'], report['partitions'], report['timespent_zero_rows'], report['rows_outside_catalog']), ('range_check_ok', 7, 3, 2, 0))
            self.assertEqual((report['timespent_min'], report['timespent_max'], report['duration_min'], report['duration_max'], report['violations']), (0, 255, 4, 255, []))
            self.assertEqual(out.count('\n'), 1); self.assertNotIn(str(tmp), out)
            self.assertEqual(sorted(str(p) for p in Path(tmp).rglob('*')), before)                     # nothing written

    def test_range_check_on_an_out_of_range_partition_exits_2_and_prints_the_evidence(self):
        for label, kwargs, expect in (('timespent 256', {'extra_rows': [{'user_id': 1, 'item_id': 10, 'timespent': 256, 'like': False}]}, 'timespent range'),
                                      ('timespent -1', {'extra_rows': [{'user_id': 1, 'item_id': 10, 'timespent': -1, 'like': False}]}, 'timespent range'),
                                      ('non-int', {'extra_rows': [{'user_id': 1, 'item_id': 10, 'timespent': 1.5, 'like': False}]}, 'non-int timespent'),
                                      ('outside catalog', {'extra_rows': [{'user_id': 1, 'item_id': 99, 'timespent': 3, 'like': False}]}, 'outside the catalog'),
                                      ('duration 0', {'durations': {10: 0, 20: 4, 30: 255}}, 'duration range')):
            with self.subTest(label), tempfile.TemporaryDirectory() as tmp:
                spec, fake, common = self._world(tmp, **kwargs)
                code, out, err = self._cli(fake, ['range-check', *common])
                self.assertEqual(code, 2); self._one_error_line(err); self.assertIn(expect, err)
                report = json.loads(out)                                   # the evidence still reaches stdout
                self.assertEqual(report['status'], 'range_check_failed'); self.assertTrue(report['violations'])
                self.assertFalse((Path(tmp) / 'stats').exists())

    def test_prepare_then_verify_round_trip_and_the_sidecar_binding(self):
        from vkgr import item_stats
        from vkgr.data_audit import file_sha256
        with tempfile.TemporaryDirectory() as tmp:
            spec, fake, common = self._world(tmp)
            code, out, err = self._cli(fake, ['prepare-stats', *common, '--output', str(Path(tmp) / 'stats')])
            self.assertEqual((code, err), (0, '')); self.assertEqual(json.loads(out)['status'], 'stats_prepared')
            manifest_path = Path(tmp) / 'stats' / 'manifest.json'
            self.assertEqual(json.loads(out)['manifest_sha256'], file_sha256(manifest_path))
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual((manifest['stats_version'], manifest['cutoff_week'], manifest['rows_total'], manifest['items_metadata_sha256'], manifest['history_manifest_sha256']),
                             ('ITEM_STATS_V1', 24, 7, spec['items_metadata_sha256'], spec['history_sha256']))
            self.assertEqual(manifest['binding']['inputs_sha256'], file_sha256(Path(tmp) / 'inputs.json'))
            self.assertNotIn('query_sha', manifest['binding']); self.assertEqual(manifest['binding']['consumer'], {'source_sha': 'x'})
            self.assertEqual(manifest['binding']['raw_records'], [{'role': 'week_00.parquet', 'sha256': 'r'}])
            loaded = item_stats.load_item_stats(manifest_path, file_sha256(manifest_path), spec['items_metadata_sha256'], spec['history_sha256'], self.CATALOG, {10, 30})
            self.assertEqual(loaded['stats']['n'], [3, 2, 2])
            code, out, err = self._cli(fake, ['verify-stats', *common, '--manifest', str(manifest_path), '--manifest-sha', file_sha256(manifest_path)])
            self.assertEqual((code, err, json.loads(out)), (0, '', {'status': 'stats_verified'}))

    def test_verify_stats_fails_closed(self):
        from vkgr.data_audit import file_sha256
        with tempfile.TemporaryDirectory() as tmp:
            spec, fake, common = self._world(tmp)
            self.assertEqual(self._cli(fake, ['prepare-stats', *common, '--output', str(Path(tmp) / 'stats')])[0], 0)
            manifest = Path(tmp) / 'stats' / 'manifest.json'; sha = file_sha256(manifest)
            verify = lambda f, sha=sha: self._cli(f, ['verify-stats', *common, '--manifest', str(manifest), '--manifest-sha', sha])
            self.assertEqual(verify(fake)[0], 0)
            code, out, err = verify(fake, '0' * 64)
            self.assertEqual((code, out), (2, '')); self._one_error_line(err)
            drift = [(w, [dict(r, timespent=r['timespent'] + 1) if r['item_id'] == 10 and r['timespent'] < 255 else r for r in rows]) for w, rows in self._partitions()]
            _, drifted, _ = self._world(Path(tmp) / 'drift', partitions=drift)
            code, out, err = verify(drifted)
            self.assertEqual((code, out), (2, '')); self._one_error_line(err); self.assertIn('raw stats replay mismatch', err)

    def test_prepare_stats_refuses_bad_partitions_and_never_overwrites(self):
        r = self._row
        for label, parts in (('validation week', self._partitions() + [(25, [r(10, 3)])]), ('timespent 256', [(0, [r(10, 256), r(20, 4)]), (1, [r(30, 5)])]),
                             ('item without rows', [(0, [r(10, 3), r(20, 4)])])):
            with self.subTest(label), tempfile.TemporaryDirectory() as tmp:
                spec, fake, common = self._world(tmp, partitions=parts)
                code, out, err = self._cli(fake, ['prepare-stats', *common, '--output', str(Path(tmp) / 'stats')])
                self.assertEqual((code, out), (2, '')); self._one_error_line(err)
                self.assertFalse((Path(tmp) / 'stats').exists())
        with tempfile.TemporaryDirectory() as tmp:
            spec, fake, common = self._world(tmp)
            (Path(tmp) / 'stats').mkdir()
            code, out, err = self._cli(fake, ['prepare-stats', *common, '--output', str(Path(tmp) / 'stats')])
            self.assertEqual((code, out), (2, '')); self._one_error_line(err); self.assertEqual(list((Path(tmp) / 'stats').iterdir()), [])

    def test_stats_commands_reject_a_wrong_metadata_pin_or_a_bad_inputs_spec(self):
        from vkgr.data_audit import file_sha256
        with tempfile.TemporaryDirectory() as tmp:
            spec, fake, common = self._world(tmp)
            (Path(tmp) / 'bad.json').write_text(json.dumps({**spec, 'items_metadata_sha256': 'f' * 64}))
            bad = ['--inputs', str(Path(tmp) / 'bad.json'), '--inputs-sha', file_sha256(Path(tmp) / 'bad.json')]
            for argv in (['range-check', *bad], ['prepare-stats', *bad, '--output', str(Path(tmp) / 's1')]):
                code, out, err = self._cli(fake, argv)
                self.assertEqual((code, out), (2, '')); self._one_error_line(err)
            (Path(tmp) / 'schema.json').write_text(json.dumps({**spec, 'schema': 'p-feat04b-inputs-v0'}))
            code, out, err = self._cli(fake, ['range-check', '--inputs', str(Path(tmp) / 'schema.json'), '--inputs-sha', file_sha256(Path(tmp) / 'schema.json')])
            self.assertEqual((code, out), (2, '')); self._one_error_line(err)
            for missing in (['prepare-stats', *common], ['verify-stats', *common, '--manifest', 'm']):
                with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                    r4().main(missing)                                      # --output / --manifest-sha are required

    def test_stats_commands_do_not_read_the_queries(self):
        """The statistics never look at the validation queries that real_inputs loads: swapping them changes no output byte."""
        from vkgr.data_audit import file_sha256
        shas = []
        for queries in ([{'target': 10}], [{'target': 20}, {'target': 30}, {'target': 99}]):
            with tempfile.TemporaryDirectory() as tmp:
                spec, fake, common = self._world(tmp)
                swapped = lambda s, with_partitions=False, fake=fake, queries=queries: (lambda d, p, b: ({**d, 'queries': queries}, p, b))(*fake(s, with_partitions))
                self.assertEqual(self._cli(swapped, ['prepare-stats', *common, '--output', str(Path(tmp) / 'stats')])[0], 0)
                shas.append((file_sha256(Path(tmp) / 'stats' / 'stats.json'), json.loads((Path(tmp) / 'stats' / 'manifest.json').read_text())['stats_sha256']))
        self.assertEqual(shas[0], shas[1])


REAL_CONFIG_DIR = ROOT / 'configs' / 'feat04b'
REAL_VARIANTS = (('c-i4-popfresh', C), ('d-i5-quality', D), ('e-q2-behavior', E))
REAL_SEEDS = (42, 2024, 2025)


def real_config_files():
    """{(tag, seed): (path, feature)} for the nine real FEAT-04b run configs."""
    return {(tag, seed): (REAL_CONFIG_DIR / f'p-feat04b-{tag}-s{seed}.json', feature) for tag, feature in REAL_VARIANTS for seed in REAL_SEEDS}


def guard_real_configs_match_anchor():
    """AC10: the nine real configs exist; each differs from the anchor H1 config (php01, dim/knob removed) ONLY in schema, task_id, feature and seed;
    temperature is the float 0.1 and learning_rate the float 0.001; the 04a Q2 config differs from them in the same four keys only."""
    anchor = json.loads((ROOT / 'configs' / 'php01' / 'p-php01-h1-t010-s42.json').read_text())
    anchor = {k: v for k, v in anchor.items() if k not in ('dim', 'knob')}
    q2 = json.loads((ROOT / 'configs' / 'feat04a' / 'p-feat04a-a-q2t010-s42.json').read_text())
    m = r4()
    files = real_config_files()
    ensure(len(files) == 9, 'nine real configs expected')
    for (tag, seed), (path, feature) in files.items():
        ensure(path.is_file(), f'missing real config {path.name}')
        raw = path.read_text()
        value = json.loads(raw)
        ensure(raw == json.dumps(value, indent=1, sort_keys=True) + '\n', f'{path.name}: canonical json formatting (indent 1, sorted keys, final newline)')
        ensure(value['schema'] == 'p-feat04b-run-v1' and value['task_id'] == 'FEAT-04B', f'{path.name}: schema/task_id')
        ensure(value['feature'] == feature and value['seed'] == seed, f'{path.name}: feature/seed')
        ensure(type(value['temperature']) is float and value['temperature'] == 0.1, f'{path.name}: temperature must be float 0.1')
        ensure(type(value['learning_rate']) is float and value['learning_rate'] == 0.001, f'{path.name}: learning_rate must be float 0.001')
        varying = ('schema', 'task_id', 'feature', 'seed')
        ensure(set(value) == set(anchor) | {'feature', 'side_version'}, f'{path.name}: field set differs from the anchor config')
        for reference in (anchor, q2):
            same = {k: v for k, v in value.items() if k not in varying and k in reference}
            ensure(all(reference[k] == same[k] and type(reference[k]) is type(same[k]) for k in same if k not in ('feature', 'side_version')),
                   f'{path.name}: differs from the reference config outside schema/task_id/feature/seed')
        ensure(value['side_version'] == q2['side_version'], f'{path.name}: side_version')
        m.validate_config(value)
        for bad in ({'temperature': 0.15}, {'feature': CONTROL}, {'feature': A}, {'learning_rate': 0.0003}):
            ensure(rejects(lambda bad=bad, value=value: m.validate_config({**value, **bad})), f'{path.name}: real path must reject {bad}')
    ensure(len({p.name for p, _ in files.values()}) == 9, 'config names are unique')
    ensure(sorted(p.name for p in REAL_CONFIG_DIR.glob('p-feat04b-*.json')) == sorted(p.name for p, _ in files.values()), 'no stray p-feat04b config files')


class RealConfigFileTests(unittest.TestCase):
    def test_nine_real_configs_match_the_anchor(self):
        guard_real_configs_match_anchor()

    def test_seeds_are_the_three_registered_ones_per_variant(self):
        for tag, feature in REAL_VARIANTS:
            seeds = sorted(json.loads(p.read_text())['seed'] for (t, _), (p, f) in real_config_files().items() if t == tag)
            self.assertEqual(seeds, sorted(REAL_SEEDS))

    def test_a_config_with_another_seed_is_rejected_by_the_real_path(self):
        path, _ = real_config_files()[('c-i4-popfresh', 42)]
        with self.assertRaises(ValueError):
            r4().validate_config({**json.loads(path.read_text()), 'seed': 7})


if __name__ == '__main__':
    unittest.main()
