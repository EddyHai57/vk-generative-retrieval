"""GTOK-01R FEAT-001: repaired RQ-VAE fit path (dead-code reset, optimizer arms, read-only diagnostics).

Pure synthetic data, CPU only, no engineering contract, no hard-coded temp paths.
"""
import ast
import hashlib
import inspect
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from vkgr import sid_rqvae as rq
from vkgr import sid_rqvae_r as rr
from vkgr import sid_tok_audit as au

ROOT = Path(__file__).resolve().parents[1]

# sha256 of the frozen files at BASE 7a3b8157 (git show BASE:<path> | shasum -a 256).
FROZEN_SHA256 = {
    'src/vkgr/sid_rqvae.py': '52173bea9b80ffeb40e25943e3d7069ec53f9bd34a4509c97278194714065067',
    'src/vkgr/sid_tok_audit.py': 'd4f0afb554a6f48f787c91c81ffcb0baa7360a614b91152aa2d73cfd91b3f8ba',
    'src/vkgr/sid_v2.py': 'd094606ddab0460ef470e7b7cf86793af2c4352c8237842a803d6cd1c06470ad',
    'src/vkgr/sid_tok_run.py': '9ca27192fc2c82343e852d7df074a4f61a2d638cddc8d28cd6b56287bf6dd58a',
    'src/vkgr/generative_tok_run.py': '34aac1900e8dc0876d11e71f8ee11e1d59485cb3dd9d1a433b22903d94dc719a',
    'src/vkgr/generative_tok_data.py': '48498849f9c96fe4283e5a2e48e8ce3a1a1663ed47db2c9e981d737ebc73ee4c',
    'src/vkgr/generative_b1_run.py': 'd161aec55f74339129d556991532ec49ce9e520edc6425c7179398c4b5055af5',
    'src/vkgr/generative_data_v2.py': 'd90f4b4338faefb6a9c72649e965a8e7480c7566e2b60ec683b9a4282b9b8157',
    'src/vkgr/generative_b1_batched.py': 'd9f0ffba92fe57cc3dcd137b7083bdc92459561aa03d2bcacb4f5ebdb015bbb6',
    'src/vkgr/sid.py': '9602580f907cc2e8cfa3684aec7828df2bdfc87b59e71b6fd3865a9a19de91a3',
    'src/vkgr/sid_run.py': 'bea7121443d7dd7afb19a96454bceb571a4a58b184416a0e7b7af908c771bfed',
    'src/vkgr/tiger.py': '09ab37d79da8e46dabf522d8d73d6dbd88b5b77985375e459e4977896a18c352',
    'configs/gtok01/g-tok-01.json': 'b597c3192f7abcdeb136228349e857be15d55b4dba480539cdf694fe1f6924a3',
    'configs/gtok01/g-tok-t1-rqvae-b256-s42.json': 'd3f7de5c362c3fef4fb84f9d9d56e4cdbfe43aec5d7c21ccb27feac1d390cca2',
    'configs/gtok01/g-tok-t2-rqvae-b16-s42.json': '3dcfde7c22565dd812a097df30a9adb680f9bab46bd744949e3b3fd922fe14d9',
    'configs/gtok01/g-tok-t3-bhkm-b256-s42.json': '7c81651b7462e9277e57b34aff87514938012b7d8d6a11e85fdb621dbf09e570',
}

RESET = {'reset_check_every': 100, 'dead_window': 500, 'reset_until': 270000, 'rng_reset_offset': 7919}
EXPECTED_PREREG_R = {
    'arms': {
        'p0': {'optimizer': 'adagrad', 'lr': 0.4, 'initial_accumulator_value': 0.1, 'eps': 1e-7, 'reset': False},
        'l1': dict({'optimizer': 'adagrad', 'lr': 0.4, 'initial_accumulator_value': 0.1, 'eps': 1e-7,
                    'reset': True}, **RESET),
        'l2': dict({'optimizer': 'adagrad', 'lr': 0.1, 'initial_accumulator_value': 0.1, 'eps': 1e-7,
                    'reset': True}, **RESET),
        'l3': dict({'optimizer': 'adagrad', 'lr': 0.025, 'initial_accumulator_value': 0.1, 'eps': 1e-7,
                    'reset': True}, **RESET),
        'l4': dict({'optimizer': 'adam', 'lr': 1e-3, 'betas': (0.9, 0.999), 'eps': 1e-8, 'weight_decay': 0.0,
                    'reset': True}, **RESET),
    },
    'arm_order': ('l1', 'l2', 'l3', 'l4'),
    'reset': RESET,
    'probe_steps': 30000,
    'probe_wall_cap_seconds': 3000,
    'usage_diag_every': 2000,
    'diag_subset_rows': 8192,
    'diag_subset_seed': 42,
    'diag_fine_steps': (0, 1, 2, 3, 5, 10, 20, 50, 100, 200, 300, 500, 750, 1000, 1250, 1500, 1750, 2000,
                        2500, 3000, 4000, 5000),
    'diag_every_after': 1000,
}


def unit_rows(rng, n, d=64):
    x = rng.normal(size=(n, d)).astype(np.float32)
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def blobs(n_per=64, centres=8, d=64, seed=0, noise=0.05):
    rng = np.random.RandomState(seed)
    c = unit_rows(rng, centres, d)
    x = np.concatenate([c[i] + noise * rng.normal(size=(n_per, d)) for i in range(centres)]).astype(np.float32)
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x.astype(np.float32)


def clock_stepping(delta):
    state = {'t': -delta}

    def clock():
        state['t'] += delta
        return state['t']
    return clock


def sha256_of(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Prereg(unittest.TestCase):
    def test_prereg_r_arms_and_constants(self):
        self.assertEqual(rr.PREREG_R, EXPECTED_PREREG_R)
        self.assertEqual(list(rr.PREREG_R['arms']), ['p0', 'l1', 'l2', 'l3', 'l4'])
        for name in rr.PREREG_R['arms']:
            self.assertEqual(name, name.lower())

    def test_make_optimizer_types_and_all_hyperparameters(self):
        model = rq.RQVAE(4)
        for arm in ('p0', 'l1'):
            opt = rr.make_optimizer(model.parameters(), arm)
            self.assertIs(type(opt), torch.optim.Adagrad)
            group = opt.param_groups[0]
            self.assertEqual((group['lr'], group['initial_accumulator_value'], group['eps'], group['weight_decay']),
                             (0.4, 0.1, 1e-7, 0))
        for arm, lr in (('l2', 0.1), ('l3', 0.025)):
            opt = rr.make_optimizer(model.parameters(), arm)
            self.assertIs(type(opt), torch.optim.Adagrad)
            group = opt.param_groups[0]
            self.assertEqual((group['lr'], group['initial_accumulator_value'], group['eps'], group['weight_decay']),
                             (lr, 0.1, 1e-7, 0))
        opt = rr.make_optimizer(model.parameters(), 'l4')
        self.assertIs(type(opt), torch.optim.Adam)
        group = opt.param_groups[0]
        self.assertEqual((group['lr'], tuple(group['betas']), group['eps'], group['weight_decay'], group['amsgrad']),
                         (1e-3, (0.9, 0.999), 1e-8, 0.0, False))
        covered = {id(p) for g in opt.param_groups for p in g['params']}
        self.assertEqual(covered, {id(p) for p in model.parameters()})
        for bad in ('L1', 'P0', 'l5', '', None):
            with self.subTest(arm=bad):
                with self.assertRaises(ValueError):
                    rr.make_optimizer(model.parameters(), bad)

    def test_p0_optimizer_matches_the_old_fit_optimizer_exactly(self):
        model = rq.RQVAE(4)
        old = torch.optim.Adagrad(model.parameters(), lr=rq.PREREG['lr'],
                                  initial_accumulator_value=rq.PREREG['initial_accumulator_value'],
                                  eps=rq.PREREG['eps'])
        new = rr.make_optimizer(model.parameters(), 'p0')
        self.assertEqual(old.defaults, new.defaults)


class FrozenFiles(unittest.TestCase):
    def test_frozen_files_are_byte_identical_to_base(self):
        for path, expected in FROZEN_SHA256.items():
            with self.subTest(path=path):
                self.assertEqual(sha256_of(ROOT / path), expected)

    def test_gate_constants_unchanged(self):
        self.assertEqual(au.USAGE_NUM_DEN, (4, 5))
        self.assertEqual(au.REQUIRED_STEPS, 300000)
        self.assertEqual(rq.PREREG['usage_gate_num_den'], (4, 5))
        self.assertEqual(rq.PREREG['max_steps'], 300000)
        self.assertEqual(rq.PREREG['wall_cap_seconds'], 7200)


class LastUsed(unittest.TestCase):
    def test_new_last_used_is_zero_int64_per_layer(self):
        model = rq.RQVAE(8)
        table = rr.new_last_used(model)
        self.assertEqual(len(table), 3)
        for tensor in table:
            self.assertEqual(tensor.dtype, torch.int64)
            self.assertEqual(tuple(tensor.shape), (8,))
            self.assertEqual(tensor.tolist(), [0] * 8)
            self.assertEqual(tensor.device, model.codebooks[0].device)

    def test_touch_matches_reference_set_of_selected_codes_with_duplicates(self):
        rng = np.random.RandomState(5)
        K = 16
        table = [torch.zeros(K, dtype=torch.int64) for _ in range(3)]
        reference = [np.zeros(K, np.int64) for _ in range(3)]
        for step in (3, 7, 12):
            codes = [torch.from_numpy(rng.randint(0, 4 + 3 * layer, size=256)) for layer in range(3)]
            rr.touch_last_used(table, codes, step)
            for layer in range(3):
                selected = np.bincount(codes[layer].numpy(), minlength=K) > 0
                reference[layer][selected] = step
                self.assertEqual(table[layer].tolist(), reference[layer].tolist(), (step, layer))
                self.assertEqual(table[layer].dtype, torch.int64)

    def test_touch_updates_all_three_layers_with_the_current_step(self):
        table = [torch.zeros(4, dtype=torch.int64) for _ in range(3)]
        rr.touch_last_used(table, [torch.tensor([1]), torch.tensor([2]), torch.tensor([3, 3])], 9)
        self.assertEqual([t.tolist() for t in table], [[0, 9, 0, 0], [0, 0, 9, 0], [0, 0, 0, 9]])


def make_world(arm='l1', K=8, rows=16, seed=0):
    torch.manual_seed(seed)
    model = rq.RQVAE(K)
    optimizer = rr.make_optimizer(model.parameters(), arm)
    generator = torch.Generator().manual_seed(100 + seed)
    # four distinct residual tensors (z, r1, r2, r3) that still carry autograd history like real ones
    residuals = [torch.randn(rows, 32, generator=generator).add(10.0 * i).requires_grad_(True) for i in range(4)]
    return model, optimizer, residuals, rr.new_last_used(model)


def rng_state_equal(a, b):
    sa, sb = a.get_state(), b.get_state()
    return np.array_equal(sa[1], sb[1]) and sa[2:] == sb[2:]


class ResetDeadCodes(unittest.TestCase):
    def test_boundary_table_per_layer_sources_and_rng_discipline(self):
        model, optimizer, residuals, last_used = make_world()
        cfg = rr.PREREG_R['arms']['l1']
        # layer 0: gap 0 / 499 / 500 / 501 / 1000 / 100 / 0 / 0 at step 1000 -> codes 2, 3, 4 dead
        last_used[0] = torch.tensor([1000, 501, 500, 499, 0, 900, 1000, 1000])
        last_used[1] = torch.full((8,), 1000)  # nothing dead: rng must not move for this layer
        last_used[2] = torch.tensor([0, 1000, 1000, 1000, 1000, 1000, 1000, 0])  # codes 0 and 7 dead
        before = [c.detach().clone() for c in model.codebooks]
        rng = np.random.RandomState(42 + 7919)
        log = rr.reset_dead_codes(model, optimizer, residuals, last_used, 1000, rng, cfg)
        self.assertEqual(log, [{'step': 1000, 'layer': 0, 'n_reset': 3}, {'step': 1000, 'layer': 2, 'n_reset': 2}])
        reference = np.random.RandomState(42 + 7919)
        rows0 = reference.choice(16, 3, replace=False)
        rows2 = reference.choice(16, 2, replace=False)  # layer 1 had no dead code -> no draw
        self.assertTrue(rng_state_equal(rng, reference))
        self.assertFalse(torch.equal(residuals[0].detach(), residuals[2].detach()))
        c0, c1, c2 = (c.detach() for c in model.codebooks)
        source0, source2 = residuals[0].detach(), residuals[2].detach()
        for code, row in zip((2, 3, 4), rows0):
            self.assertTrue(torch.equal(c0[code], source0[row]), code)
        for code, row in zip((0, 7), rows2):
            self.assertTrue(torch.equal(c2[code], source2[row]), code)
        for code in (0, 1, 5, 6, 7):
            self.assertTrue(torch.equal(c0[code], before[0][code]), code)
        self.assertTrue(torch.equal(c1, before[1]))
        for code in range(1, 7):
            self.assertTrue(torch.equal(c2[code], before[2][code]), code)
        self.assertEqual(last_used[0].tolist(), [1000, 501, 1000, 1000, 1000, 900, 1000, 1000])
        self.assertEqual(last_used[1].tolist(), [1000] * 8)
        self.assertEqual(last_used[2].tolist(), [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000])
        # next checkpoint: the codes reset above are not reset again, only code 1 (gap 599) is dead
        log2 = rr.reset_dead_codes(model, optimizer, residuals, last_used, 1100, rng, cfg)
        self.assertEqual(log2, [{'step': 1100, 'layer': 0, 'n_reset': 1}])
        row = reference.choice(16, 1, replace=False)[0]
        self.assertTrue(torch.equal(model.codebooks[0].detach()[1], source0[row]))
        self.assertEqual(last_used[0][1].item(), 1100)

    def test_n_dead_zero_never_advances_the_reset_stream(self):
        model, optimizer, residuals, last_used = make_world()
        cfg = rr.PREREG_R['arms']['l1']
        for table in last_used:
            table.fill_(1000)
        rng = np.random.RandomState(42 + 7919)
        fresh = np.random.RandomState(42 + 7919)
        before = [c.detach().clone() for c in model.codebooks]
        self.assertEqual(rr.reset_dead_codes(model, optimizer, residuals, last_used, 1000, rng, cfg), [])
        self.assertTrue(rng_state_equal(rng, fresh))
        for got, want in zip(model.codebooks, before):
            self.assertTrue(torch.equal(got.detach(), want))
        self.assertEqual(rng.choice(16, 3, replace=False).tolist(), fresh.choice(16, 3, replace=False).tolist())

    def test_checkpoint_schedule(self):
        cfg = rr.PREREG_R['arms']['l1']
        for step, stale, checked in ((400, -10 ** 6, False), (500, 0, True), (550, -10 ** 6, False),
                                     (270000, 0, True), (270100, -10 ** 6, False), (499, -10 ** 6, False)):
            with self.subTest(step=step):
                model, optimizer, residuals, last_used = make_world()
                for table in last_used:
                    table.fill_(stale)  # artificial table: everything would be dead if the step were checked
                before = [c.detach().clone() for c in model.codebooks]
                rng = np.random.RandomState(1)
                fresh = np.random.RandomState(1)
                log = rr.reset_dead_codes(model, optimizer, residuals, last_used, step, rng, cfg)
                if checked:
                    self.assertEqual(log, [{'step': step, 'layer': layer, 'n_reset': 8} for layer in range(3)])
                    for table in last_used:
                        self.assertEqual(table.tolist(), [step] * 8)
                else:
                    self.assertEqual(log, [])
                    self.assertTrue(rng_state_equal(rng, fresh))
                    for got, want in zip(model.codebooks, before):
                        self.assertTrue(torch.equal(got.detach(), want))
                    for table in last_used:
                        self.assertEqual(table.tolist(), [stale] * 8)

    def test_reset_due_follows_the_given_cadence_and_window(self):
        cfg = {'reset_check_every': 10, 'dead_window': 20, 'reset_until': 60}
        for step, expected in ((10, False), (20, True), (25, False), (30, True), (60, True), (70, False), (100, False)):
            with self.subTest(step=step):
                self.assertIs(rr.reset_due(step, cfg), expected)

    def test_optimizer_rows_are_reset_for_adagrad(self):
        model, optimizer, residuals, last_used = make_world('l1')
        cfg = rr.PREREG_R['arms']['l1']
        for param in model.codebooks:
            optimizer.state[param]['sum'].fill_(5.0)
        encoder_state = optimizer.state[next(model.encoder.parameters())]['sum'].clone()
        last_used[0] = torch.tensor([1000, 1000, 0, 1000, 1000, 0, 1000, 1000])
        last_used[1].fill_(1000)
        last_used[2].fill_(1000)
        rr.reset_dead_codes(model, optimizer, residuals, last_used, 1000, np.random.RandomState(3), cfg)
        state = optimizer.state[model.codebooks[0]]['sum']
        for code in range(8):
            expected = 0.1 if code in (2, 5) else 5.0
            self.assertTrue(bool((state[code] == expected).all()), code)
        for layer in (1, 2):
            self.assertTrue(bool((optimizer.state[model.codebooks[layer]]['sum'] == 5.0).all()))
        self.assertTrue(torch.equal(optimizer.state[next(model.encoder.parameters())]['sum'], encoder_state))

    def test_optimizer_rows_are_reset_for_adam(self):
        model, optimizer, residuals, last_used = make_world('l4')
        cfg = rr.PREREG_R['arms']['l4']
        optimizer.zero_grad()
        sum((c ** 2).sum() for c in model.codebooks).backward()
        optimizer.step()
        for param in model.codebooks:
            optimizer.state[param]['exp_avg'].fill_(3.0)
            optimizer.state[param]['exp_avg_sq'].fill_(4.0)
        steps = [optimizer.state[param]['step'].clone() for param in model.codebooks]
        last_used[0] = torch.tensor([1000, 0, 1000, 1000, 1000, 1000, 0, 1000])
        last_used[1].fill_(1000)
        last_used[2].fill_(1000)
        rr.reset_dead_codes(model, optimizer, residuals, last_used, 1000, np.random.RandomState(3), cfg)
        state = optimizer.state[model.codebooks[0]]
        for code in range(8):
            dead = code in (1, 6)
            self.assertTrue(bool((state['exp_avg'][code] == (0.0 if dead else 3.0)).all()), code)
            self.assertTrue(bool((state['exp_avg_sq'][code] == (0.0 if dead else 4.0)).all()), code)
        for layer, step in zip(range(3), steps):
            self.assertTrue(torch.equal(optimizer.state[model.codebooks[layer]]['step'], step))
        for layer in (1, 2):
            self.assertTrue(bool((optimizer.state[model.codebooks[layer]]['exp_avg'] == 3.0).all()))

    def test_reset_source_must_be_detached(self):
        model, optimizer, residuals, last_used = make_world()
        cfg = rr.PREREG_R['arms']['l1']
        for table in last_used:
            table.zero_()
        self.assertTrue(residuals[0].requires_grad)
        with mock.patch.object(rr, '_layer_source', lambda r, layer: r[layer]):
            with self.assertRaises(AssertionError):
                rr.reset_dead_codes(model, optimizer, residuals, last_used, 500, np.random.RandomState(1), cfg)

    def test_more_dead_codes_than_batch_rows_asserts(self):
        model, optimizer, _, last_used = make_world()
        cfg = rr.PREREG_R['arms']['l1']
        small = [torch.randn(4, 32) for _ in range(4)]
        for table in last_used:
            table.zero_()
        with self.assertRaises(AssertionError):
            rr.reset_dead_codes(model, optimizer, small, last_used, 500, np.random.RandomState(1), cfg)


def pair_data(K):
    if K == 256:
        return unit_rows(np.random.RandomState(21), 1024), 256
    return unit_rows(np.random.RandomState(22), 512), 64


class P0Equivalence(unittest.TestCase):
    """fit_r('p0', diagnostics=False) must reproduce rq.fit bit for bit."""

    def assert_pair(self, K, max_steps, eval_every, **extra):
        X, batch = pair_data(K)
        kw = dict(K=K, seed=7, max_steps=max_steps, batch_size=batch, device='cpu', wall_cap_seconds=1e9,
                  eval_every=eval_every, **extra)
        old = rq.fit(X, clock=clock_stepping(0.01), **kw)
        new = rr.fit_r(X, arm='p0', diagnostics=False, clock=clock_stepping(0.01), **kw)
        self.assertEqual(old['status'], 'complete')
        self.assertEqual(new['status'], 'complete')
        for key in ('status', 'step', 'elapsed', 'K', 'seed', 'init_digest', 'prefix_digest', 'loss_log',
                    'usage_log'):
            self.assertEqual(new[key], old[key], key)
        self.assertTrue(len(old['usage_log']) >= 1)
        self.assertIsNotNone(old['prefix_digest'])
        self.assertEqual(rq.state_digest(new['state']['state_dict']), rq.state_digest(old['state']['state_dict']))
        for name, tensor in old['state']['state_dict'].items():
            self.assertTrue(torch.equal(new['state']['state_dict'][name], tensor), name)
        self.assertEqual(new['reset_log'], [])
        self.assertEqual(new['diagnostics'], [])

    def test_p0_matches_fit_past_the_first_possible_reset_step_k256(self):
        self.assert_pair(256, 600, 200, probe_from=100, probe_step=300, log_every=100)

    def test_p0_matches_fit_past_the_first_possible_reset_step_k16(self):
        self.assert_pair(16, 600, 200, probe_from=100, probe_step=300, log_every=100)

    def test_p0_matches_fit_k16_eval_every_2000(self):
        self.assert_pair(16, 4000, 2000)

    def test_p0_matches_fit_k256_eval_every_2000(self):
        self.assert_pair(256, 4000, 2000)

    def test_p0_matches_fit_k16_eval_every_10000(self):
        self.assert_pair(16, 10100, 10000)

    def test_p0_matches_fit_k256_eval_every_10000(self):
        self.assert_pair(256, 10100, 10000)

    def test_failure_statuses_match_fit_and_carry_no_state(self):
        X = blobs(n_per=32, centres=8, d=64, seed=6)
        kw = dict(K=4, seed=1, batch_size=64, device='cpu')
        scenarios = {
            'fail_wall': dict(max_steps=5000, wall_cap_seconds=7200, clock=lambda: clock_stepping(1000.0)),
            'fail_wall_projected': dict(max_steps=300000, wall_cap_seconds=7200, probe_from=100, probe_step=300,
                                        clock=lambda: clock_stepping(3.0)),
        }
        for name, scenario in scenarios.items():
            with self.subTest(status=name):
                args = dict(scenario)
                make_clock = args.pop('clock')
                old = rq.fit(X, clock=make_clock(), **args, **kw)
                new = rr.fit_r(X, arm='p0', diagnostics=False, clock=make_clock(), **args, **kw)
                self.assertEqual(old['status'], name)
                for key in ('status', 'step', 'elapsed', 'init_digest', 'loss_log', 'usage_log'):
                    self.assertEqual(new[key], old[key], key)
                self.assertNotIn('state', old)
                self.assertNotIn('state', new)
        with self.subTest(status='fail_nonfinite'):
            outcomes = []
            for runner in (lambda: rq.fit(X, max_steps=1000, wall_cap_seconds=1e9, **kw),
                           lambda: rr.fit_r(X, arm='p0', diagnostics=False, max_steps=1000, wall_cap_seconds=1e9,
                                            **kw)):
                real = rq.loss_terms
                calls = {'n': 0}

                def poisoned(*a, **k):
                    t = real(*a, **k)
                    calls['n'] += 1
                    if calls['n'] == 120:
                        t = dict(t)
                        t['total'] = t['total'] * float('nan')
                    return t
                with mock.patch.object(rq, 'loss_terms', poisoned):
                    outcomes.append(runner())
            old, new = outcomes
            self.assertEqual(old['status'], 'fail_nonfinite')
            for key in ('status', 'step', 'init_digest', 'loss_log', 'usage_log'):
                self.assertEqual(new[key], old[key], key)
            self.assertNotIn('state', new)

    def test_optimizer_is_created_after_codebook_initialisation(self):
        X = blobs(n_per=32, centres=8, d=64, seed=6)
        events = []
        real_init, real_make = rq.init_codebooks, rr.make_optimizer

        def init_spy(*a, **k):
            events.append('init_codebooks')
            return real_init(*a, **k)

        def make_spy(*a, **k):
            events.append('make_optimizer')
            return real_make(*a, **k)
        with mock.patch.object(rq, 'init_codebooks', init_spy), mock.patch.object(rr, 'make_optimizer', make_spy):
            rr.fit_r(X, arm='p0', K=4, seed=1, max_steps=2, batch_size=64, device='cpu', wall_cap_seconds=1e9,
                     probe_from=100, probe_step=200, diagnostics=False)
        self.assertEqual(events, ['init_codebooks', 'make_optimizer'])

    def test_p0_never_touches_the_reset_machinery(self):
        X = blobs(n_per=32, centres=8, d=64, seed=6)
        calls = []

        def spy(name):
            def inner(*a, **k):
                calls.append(name)
                return []
            return inner
        with mock.patch.object(rr, 'new_last_used', spy('new')), mock.patch.object(rr, 'touch_last_used', spy('touch')), \
                mock.patch.object(rr, 'reset_dead_codes', spy('reset')):
            result = rr.fit_r(X, arm='p0', K=4, seed=1, max_steps=30, batch_size=64, device='cpu',
                              wall_cap_seconds=1e9, probe_from=100, probe_step=200, diagnostics=False)
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(calls, [])

    def test_invalid_arguments_are_rejected_like_fit(self):
        X = blobs(n_per=32, centres=8, d=64, seed=6)
        kw = dict(K=4, max_steps=5, batch_size=64, device='cpu', wall_cap_seconds=1e9)
        for arm in ('P0', 'l5', None):
            with self.subTest(arm=arm):
                with self.assertRaises(ValueError):
                    rr.fit_r(X, arm=arm, **kw)
        with self.assertRaises(ValueError):
            rr.fit_r(X.astype(np.float64), arm='p0', **kw)
        with self.assertRaises(ValueError):
            rr.fit_r(X, arm='p0', probe_from=150, probe_step=300, **kw)


def constant_model(K=3):
    """Hand-computable model: encoder outputs the constant z, decoder outputs the constant 1."""
    torch.manual_seed(0)
    model = rq.RQVAE(K)
    z = torch.full((32,), 0.5)
    row0 = z.clone()
    row0[0] += 0.25
    with torch.no_grad():
        for layer in model.encoder:
            if isinstance(layer, torch.nn.Linear):
                layer.weight.zero_()
                layer.bias.zero_()
        model.encoder[-1].bias.copy_(z)
        for layer in model.decoder:
            if isinstance(layer, torch.nn.Linear):
                layer.weight.zero_()
                layer.bias.fill_(1.0)
        # level 0: row 0 is 0.25 away from z; level 1: row 0 equals the leftover; level 2: row 1 is the zero vector
        model.codebooks[0].copy_(torch.stack([row0, z + 1.0, z - 1.0]))
        left = z - row0
        model.codebooks[1].copy_(torch.stack([left, left + 1.0, left - 1.0]))
        model.codebooks[2].copy_(torch.stack([torch.ones(32), torch.zeros(32), -torch.ones(32)]))
    return model


class Diagnostics(unittest.TestCase):
    def test_diagnose_hand_computed_on_a_constant_model(self):
        model = constant_model()
        x = torch.from_numpy(unit_rows(np.random.RandomState(3), 8))
        record = rr.diagnose(model, x, 7)
        self.assertEqual(record['step'], 7)
        recon = float(((x.double() - 1.0) ** 2).sum(1).mean())
        self.assertAlmostEqual(record['recon'], recon, places=5)
        self.assertAlmostEqual(record['codebook'], 0.0625, places=6)
        self.assertAlmostEqual(record['commit'], 0.25 * 0.0625, places=6)
        self.assertAlmostEqual(record['z_std_mean'], 0.0, places=7)
        self.assertEqual(record['dead_relu'], [1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        self.assertEqual(record['usage_sub'], [1, 1, 1])
        for got, want in zip(record['perplexity'], (1.0, 1.0, 1.0)):
            self.assertAlmostEqual(got, want, places=6)
        for got, want in zip(record['nearest_dist'], (0.25, 0.0, 0.0)):
            self.assertAlmostEqual(got, want, places=6)
        self.assertIsNone(record['grad_norm'])
        self.assertIsNone(record['rel_update'])

    def test_diagnose_statistics_on_a_random_model_match_numpy(self):
        torch.manual_seed(4)
        model = rq.RQVAE(16)
        x = torch.from_numpy(unit_rows(np.random.RandomState(8), 200))
        with torch.no_grad():
            z = model.encoder(x).numpy()
            q = rq.quantize(torch.from_numpy(z), list(model.codebooks))
        record = rr.diagnose(model, x, 1)
        self.assertAlmostEqual(record['z_std_mean'], float(np.std(z.astype(np.float64), axis=0).mean()), places=5)
        for layer in range(3):
            codes = q['codes'][layer].numpy()
            counts = np.bincount(codes, minlength=16)
            self.assertEqual(record['usage_sub'][layer], int((counts > 0).sum()))
            p = counts[counts > 0] / counts.sum()
            self.assertAlmostEqual(record['perplexity'][layer], float(np.exp(-(p * np.log(p)).sum())), places=5)
            distance = np.linalg.norm((q['residuals'][layer] - q['embeds'][layer]).numpy(), axis=1).mean()
            self.assertAlmostEqual(record['nearest_dist'][layer], float(distance), places=5)
        self.assertEqual(len(record['dead_relu']), 6)
        self.assertTrue(all(0.0 <= v <= 1.0 for v in record['dead_relu']))

    def test_diag_due_schedule(self):
        fine = EXPECTED_PREREG_R['diag_fine_steps']
        for step in fine:
            self.assertTrue(rr.diag_due(step), step)
        for step in (4, 6, 7, 49, 51, 499, 501, 1001, 2750, 3500, 5001, 5500, 6001, 6999):
            self.assertFalse(rr.diag_due(step), step)
        for step in (6000, 7000, 100000, 300000):
            self.assertTrue(rr.diag_due(step), step)

    def test_diag_subset_uses_its_own_generator(self):
        np.random.seed(123)
        state = np.random.get_state()
        for n in (50, 8192, 10000):
            rows = rr.diag_subset_rows(n)
            want = np.random.RandomState(42).choice(n, min(n, 8192), replace=False)
            self.assertEqual(rows.tolist(), want.tolist())
            self.assertEqual(len(set(rows.tolist())), len(rows))
        after = np.random.get_state()
        self.assertTrue(np.array_equal(after[1], state[1]) and after[2:] == state[2:])

    def _small_run(self, arm, diagnostics, **extra):
        X = blobs(n_per=32, centres=8, d=64, seed=6)
        kw = dict(K=16, seed=11, max_steps=120, batch_size=64, device='cpu', wall_cap_seconds=1e9,
                  eval_every=50, log_every=50, probe_from=100, probe_step=200)
        kw.update(extra)
        return rr.fit_r(X, arm=arm, diagnostics=diagnostics, clock=clock_stepping(0.01), **kw)

    def test_diagnostics_do_not_change_training_p0(self):
        on = self._small_run('p0', True)
        off = self._small_run('p0', False)
        self.assertEqual(rq.state_digest(on['state']['state_dict']), rq.state_digest(off['state']['state_dict']))
        for key in ('loss_log', 'usage_log', 'recon_log', 'init_digest'):
            self.assertEqual(on[key], off[key], key)
        self.assertGreater(len(on['diagnostics']), 0)
        self.assertEqual(off['diagnostics'], [])

    def test_diagnostics_do_not_change_training_l1_with_resets(self):
        small = {'reset_check_every': 10, 'dead_window': 20, 'reset_until': 100}
        with mock.patch.dict(rr.PREREG_R['arms']['l1'], small):
            on = self._small_run('l1', True, K=64)
            off = self._small_run('l1', False, K=64)
        self.assertGreater(len(on['reset_log']), 0)
        self.assertEqual(on['reset_log'], off['reset_log'])
        self.assertEqual(rq.state_digest(on['state']['state_dict']), rq.state_digest(off['state']['state_dict']))
        self.assertEqual(on['loss_log'], off['loss_log'])

    def test_diagnostics_leave_global_random_state_alone(self):
        np.random.seed(123)
        np_before = np.random.get_state()
        self._small_run('p0', True)
        np_after = np.random.get_state()
        self.assertTrue(np.array_equal(np_before[1], np_after[1]) and np_before[2:] == np_after[2:])
        # both runs start from torch.manual_seed(seed): the generator must end in the same state with or without
        # diagnostics, so the diagnostics consumed no torch random numbers
        self._small_run('p0', True)
        torch_on = torch.get_rng_state().clone()
        self._small_run('p0', False)
        torch_off = torch.get_rng_state().clone()
        self.assertTrue(torch.equal(torch_on, torch_off))

    def test_record_steps_fields_and_gradient_statistics(self):
        result = self._small_run('p0', True)
        records = result['diagnostics']
        self.assertEqual([r['step'] for r in records], [0, 1, 2, 3, 5, 10, 20, 50, 100])
        keys = {'step', 'recon', 'codebook', 'commit', 'z_std_mean', 'dead_relu', 'usage_sub', 'perplexity',
                'nearest_dist', 'grad_norm', 'rel_update'}
        for record in records:
            self.assertEqual(set(record), keys)
            self.assertEqual(len(record['dead_relu']), 6)
            self.assertEqual(len(record['usage_sub']), 3)
            self.assertEqual(len(record['perplexity']), 3)
            self.assertEqual(len(record['nearest_dist']), 3)
            for name in ('recon', 'codebook', 'commit', 'z_std_mean'):
                self.assertTrue(np.isfinite(record[name]), (record['step'], name))
        self.assertIsNone(records[0]['grad_norm'])
        self.assertIsNone(records[0]['rel_update'])
        for record in records[1:]:
            self.assertEqual(set(record['grad_norm']), {'encoder', 'decoder', 'codebooks'})
            self.assertEqual(set(record['rel_update']), {'encoder', 'decoder', 'codebooks'})
            for group in ('encoder', 'decoder', 'codebooks'):
                self.assertGreater(record['grad_norm'][group], 0.0, (record['step'], group))
                self.assertGreater(record['rel_update'][group], 0.0, (record['step'], group))

    def test_step_one_gradient_norm_and_update_match_a_manual_step(self):
        X = blobs(n_per=32, centres=8, d=64, seed=6)
        kw = dict(K=4, seed=5, batch_size=64, device='cpu', wall_cap_seconds=1e9, probe_from=100, probe_step=200)
        record = rr.fit_r(X, arm='p0', max_steps=1, diagnostics=True, **kw)['diagnostics'][1]
        # replay step 1 by hand with the same seed / order / initialisation
        torch.manual_seed(5)
        model = rq.RQVAE(4)
        order = np.random.RandomState(5).permutation(len(X))
        Xt = torch.from_numpy(X)
        rq.init_codebooks(model, rq._first_batch(Xt, order, 64), 5, rq.PREREG['kmeans_iters'])
        optimizer = rr.make_optimizer(model.parameters(), 'p0')
        batch = Xt[torch.from_numpy(order[:64])]
        out = model(batch)
        terms = rq.loss_terms(batch, out['xhat'], out['q'])
        optimizer.zero_grad(set_to_none=True)
        terms['total'].backward()
        for name, params in (('encoder', model.encoder.parameters()), ('decoder', model.decoder.parameters()),
                             ('codebooks', model.codebooks.parameters())):
            norm = float(torch.sqrt(sum((p.grad.double() ** 2).sum() for p in params)))
            self.assertAlmostEqual(record['grad_norm'][name], norm, places=4, msg=name)
        before = {n: torch.cat([p.detach().reshape(-1) for p in ps]).clone() for n, ps in
                  (('encoder', model.encoder.parameters()), ('decoder', model.decoder.parameters()),
                   ('codebooks', model.codebooks.parameters()))}
        optimizer.step()
        for name, ps in (('encoder', model.encoder.parameters()), ('decoder', model.decoder.parameters()),
                         ('codebooks', model.codebooks.parameters())):
            after = torch.cat([p.detach().reshape(-1) for p in ps])
            want = float((after - before[name]).norm() / before[name].norm())
            self.assertAlmostEqual(record['rel_update'][name], want, places=5, msg=name)


class LArms(unittest.TestCase):
    SMALL = {'reset_check_every': 10, 'dead_window': 20, 'reset_until': 60}

    def run_l1(self, **extra):
        X = unit_rows(np.random.RandomState(31), 1024)
        kw = dict(K=64, seed=3, max_steps=100, batch_size=64, device='cpu', wall_cap_seconds=1e9,
                  eval_every=50, log_every=50, check_every=50, probe_from=50, probe_step=100)
        kw.update(extra)
        with mock.patch.dict(rr.PREREG_R['arms']['l1'], self.SMALL):
            return rr.fit_r(X, arm='l1', diagnostics=False, clock=clock_stepping(0.01), **kw)

    def test_resets_follow_the_small_window_and_reproduce(self):
        a = self.run_l1()
        b = self.run_l1()
        self.assertEqual(a['status'], 'complete')
        self.assertGreater(len(a['reset_log']), 0)
        self.assertEqual(a['reset_log'], b['reset_log'])
        self.assertEqual(rq.state_digest(a['state']['state_dict']), rq.state_digest(b['state']['state_dict']))
        steps = [e['step'] for e in a['reset_log']]
        for entry in a['reset_log']:
            self.assertEqual(set(entry), {'step', 'layer', 'n_reset'})
            self.assertEqual(entry['step'] % 10, 0)
            self.assertTrue(20 <= entry['step'] <= 60, entry)
            self.assertIn(entry['layer'], (0, 1, 2))
            self.assertTrue(1 <= entry['n_reset'] <= 64)
        self.assertTrue(any(s % 50 != 0 for s in steps), 'resets must follow reset_check_every, not check_every')
        self.assertEqual(steps, sorted(steps))

    def test_reset_stream_internal_state_and_call_order(self):
        events, first_rng_state, pre_checks = [], [], []
        real_reset, real_touch, real_make = rr.reset_dead_codes, rr.touch_last_used, rr.make_optimizer
        touched = {}
        post = {'table': None}

        def make_spy(*a, **k):
            optimizer = real_make(*a, **k)
            real_step = optimizer.step

            def step(*sa, **sk):
                events.append('step')
                return real_step(*sa, **sk)
            optimizer.step = step
            return optimizer

        def touch_spy(last_used, codes, step):
            touched['codes'] = [c.clone() for c in codes]
            touched['step'] = step
            return real_touch(last_used, codes, step)

        def reset_spy(model, optimizer, residuals, last_used, step, rng_reset, cfg):
            events.append('reset')
            if post['table'] is None:
                post['table'] = [torch.zeros_like(t) for t in last_used]
            expected = [t.clone() for t in post['table']]
            for table, codes in zip(expected, touched['codes']):
                table[codes] = step
            self.assertEqual(touched['step'], step)
            for got, want in zip(last_used, expected):
                pre_checks.append(torch.equal(got, want))
            if not first_rng_state:
                first_rng_state.append(rng_reset.get_state())
            for layer in range(3):
                self.assertFalse(rr._layer_source(residuals, layer).requires_grad)
            log = real_reset(model, optimizer, residuals, last_used, step, rng_reset, cfg)
            post['table'] = [t.clone() for t in last_used]
            return log
        with mock.patch.object(rr, 'make_optimizer', make_spy), mock.patch.object(rr, 'touch_last_used', touch_spy), \
                mock.patch.object(rr, 'reset_dead_codes', reset_spy):
            result = self.run_l1(max_steps=60, probe_from=20, probe_step=40, check_every=20)
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(events, ['step', 'reset'] * 60)
        self.assertEqual(len(pre_checks), 180)
        self.assertTrue(all(pre_checks), 'last_used must record every code selected at every step, all 3 layers')
        fresh = np.random.RandomState(3 + 7919).get_state()
        self.assertTrue(np.array_equal(first_rng_state[0][1], fresh[1]) and first_rng_state[0][2:] == fresh[2:])

    def test_no_reset_after_reset_until(self):
        result = self.run_l1(max_steps=100)
        self.assertTrue(all(e['step'] <= 60 for e in result['reset_log']))
        with mock.patch.dict(rr.PREREG_R['arms']['l1'], dict(self.SMALL, reset_until=100)):
            X = unit_rows(np.random.RandomState(31), 1024)
            later = rr.fit_r(X, arm='l1', diagnostics=False, K=64, seed=3, max_steps=100, batch_size=64,
                             device='cpu', wall_cap_seconds=1e9, probe_from=50, probe_step=100, check_every=50)
        self.assertTrue(any(e['step'] > 60 for e in later['reset_log']), 'window extended: later resets exist')

    def test_project_flag_only_disables_the_extrapolation(self):
        X = blobs(n_per=32, centres=8, d=64, seed=6)
        kw = dict(K=4, seed=1, batch_size=64, device='cpu', max_steps=2000, wall_cap_seconds=30,
                  probe_from=100, probe_step=300, diagnostics=False)
        projected = rr.fit_r(X, arm='p0', clock=clock_stepping(3.0), **kw)
        self.assertEqual((projected['status'], projected['step']), ('fail_wall_projected', 300))
        self.assertNotIn('state', projected)
        unprojected = rr.fit_r(X, arm='p0', project=False, clock=clock_stepping(3.0), **kw)
        self.assertEqual((unprojected['status'], unprojected['step']), ('fail_wall', 1100))
        self.assertNotIn('state', unprojected)
        explicit = rr.fit_r(X, arm='p0', project=True, clock=clock_stepping(3.0), **kw)
        self.assertEqual(explicit['status'], 'fail_wall_projected')
        ok = rr.fit_r(X, arm='p0', project=False, clock=clock_stepping(0.0001), **kw)
        self.assertEqual(ok['status'], 'complete')
        self.assertIsNotNone(ok['prefix_digest'])

    def test_recon_log_matches_window_means_and_loss_log(self):
        X = blobs(n_per=32, centres=8, d=64, seed=6)
        totals, recons = [], []
        real = rq.loss_terms

        def collect(*a, **k):
            terms = real(*a, **k)
            totals.append(float(terms['total'].detach()))
            recons.append(float(terms['recon'].detach()))
            return terms
        with mock.patch.object(rq, 'loss_terms', collect):
            result = rr.fit_r(X, arm='p0', K=4, seed=1, max_steps=150, batch_size=64, device='cpu',
                              wall_cap_seconds=1e9, log_every=50, probe_from=100, probe_step=200,
                              diagnostics=False)
        self.assertEqual([s for s, _ in result['recon_log']], [50, 100, 150])
        self.assertEqual([s for s, _ in result['loss_log']], [50, 100, 150])
        for i, ((_, recon), (_, total)) in enumerate(zip(result['recon_log'], result['loss_log'])):
            window = slice(i * 50, i * 50 + 50)
            self.assertAlmostEqual(recon, sum(recons[window]) / 50, places=4)
            self.assertAlmostEqual(total, sum(totals[window]) / 50, places=4)
            self.assertLessEqual(recon, total + 1e-9)

    def test_ms_per_step_is_elapsed_milliseconds_over_steps(self):
        X = blobs(n_per=32, centres=8, d=64, seed=6)
        kw = dict(K=4, seed=1, batch_size=64, device='cpu', wall_cap_seconds=1e9, probe_from=100, probe_step=200,
                  diagnostics=False)
        done = rr.fit_r(X, arm='p0', max_steps=400, clock=clock_stepping(0.01), **kw)
        self.assertAlmostEqual(done['elapsed'], 0.04, places=9)
        self.assertAlmostEqual(done['ms_per_step'], 0.04 * 1000 / 400, places=9)
        failed = rr.fit_r(X, arm='p0', max_steps=5000, clock=clock_stepping(1000.0), project=False,
                          **dict(kw, wall_cap_seconds=7200))
        self.assertEqual(failed['status'], 'fail_wall')
        self.assertAlmostEqual(failed['ms_per_step'], failed['elapsed'] * 1000 / failed['step'], places=9)

    def test_adam_arm_trains_and_resets_without_error(self):
        small = {'reset_check_every': 10, 'dead_window': 20, 'reset_until': 60}
        X = unit_rows(np.random.RandomState(31), 1024)
        with mock.patch.dict(rr.PREREG_R['arms']['l4'], small):
            result = rr.fit_r(X, arm='l4', K=64, seed=3, max_steps=100, batch_size=64, device='cpu',
                              wall_cap_seconds=1e9, probe_from=50, probe_step=100, check_every=50,
                              diagnostics=False)
        self.assertEqual(result['status'], 'complete')
        for tensor in result['state']['state_dict'].values():
            self.assertTrue(bool(torch.isfinite(tensor).all()))


class StaticChecks(unittest.TestCase):
    def test_module_source_has_no_data_access_and_no_bincount(self):
        source = inspect.getsource(rr)
        doc = ast.get_docstring(ast.parse(source), clean=False)
        body = source.replace(doc, '', 1) if doc else source
        for token in ('validation-queries', 'examples.jsonl', 'support.json', 'history', 'target', 'open(',
                      'bincount'):
            self.assertNotIn(token, body, token)


if __name__ == '__main__':
    unittest.main()
