"""GFEAT-02 entry (vkgr.generative_b1_arch_feat) on the frozen-base engineering fixture (local harness only: G1 engineering contract).

Run inside the frozen-base harness with OMP_NUM_THREADS=1 and ``discover -s tests``.
"""
import contextlib
import copy
import hashlib
import io
import json
import os
import unittest
from pathlib import Path
from unittest import mock

import torch

import arch_feat_checks as fc
from test_generative_feat import flip_outcomes, side_world, tmpdir, v2_context

ROOT = Path(__file__).resolve().parents[1]
B1 = {'mode': 'batched', 'batch_size': 1}


def arch_config(**overrides):
    """The GARCH-01 config of the same engineering structure (what feature=none must reproduce)."""
    value = {k: v for k, v in fc.config('none', **overrides).items() if k not in ('feature', 'side_version')}
    value.update(schema='g-b1-arch-run-v1', task_id='G-ARCH-01')
    return value


def projection(run):
    return ([(e['epoch'], e['model_sha256'], e['evaluation']['computation_sha256'], json.dumps(e['metrics'], sort_keys=True)) for e in run['epochs']],
            run['losses'], run['selection'], run['best']['epoch'], run['best']['metric'])


def without_model(run):
    """Same projection without the model digests (a model with extra all-zero side tables has a different digest by construction)."""
    epochs, losses, selection, best, metric = projection(run)
    return [e[:1] + e[2:] for e in epochs], losses, selection, best, metric


class EntryRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tmpdir(); cls.addClassCleanup(cls._tmp.cleanup)
        cls.tmp = Path(cls._tmp.name); cls.world = w = side_world(cls.tmp)
        cls.common = (ROOT, w.receipt['manifest_path'], w.receipt['manifest_sha256'], w.hist_manifest, w.hist_sha)
        cls.side = (w.side_manifest, w.side_sha)
        cls.runs = {}

    def run_entry(self, name, feature, *, decode=None):
        from vkgr import generative_b1_arch_feat as baf
        if name not in self.runs:
            self.runs[name] = baf.run(*self.common, fc.config(feature), self.tmp / name, temporary=self.tmp / f'{name}-s', decode=decode,
                                      side=None if feature == 'none' else self.side)
        return self.runs[name]

    def run_garch01(self, name, decode=None):
        from vkgr import generative_b1_arch as ba
        if name not in self.runs:
            self.runs[name] = ba.run(*self.common, arch_config(), self.tmp / name, temporary=self.tmp / f'{name}-s', decode=decode)
        return self.runs[name]

    def test_a_none_reproduces_the_garch01_entry(self):
        """feature=none: model_sha256, computation_sha256, metrics, losses and selection equal the GARCH-01 entry epoch by epoch (serial, b=1)."""
        reference = self.run_garch01('garch01-serial')
        self.assertEqual(len(reference['epochs']), 3)
        for name, run in (('garch01 b=1', self.run_garch01('garch01-b1', B1)), ('none serial', self.run_entry('none-serial', 'none')),
                          ('none b=1', self.run_entry('none-b1', 'none', decode=B1))):
            with self.subTest(run=name):
                self.assertEqual(projection(run), projection(reference))
        none = self.runs['none-serial']
        self.assertEqual(none['binding']['init_report'], reference['binding']['init_report'])
        self.assertEqual(none['binding']['feat01'], {'feature': 'none', 'side_version': None})
        files = none['binding']['source']['files']
        for member in ('src/vkgr/generative_arch_feat.py', 'src/vkgr/generative_b1_arch_feat.py', 'src/vkgr/tiger_arch.py', 'src/vkgr/generative_feat.py', 'src/vkgr/history_side.py'):
            self.assertIn(member, files)
        self.assertNotIn('src/vkgr/generative_arch_feat.py', reference['binding']['source']['files'])

    def test_b_zero_frozen_side_run_equals_none_run(self):
        """f2/f3 with the side tables zeroed and frozen right after their creation: losses, evaluation computation, metrics and
        selection equal feature=none bit for bit (dropout 0.1 on: the side-table draws do not move the global random stream)."""
        from vkgr import generative_arch_feat as gaf
        from vkgr import generative_b1_arch_feat as baf
        none = self.run_entry('none-b1', 'none', decode=B1)
        real = gaf.TigerArchModelSide.add_side_tables

        def zeroed(self, feature):
            real(self, feature); fc.zero_side(self)
        for feature in fc.SIDE_TABLES:
            with self.subTest(feature=feature), mock.patch.object(gaf.TigerArchModelSide, 'add_side_tables', zeroed):
                run = baf.run(*self.common, fc.config(feature), self.tmp / f'zero-{feature}', temporary=self.tmp / f'zero-{feature}-s', decode=B1, side=self.side)
                self.assertEqual(without_model(run), without_model(none))

    def test_f2_f3_train_then_serial_verify(self):
        from vkgr import generative_b1_arch_feat as baf
        none = self.run_entry('none-b1', 'none', decode=B1)
        for feature in fc.SIDE_TABLES:
            with self.subTest(feature=feature):
                value = self.run_entry(feature, feature, decode=B1); run = self.tmp / feature
                self.assertEqual((value['status'], len(value['epochs'])), ('complete', 3))
                self.assertEqual(sorted(p.name for p in run.glob('*.pt')), ['best.pt', 'last.pt'])
                self.assertEqual(value['binding']['feat01'], {'feature': feature, 'side_version': 'HIST_V2_SIDE_V1'})
                self.assertEqual(value['binding']['init_report']['arch'], none['binding']['init_report']['arch'])
                self.assertEqual(value['binding']['init_report']['side']['feature'], feature)
                feat01 = json.loads((run / 'evaluation-001' / 'manifest.json').read_text())['binding']['context']['feat01']
                self.assertEqual((feat01['side_manifest_sha256'], feat01['history_manifest_sha256']), (self.world.side_sha, self.world.hist_sha))
                self.assertNotEqual(value['losses'], none['losses'])  # the side input really reaches the loss
                report = baf.verify(*self.common, run / 'manifest.json', value['manifest_sha256'], self.tmp / f'{feature}-v', temporary=self.tmp / f'{feature}-vs', side=self.side)
                self.assertEqual((report['status'], report['replayed_epochs']), ('verified', [value['best']['epoch']]))
                with self.assertRaises(ValueError):  # f2/f3 verify needs the side arguments
                    baf.verify(*self.common, run / 'manifest.json', value['manifest_sha256'], self.tmp / f'{feature}-v2', temporary=self.tmp / f'{feature}-vs2')

    def test_entries_do_not_cross_verify_and_globals_restored(self):
        from vkgr import generative_b1_arch as ba
        from vkgr import generative_b1_arch_feat as baf
        from vkgr import generative_b1_feat as feat
        from vkgr import generative_b1_run as b1
        fc.check_runner_installs_and_restores()
        before = {n: getattr(b1, n) for n in fc.RUNNER_GLOBALS}
        f2 = self.run_entry('f2_feedback', 'f2_feedback', decode=B1); none = self.run_entry('none-b1', 'none', decode=B1)
        garch = self.run_garch01('garch01-serial')
        cases = (('GARCH-01 verify on f2', ba.verify, 'f2_feedback', f2, {}), ('GFEAT-01 verify on f2', feat.verify, 'f2_feedback', f2, {'side': self.side}),
                 ('GARCH-01 verify on none', ba.verify, 'none-b1', none, {}), ('GFEAT-02 verify on GARCH-01', baf.verify, 'garch01-serial', garch, {}))
        for index, (label, call, run, value, kwargs) in enumerate(cases):
            with self.subTest(label), self.assertRaises(ValueError):
                call(*self.common, self.tmp / run / 'manifest.json', value['manifest_sha256'], self.tmp / f'x-v{index}', temporary=self.tmp / f'x-s{index}', **kwargs)
        for n in fc.RUNNER_GLOBALS:
            self.assertIs(getattr(b1, n), before[n], n)
        self.assertEqual(baf.verify(*self.common, self.tmp / 'none-b1' / 'manifest.json', none['manifest_sha256'], self.tmp / 'none-v', temporary=self.tmp / 'none-vs')['status'], 'verified')

    def test_rejected_variants_publish_nothing(self):
        from vkgr import generative_b1_arch_feat as baf
        fc.check_gates()
        cases = {'none with side': (fc.config('none'), {'side': self.side}), 'f2 without side': (fc.config('f2_feedback'), {}),
                 'history 100': (fc.config('f2_feedback', history_len=100), {'side': self.side}), 'dropout 0.6': (fc.config('f2_feedback', dropout=0.6), {'side': self.side}),
                 'workers 2': (fc.config('none'), {'eval_workers': 2}), 'batch 2': (fc.config('none'), {'decode': {'mode': 'batched', 'batch_size': 2}})}
        for index, (label, (cfg, kwargs)) in enumerate(cases.items()):
            out = self.tmp / f'rej-{index}'
            with self.subTest(label), self.assertRaises(ValueError):
                baf.run(*self.common, cfg, out, temporary=self.tmp / f'rej-{index}-s', **kwargs)
            self.assertFalse(out.exists(), label)
            if label in ('none with side', 'f2 without side', 'workers 2', 'batch 2'):  # these fail before the scratch directory exists
                self.assertFalse((self.tmp / f'rej-{index}-s').exists(), label)


class LeakageTests(unittest.TestCase):
    """Anti-leakage on the GFEAT-02 model (GFEAT-01 method): flip the raw target row and every later row of the user, REBUILD the
    sidecar from the flipped rows -> that example's encoder memory and training loss are bit-identical. Reverse control: a past event changes both."""

    def test_target_and_later_fields_cannot_change_encoder_input(self):
        from vkgr import generative_arch as ga
        from vkgr import generative_feat as gf
        from vkgr import history_side, history_v2
        from vkgr.generative_data import sid_targets
        with tmpdir() as tmp:
            world = side_world(tmp); loaded, ctx = v2_context(world)
            rows = ctx.examples.rows(list(range(len(ctx.examples))))

            def pack_from(partitions):
                side = history_side.build_side(partitions, history_v2.load_prepared(world.hist_manifest, world.hist_sha, 100), world.durations)
                return gf.build_side_pack({s: [r[-50:] for r in v] for s, v in side.items()}, loaded, ctx, world.hist_sha, world.side_sha)
            base = pack_from(world.partitions)

            def measure(feature, model, pack, row):
                tokens = gf.SIDTokensSide(ctx.codec, 50, pack); model.eval()
                with torch.no_grad():
                    memory = model.encode(sid_targets([row], ctx.codec)[0], tokens.train_side([row]))[0]
                clone = copy.deepcopy(model); cfg = fc.config(feature); torch.manual_seed(5)
                loss = gf.train_batch_feat(clone, ga.make_optimizer_arch(clone, cfg), [row], ctx, tokens, cfg, 'cpu').loss_mean
                return memory, loss
            for feature in fc.SIDE_TABLES:
                model = fc.build(ctx.codec, feature)
                for e in world.examples:
                    row = rows[e['example_id']]; week, ordinal = e['target_position']
                    mutated = copy.deepcopy(world.partitions)
                    for wk, part in mutated:
                        for k, raw in enumerate(part):
                            if raw['user_id'] == e['user_id'] and (wk, k) >= (week, ordinal):
                                flip_outcomes(raw)
                    (m1, l1), (m2, l2) = measure(feature, model, base, row), measure(feature, model, pack_from(mutated), row)
                    with self.subTest(feature=feature, example=e['example_id']):
                        self.assertTrue(torch.equal(m1, m2)); self.assertEqual(l1, l2)
                e = next(x for x in world.examples if len(loaded['train'][x['example_id']]['history_positions']) >= 2)
                row = rows[e['example_id']]; week, ordinal = loaded['train'][e['example_id']]['history_positions'][-1]
                mutated = copy.deepcopy(world.partitions); flip_outcomes(dict(mutated)[week][ordinal])
                (m1, l1), (m3, l3) = measure(feature, model, base, row), measure(feature, model, pack_from(mutated), row)
                with self.subTest(feature=feature, control='past event'):
                    self.assertFalse(torch.equal(m1, m3)); self.assertNotEqual(l1, l3)


class CliTests(unittest.TestCase):
    def test_cli_gates_before_output(self):
        from vkgr import generative_b1_arch_feat as baf
        from vkgr import generative_sharded as gs
        side = ('--side-manifest', 'z', '--side-sha', '0' * 64)
        cases = {'batch 2': ({'VKGR_DECODE': 'batched', 'VKGR_DECODE_BATCH': '2'}, 'none', ()), 'workers 2': ({'VKGR_EVAL_WORKERS': '2'}, 'none', ()),
                 'none with side': ({}, 'none', side), 'f2 without side': ({}, 'f2_feedback', ()), 'half side': ({}, 'f2_feedback', side[:2])}
        for label, (env, feature, extra) in cases.items():
            with self.subTest(label), tmpdir() as tmp:
                path = Path(tmp) / 'cfg.json'; path.write_text(json.dumps(fc.config(feature), sort_keys=True))
                argv = ['train', '--prepared', 'x', '--prepared-sha', '0' * 64, '--history-manifest', 'y', '--history-sha', '0' * 64, '--config', str(path),
                        '--config-sha', hashlib.sha256(path.read_bytes()).hexdigest(), '--output', str(Path(tmp) / 'out'), '--scratch', str(Path(tmp) / 's'), *extra]
                err = io.StringIO()
                with mock.patch.dict(os.environ, env), mock.patch.object(gs, 'cpu_quota_cores', return_value=None), contextlib.redirect_stderr(err):
                    code = baf.main(argv)
                self.assertEqual(code, 1, err.getvalue()); self.assertIn('G-FEAT02 gate failed', err.getvalue())
                self.assertFalse((Path(tmp) / 'out').exists() or (Path(tmp) / 's').exists())

    def test_cli_train_batched_b1_then_serial_verify(self):
        from vkgr import generative_b1_arch_feat as baf
        from vkgr import generative_sharded as gs
        with tmpdir() as tmp:
            tmp = Path(tmp); w = side_world(tmp)
            path = tmp / 'cfg.json'; path.write_text(json.dumps(fc.config('f3_watch_buckets'), indent=1, sort_keys=True) + '\n')
            common = ['--prepared', str(w.receipt['manifest_path']), '--prepared-sha', w.receipt['manifest_sha256'], '--history-manifest', str(w.hist_manifest),
                      '--history-sha', w.hist_sha, '--side-manifest', str(w.side_manifest), '--side-sha', w.side_sha]

            def cli(args, env):
                out, err = io.StringIO(), io.StringIO()
                clean = {k: v for k, v in os.environ.items() if not k.startswith('VKGR_')}
                with mock.patch.dict(os.environ, {**clean, **env}, clear=True), mock.patch.object(gs, 'cpu_quota_cores', return_value=None), \
                        contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    code = baf.main(args)
                self.assertEqual(code, 0, err.getvalue())
                return json.loads(out.getvalue().strip().splitlines()[-1])
            trained = cli(['train', *common, '--config', str(path), '--config-sha', hashlib.sha256(path.read_bytes()).hexdigest(),
                           '--output', str(tmp / 'run'), '--scratch', str(tmp / 'run-s')], {'VKGR_DECODE': 'batched', 'VKGR_DECODE_BATCH': '1'})
            self.assertEqual(trained['status'], 'complete')
            report = cli(['verify', *common, '--manifest', str(tmp / 'run' / 'manifest.json'), '--manifest-sha', trained['manifest_sha256'],
                          '--output', str(tmp / 'v'), '--scratch', str(tmp / 'v-s')], {})
            self.assertEqual(report['status'], 'verified')


if __name__ == '__main__':
    unittest.main()
