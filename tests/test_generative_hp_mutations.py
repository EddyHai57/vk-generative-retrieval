"""G-HP-01 mutation tests (M1-M16): inject a defect into the production code with unittest.mock and demand that the guards
named for it turn red through their own assertion (AssertionError). Every mutant is first run against a control (the guard is
green on the real code), then under the mutation. A mutant that survives makes the test fail. Guards live in hp_checks.py.

Every configurable knob has at least one mutant: width (M1, M2), layers (M3, M4), dropout (M5, M6), learning rate (M7).
"""
import contextlib
import unittest
from pathlib import Path
from unittest import mock

import torch

import hp_checks as hc
from test_generative_b1 import engineering_world
from test_generative_batched import synthetic, tmpdir
from vkgr import generative_b1_arch as gba
from vkgr import generative_b1_hp as bh
from vkgr import generative_b1_run as b1
from vkgr import generative_hp_arch as hp

ROOT = Path(__file__).resolve().parents[1]


def view(**forced):
    """``_arch_view`` that silently overrides some structure fields (the factory 'drops' a knob)."""
    return lambda: mock.patch.object(hp, '_arch_view', lambda config: {**config, **forced})


class MutationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tmpdir()
        cls.context = synthetic(Path(cls.dir.name))
        cls.guards = {
            'H3': hc.check_shapes, 'H4': hc.check_optimizer, 'D1': lambda: hc.check_dropout_a1(cls.context), 'D2': hc.check_dropout_zero,
            'R-A': hc.check_registry_accepts, 'R-B': hc.check_registry_rejects, 'R-C': hc.check_registry_isolation, 'E3': hc.check_globals_restored,
            'P2': lambda: hc.check_gate(hc.load_script('ghp01_gate'), constants=False)}

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def kill(self, mutant, mutation, *names):
        for name in names:
            self.guards[name]()
        for name in names:
            with self.subTest(mutant=mutant, guard=name), mutation():
                with self.assertRaises(AssertionError, msg=f'{mutant} survived guard {name}'):
                    self.guards[name]()

    # ---- factory: one mutant per structure knob --------------------------------------------------------------
    def test_m1_width_dropped(self):
        self.kill('M1 d_model/ff fixed to 128/1024', view(d_model=128, ff=1024), 'H3')

    def test_m2_head_dim_dropped(self):
        self.kill('M2 head_dim fixed to 64 (G1 becomes 6x64)', view(head_dim=64), 'H3')

    def test_m3_layers_dropped(self):
        self.kill('M3 layers fixed to 4+4', view(encoder_layers=4, decoder_layers=4), 'H3')

    def test_m4_only_encoder_layers_follow(self):
        self.kill('M4 decoder layers stay 4', view(decoder_layers=4), 'H3')

    def test_m5_dropout_fixed_01(self):
        self.kill('M5 dropout fixed to 0.1', view(dropout=0.1), 'D2')

    def test_m6_dropout_fixed_0(self):
        self.kill('M6 dropout fixed to 0.0', view(dropout=0.0), 'D1')

    # ---- optimizer ------------------------------------------------------------------------------------------
    def test_m7_optimizer_ignores_learning_rate(self):
        real = torch.optim.AdamW
        self.kill('M7 lr pinned to 3e-4', lambda: mock.patch.object(hp, 'AdamW', lambda params, lr, weight_decay: real(params, lr=.0003, weight_decay=weight_decay)), 'H4')

    def test_m8_optimizer_ignores_weight_decay(self):
        real = torch.optim.AdamW
        self.kill('M8 weight decay dropped', lambda: mock.patch.object(hp, 'AdamW', lambda params, lr, weight_decay: real(params, lr=lr)), 'H4')

    # ---- registry -------------------------------------------------------------------------------------------
    def test_m9_registry_lets_off_grid_through(self):
        self.kill('M9a off-grid structure d_model 192', lambda: mock.patch.object(hp, 'HP_SHAPES', hp.HP_SHAPES + ((192, 6, 64, 4, 4, 1024, 0.1),)), 'R-B')
        self.kill('M9b lr 1e-3', lambda: mock.patch.object(hp, 'LEARNING_RATES', hp.LEARNING_RATES + (.001,)), 'R-B')

    def test_m10_registry_rejects_g4_or_g5(self):
        g4 = tuple(hc.grun_config('G4')[k] for k in hp.SHAPE_KEYS)
        g5 = tuple(hc.grun_config('G5')[k] for k in hp.SHAPE_KEYS)
        for label, shape in (('G4', g4), ('G5', g5)):
            dropped = tuple(s for s in hp.HP_SHAPES if s != shape)
            self.kill(f'M10a {label} structure removed', lambda dropped=dropped: mock.patch.object(hp, 'HP_SHAPES', dropped), 'R-A')
            replaced = tuple((192, 6, 64, 4, 4, 1024, 0.1) if s == shape else s for s in hp.HP_SHAPES)  # still 8 distinct shapes
            self.kill(f'M10b {label} structure replaced', lambda replaced=replaced: mock.patch.object(hp, 'HP_SHAPES', replaced), 'R-A')

    def test_m11_task_id_not_checked(self):
        fixed = {k: v for k, v in hp.FIXED.items() if k != 'task_id'}
        self.kill('M11 task_id unchecked', lambda: mock.patch.object(hp, 'FIXED', fixed), 'R-C')

    # ---- entry ----------------------------------------------------------------------------------------------
    def test_m12_runner_without_restore(self):
        """Restore disabled in the runner chain. Removing only the G-HP ``finally`` is an equivalent mutant: the enclosing GARCH-01 runner
        restores the same four names to their pre-entry values on exit, so no global leaks (see docs/ghp01_switches.md)."""
        before = {n: getattr(b1, n) for n in hc.RUNNER_GLOBALS}
        self.guards['E3']()
        try:
            with mock.patch.object(gba, '_restore', lambda saved: None):
                with self.assertRaisesRegex(AssertionError, 'was not restored', msg='M12 survived guard E3'):
                    self.guards['E3']()
        finally:
            for name, value in before.items():
                setattr(b1, name, value)

    def test_m13_members_miss_a_ghp_file(self):
        with tmpdir() as tmp:
            _, receipt, manifest, sha = engineering_world(tmp)
            common = (ROOT, receipt['manifest_path'], receipt['manifest_sha256'], manifest, sha)
            control = bh.run(*common, hc.entry_config(max_epochs=1), Path(tmp) / 'control', temporary=Path(tmp) / 's0')
            hc.check_binding_members(control)
            for index, kept in enumerate(hc.HP_FILES):
                with self.subTest(kept=kept), mock.patch.object(bh, 'MEMBERS', gba.MEMBERS + (kept,)):
                    value = bh.run(*common, hc.entry_config(max_epochs=1), Path(tmp) / f'm{index}', temporary=Path(tmp) / f's{index + 1}')
                    with self.assertRaises(AssertionError, msg='M13 survived guard E1'):
                        hc.check_binding_members(value)

    # ---- scripts --------------------------------------------------------------------------------------------
    def test_m14_combo_counts_runs_not_knobs_or_threshold_5(self):
        combo = hc.load_script('ghp01_combo')
        hc.check_combo(combo, constants=False)
        mutants = (('M14a G2 and G3 counted as two knobs', lambda: _stack(mock.patch.dict(combo.KNOB_OF, {'G3': 'lr2'}), mock.patch.dict(combo.KNOBS, {'lr2': ('learning_rate',)}))),
                   ('M14b win threshold 5 hits', lambda: mock.patch.object(combo, 'MIN_DELTA_HITS', 5)))
        for label, mutation in mutants:
            with self.subTest(mutant=label), mutation():
                with self.assertRaises(AssertionError, msg=f'{label} survived guard P3'):
                    hc.check_combo(combo, constants=False)

    def test_m15_gate_threshold_195_or_verify_ignored(self):
        gate = hc.load_script('ghp01_gate')
        self.guards['P2']()
        real = gate.decide

        def ignore_verify(train_terminal, train_manifest, train_manifest_sha256, verify_terminal, verify_last_line, verify_run):
            best = ((train_manifest or {}).get('selection') or {}).get('best_epoch')
            return real(train_terminal, train_manifest, train_manifest_sha256, {'status': 'succeeded'}, {'status': 'verified', 'best_epoch': best},
                        {'argv': ['--manifest-sha', train_manifest_sha256]})
        for label, mutation in (('M15a threshold 195 hits', lambda: mock.patch.object(gate, 'MIN_HITS', 195)),
                                ('M15b verify not checked', lambda: mock.patch.object(gate, 'decide', ignore_verify))):
            with self.subTest(mutant=label), mutation():
                with self.assertRaises(AssertionError, msg=f'{label} survived guard P2'):
                    hc.check_gate(gate, constants=False)

    def test_m16_parity_flag_pinned_to_a_constant(self):
        with tmpdir() as tmp:
            script, _, call = hc.parity_world(tmp)
            hc.check_parity_report(call, 'control-equal')
            hc.check_parity_perturbed(call, 'control-perturbed')
            cases = ((False, hc.check_parity_report), (True, hc.check_parity_perturbed))
            for constant, guard in cases:
                with self.subTest(constant=constant), mock.patch.object(script, '_equal_steps', lambda first_diff, constant=constant: constant):
                    with self.assertRaises(AssertionError, msg=f'M16 survived with train_steps_equal pinned to {constant}'):
                        guard(call, f'm16-{constant}')


@contextlib.contextmanager
def _stack(*managers):
    with contextlib.ExitStack() as stack:
        for manager in managers:
            stack.enter_context(manager)
        yield


if __name__ == '__main__':
    unittest.main()
