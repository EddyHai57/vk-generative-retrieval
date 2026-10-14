"""GTOK-01R FEAT-001: every mutant of sid_rqvae_r (E1-E7, R1-R18, O1-O4, plus extras) must be killed by tests.

A mutant is a one-place source edit of src/vkgr/sid_rqvae_r.py that is executed inside the real module namespace
(so spies and mock.patch calls in the tests act on the mutated functions) and restored afterwards. A mutant counts
as killed only when the selected tests report at least one assertion failure and no unrelated error.
Mutants that are provably output-equivalent (R17, R18) are killed only through a direct contract assertion and are
documented as equivalent, never as behaviour-killed.
"""
import contextlib
import importlib
import io
import unittest
from pathlib import Path

import numpy as np

from vkgr import sid_rqvae as rq

P = 'tests.test_sid_rqvae_r.'
PREREG = [P + 'Prereg.test_prereg_r_arms_and_constants', P + 'Prereg.test_make_optimizer_types_and_all_hyperparameters']
EQ_FAST = [P + 'P0Equivalence.test_p0_matches_fit_past_the_first_possible_reset_step_k256',
           P + 'P0Equivalence.test_p0_matches_fit_past_the_first_possible_reset_step_k16']
EQ_ORDER = [P + 'P0Equivalence.test_optimizer_is_created_after_codebook_initialisation']
EQ_NO_RESET = [P + 'P0Equivalence.test_p0_never_touches_the_reset_machinery']
RESET = [P + 'ResetDeadCodes']
LAST_USED = [P + 'LastUsed']
SPY = [P + 'LArms.test_reset_stream_internal_state_and_call_order']
LARMS_LOG = [P + 'LArms.test_resets_follow_the_small_window_and_reproduce',
             P + 'LArms.test_no_reset_after_reset_until']
PROJECT = [P + 'LArms.test_project_flag_only_disables_the_extrapolation']
MS_STEP = [P + 'LArms.test_ms_per_step_is_elapsed_milliseconds_over_steps']
DIAG_RNG = [P + 'Diagnostics.test_diag_subset_uses_its_own_generator',
            P + 'Diagnostics.test_diagnostics_leave_global_random_state_alone']
DIAG_TRAIN = [P + 'Diagnostics.test_diagnostics_do_not_change_training_p0',
              P + 'Diagnostics.test_diagnostics_do_not_change_training_l1_with_resets']
DIAG_GRAD = [P + 'Diagnostics.test_record_steps_fields_and_gradient_statistics',
             P + 'Diagnostics.test_step_one_gradient_norm_and_update_match_a_manual_step']
INIT_ZERO = [P + 'LastUsed.test_new_last_used_is_zero_int64_per_layer']
CONTRACT_SCHEDULE = [P + 'ResetDeadCodes.test_checkpoint_schedule']

ALL_SELECTED = (PREREG + EQ_FAST + EQ_ORDER + EQ_NO_RESET + RESET + LAST_USED + SPY + LARMS_LOG + PROJECT + MS_STEP
                + DIAG_RNG + DIAG_TRAIN + DIAG_GRAD)


def run_ids(ids):
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name) for name in ids)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=0).run(suite)
    result.transcript = stream.getvalue()
    return result


@contextlib.contextmanager
def mutated(case, edits):
    """Apply the (old, new) source edits inside the live module namespace; restore it afterwards."""
    module = importlib.import_module('vkgr.sid_rqvae_r')
    path = Path(module.__file__)
    source = path.read_text()
    for old, new in edits:
        case.assertEqual(source.count(old), 1, 'mutation anchor must occur exactly once: ' + repr(old))
        source = source.replace(old, new, 1)
    saved = dict(module.__dict__)
    try:
        exec(compile(source, str(path), 'exec'), module.__dict__)
        yield module
    finally:
        module.__dict__.clear()
        module.__dict__.update(saved)


def assert_killed(case, edits, test_ids):
    with mutated(case, edits):
        result = run_ids(test_ids)
    case.assertFalse(result.wasSuccessful(), 'mutant survived: ' + repr(edits))
    case.assertTrue(result.failures, 'mutant killed by errors only, not by an assertion:\n' + result.transcript[-1500:])
    case.assertEqual(result.errors, [], 'mutant raised unrelated errors:\n' + result.transcript[-1500:])


def one(old, new):
    return [(old, new)]


MUTANTS = {
    # --- E: equivalence with rq.fit, diagnostics are read-only --------------------------------------------------
    'E1a p0 path consumes an extra random permutation': (
        one("    order = rng.permutation(n)\n    position = 0",
            "    order = rng.permutation(n)\n    rng.permutation(n)\n    position = 0"), EQ_FAST),
    'E1b torch seed applied after model creation': (
        [("    torch.manual_seed(seed)\n", ""),
         ("    X = torch.from_numpy(content).to(device)\n",
          "    torch.manual_seed(seed)\n    X = torch.from_numpy(content).to(device)\n")], EQ_FAST),
    'E1c data stream seeded differently': (
        one("    rng = np.random.RandomState(seed)\n", "    rng = np.random.RandomState(seed + 1)\n"), EQ_FAST),
    'E2 optimizer created before the k-means initialisation': (
        [("    optimizer = make_optimizer(model.parameters(), arm)\n", ""),
         ("    rq.init_codebooks(model, rq._first_batch(",
          "    optimizer = make_optimizer(model.parameters(), arm)\n    rq.init_codebooks(model, rq._first_batch(")],
        EQ_ORDER),
    'E3 p0 arm resets dead codes': (
        one("'eps': 1e-7, 'reset': False},", "'eps': 1e-7, 'reset': True, **_RESET},"),
        EQ_FAST + EQ_NO_RESET + PREREG),
    'E4 eval_every ignored': (
        one("        if step % eval_every == 0:", "        if step % 10000 == 0:"), EQ_FAST),
    'E5a project inverted': (
        one("            if project and elapsed", "            if not project and elapsed"), PROJECT),
    'E5b project ignored': (
        one("            if project and elapsed", "            if elapsed"), PROJECT),
    'E6a diagnostics subset drawn from the global numpy stream': (
        one("    generator = np.random.RandomState(PREREG_R['diag_subset_seed'])\n",
            "    generator = np.random\n"), DIAG_RNG),
    'E6b diagnostics consume torch random numbers': (
        one("        enc_dead, z = _relu_dead_fractions(model.encoder, x)\n",
            "        torch.rand(1)\n        enc_dead, z = _relu_dead_fractions(model.encoder, x)\n"), DIAG_RNG),
    'E7a diagnostics modify a parameter': (
        one("        enc_dead, z = _relu_dead_fractions(model.encoder, x)\n",
            "        next(model.parameters()).add_(1e-3)\n        enc_dead, z = _relu_dead_fractions(model.encoder, x)\n"),
        DIAG_TRAIN),
    'E7b diagnostics zero the gradients before reading them': (
        one("diagnose(model, sub, step, group_norms(model), rel_update(model, before))",
            "diagnose(model, sub, step, (model.zero_grad(set_to_none=False) or group_norms(model)), "
            "rel_update(model, before))"), DIAG_GRAD),
    # --- R: dead-code reset ---------------------------------------------------------------------------------------
    'R1 reset source is z for every layer': (
        one("    return residuals[layer].detach()", "    return residuals[0].detach()"), RESET),
    'R2 reset source is residuals[l+1]': (
        one("    return residuals[layer].detach()", "    return residuals[layer + 1].detach()"), RESET),
    'R3 reset source not detached': (
        one("    return residuals[layer].detach()", "    return residuals[layer]"), RESET + SPY),
    'R4 live codes are reset instead of dead ones': (
        one("dead = (step - table >= cfg['dead_window']).nonzero().flatten()",
            "dead = (step - table < cfg['dead_window']).nonzero().flatten()"), RESET),
    'R5 adagrad accumulator row not reset': (
        one("state['sum'][rows] = optimizer.defaults['initial_accumulator_value']",
            "state['sum'][rows] = state['sum'][rows]"), RESET),
    'R6 adam first moment row not reset': (
        one("state['exp_avg'][rows] = 0.0", "state['exp_avg'][rows] = state['exp_avg'][rows]"), RESET),
    'R7 window comparison > instead of >=': (
        one("dead = (step - table >= cfg['dead_window']).nonzero().flatten()",
            "dead = (step - table > cfg['dead_window']).nonzero().flatten()"), RESET),
    'R8a dead window 499': (one("'dead_window': 500", "'dead_window': 499"), RESET + PREREG),
    'R8b dead window 501': (one("'dead_window': 500", "'dead_window': 501"), RESET + PREREG),
    'R9 reset executed before optimizer.step()': (
        [("        optimizer.step()\n        if cfg['reset']:\n", "        if cfg['reset']:\n"),
         ("            reset_log.extend(fired)\n", "            reset_log.extend(fired)\n        optimizer.step()\n")],
        SPY),
    'R10 resets continue after reset_until': (
        one("cfg['dead_window'] <= step <= cfg['reset_until']", "cfg['dead_window'] <= step"), RESET + LARMS_LOG),
    'R11 reset cadence follows check_every': (
        one("        if cfg['reset']:\n            touch_last_used(",
            "        if cfg['reset'] and step % check_every == 0:\n            touch_last_used("),
        SPY + LARMS_LOG),
    'R12 choice is called even when no code is dead': (
        one("        if n_dead == 0:\n            continue\n", ""), RESET),
    'R13 reset_log layer is 1-based': (one("'layer': layer,", "'layer': layer + 1,"), RESET + LARMS_LOG),
    'R14a last_used written as step-1': (
        one("        table[index] = step\n", "        table[index] = step - 1\n"), LAST_USED + SPY),
    'R14b last_used updated for two layers only': (
        one("for table, index in zip(last_used, codes):", "for table, index in zip(last_used[:2], codes[:2]):"),
        LAST_USED + SPY),
    'R15a reset stream seeded without the offset': (
        one("np.random.RandomState(seed + cfg['rng_reset_offset'])", "np.random.RandomState(seed)"), SPY),
    'R15b reset stream shared with the data stream': (
        one("rng_reset = np.random.RandomState(seed + cfg['rng_reset_offset']) if cfg['reset'] else None",
            "rng_reset = rng if cfg['reset'] else None"), SPY),
    'R16 reset_until comparison < instead of <=': (
        one("cfg['dead_window'] <= step <= cfg['reset_until']", "cfg['dead_window'] <= step < cfg['reset_until']"),
        RESET),
    'R17 no dead_window lower bound (equivalent in reachable states)': (
        one("cfg['dead_window'] <= step <= cfg['reset_until']", "step <= cfg['reset_until']"), CONTRACT_SCHEDULE),
    'R18 last_used initial value -inf (equivalent for step >= dead_window)': (
        one("torch.zeros(c.shape[0], dtype=torch.int64, device=c.device)",
            "torch.full((c.shape[0],), -(2 ** 40), dtype=torch.int64, device=c.device)"), INIT_ZERO),
    # --- O: optimizer arm table -----------------------------------------------------------------------------------
    'O1 l2 lr taken from l1': (one("'l2': {'optimizer': 'adagrad', 'lr': 0.1,", "'l2': {'optimizer': 'adagrad', 'lr': 0.4,"),
                               PREREG),
    'O2 l3 lr wrong': (one("'l3': {'optimizer': 'adagrad', 'lr': 0.025,", "'l3': {'optimizer': 'adagrad', 'lr': 0.05,"),
                       PREREG),
    'O3 l4 betas wrong': (one("'betas': (0.9, 0.999)", "'betas': (0.9, 0.99)"), PREREG),
    'O4 l1 accumulator wrong': (
        one("'l1': {'optimizer': 'adagrad', 'lr': 0.4, 'initial_accumulator_value': 0.1,",
            "'l1': {'optimizer': 'adagrad', 'lr': 0.4, 'initial_accumulator_value': 0.01,"), PREREG),
    # --- extras (not required by the plan) ------------------------------------------------------------------------
    'X1 reset forgets to refresh last_used': (one("        table[dead] = step\n", "        pass\n"), RESET),
    'X2 adam second moment row not reset': (
        one("state['exp_avg_sq'][rows] = 0.0", "state['exp_avg_sq'][rows] = state['exp_avg_sq'][rows]"), RESET),
    'X3 reset cadence hard-coded to 100': (
        one("step % cfg['reset_check_every'] == 0", "step % 100 == 0"), RESET),
    'X4 replacement rows are the first rows, not the drawn ones': (
        one("codebook[dead] = src[rows]", "codebook[dead] = src[:n_dead]"), RESET),
    'X5 ms_per_step missing the factor 1000': (
        one("'ms_per_step': elapsed * 1000.0 / step", "'ms_per_step': elapsed / step"), MS_STEP),
    'X6 recon_log records the total loss': (
        one("recon_log.append((step, float(running_recon) / counted))",
            "recon_log.append((step, float(running) / counted))"), [P + 'LArms.test_recon_log_matches_window_means_and_loss_log']),
}

REQUIRED = ([f'E{i}' for i in range(1, 8)] + [f'R{i}' for i in range(1, 19)] + [f'O{i}' for i in range(1, 5)])


class Mutants(unittest.TestCase):
    def test_unmutated_selected_tests_pass(self):
        result = run_ids(ALL_SELECTED)
        self.assertTrue(result.wasSuccessful(), result.transcript[-3000:])

    def test_every_required_mutant_is_defined(self):
        names = [name.split(' ')[0] for name in MUTANTS]
        for tag in REQUIRED:
            self.assertTrue(any(n == tag or n.startswith(tag) and not n[len(tag)].isdigit() for n in names), tag)

    def test_mutants_are_killed(self):
        for name, (edits, ids) in MUTANTS.items():
            with self.subTest(mutant=name):
                assert_killed(self, edits, ids)


class EquivalentMutants(unittest.TestCase):
    """R9, R17, R18 and E2 change no output; they are killed by order / contract / init assertions only."""

    @staticmethod
    def small_run(module):
        from unittest import mock
        small = {'reset_check_every': 10, 'dead_window': 20, 'reset_until': 100}
        x = np.random.RandomState(31).normal(size=(1024, 64)).astype(np.float32)
        x = (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)
        with mock.patch.dict(module.PREREG_R['arms']['l1'], small):
            result = module.fit_r(x, arm='l1', K=64, seed=3, max_steps=100, batch_size=64, device='cpu',
                                  wall_cap_seconds=1e9, probe_from=50, probe_step=100, check_every=50,
                                  diagnostics=False)
        return result['reset_log'], rq.state_digest(result['state']['state_dict']), result['loss_log']

    def test_equivalent_mutants_do_not_change_reset_log_or_final_weights(self):
        baseline_module = importlib.import_module('vkgr.sid_rqvae_r')
        baseline = self.small_run(baseline_module)
        self.assertGreater(len(baseline[0]), 0)
        for name in ('R9 reset executed before optimizer.step()',
                     'R17 no dead_window lower bound (equivalent in reachable states)',
                     'R18 last_used initial value -inf (equivalent for step >= dead_window)',
                     'E2 optimizer created before the k-means initialisation'):
            with self.subTest(mutant=name):
                with mutated(self, MUTANTS[name][0]) as module:
                    self.assertEqual(self.small_run(module), baseline)


if __name__ == '__main__':
    unittest.main()
