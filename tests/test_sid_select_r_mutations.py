"""GTOK-01R FEAT-002: every mutant S1-S14 of the selection / diagnostic-criteria code (plus extras X*) must be killed.
A mutant is a one-place source edit of src/vkgr/sid_rqvae_r.py executed inside the real module namespace and
restored afterwards. It counts as killed only when the selected tests report at least one assertion failure and no
unrelated error (same harness as tests/test_sid_rqvae_r_mutations.py).
"""
import contextlib
import importlib
import io
import unittest
from pathlib import Path

P = 'tests.test_sid_select_r.'
CONST = [P + 'Constants']
SELECT = [P + 'SelectArm']
PREFLIGHT = [P + 'Preflight']
OVERRIDE = [P + 'Override']
I1 = [P + 'I1']
I2 = [P + 'I2']
SST_I4 = [P + 'SstAndI4']
SCHEMAS = [P + 'Schemas', P + 'ProbeSet', P + 'DecisionSchema']
ALL_SELECTED = CONST + SELECT + PREFLIGHT + OVERRIDE + I1 + I2 + SST_I4 + SCHEMAS


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
    'S1 arm order reversed (L4 first)': (
        one("    for arm in order:\n        if _qualified(per_arm[arm]):",
            "    for arm in reversed(order):\n        if _qualified(per_arm[arm]):"), SELECT),
    'S2 pick the arm with the highest R2 instead of the first qualified': (
        one("    decision, listed = _mechanical(control, per_arm)\n",
            "    decision, listed = _mechanical(control, per_arm)\n"
            "    if decision in PREREG_R['arm_order']:\n"
            "        decision = max((a for a in PREREG_R['arm_order'] if _qualified(per_arm[a])),\n"
            "                       key=lambda a: by[('T1', a)]['final']['r2'])\n"), SELECT),
    'S3 only the first K is checked': (
        one("    return all(_passes(c, QUALIFY) for c in per_variant.values())",
            "    return all(_passes(c, QUALIFY) for c in list(per_variant.values())[:1])"), SELECT),
    'S4 (d) checks only the end point': (
        one("for s in sel['d_points'])", "for s in sel['d_points'][-1:])"), SELECT),
    'S5a (d) starts at 10000 (constant)': (
        one("'d_points': (20000, 22000, 24000, 26000, 28000, 30000),",
            "'d_points': (10000, 12000, 14000, 16000, 18000, 20000, 22000, 24000, 26000, 28000, 30000),"),
        SELECT + CONST),
    'S5b (d) starts at 18000 (constant)': (
        one("'d_points': (20000, 22000, 24000, 26000, 28000, 30000),",
            "'d_points': (18000, 20000, 22000, 24000, 26000, 28000, 30000),"), SELECT + CONST),
    'S6a (a1) threshold >= turned into >': (
        one("    return all(au.usage_ok(u, K) for u in used)",
            "    return all(u * au.USAGE_NUM_DEN[1] > K * au.USAGE_NUM_DEN[0] for u in used)"), CONST + SELECT),
    'S6b (e) threshold >= turned into >': (
        one("    out['e'] = final['r2'] >= sel['e_r2_min']", "    out['e'] = final['r2'] > sel['e_r2_min']"), SELECT),
    'S7 (a2) used as a qualification condition instead of (a1)': (
        one("QUALIFY = ('a1', 'b', 'c', 'd', 'e')", "QUALIFY = ('a2', 'b', 'c', 'd', 'e')"), SELECT),
    'S8 non-complete arms (null criteria) pass qualification and ESCALATE': (
        one("    return all(c[k] is True for k in keys)", "    return all(c[k] is not False for k in keys)"), SELECT),
    'S9 override trusts its arguments instead of the fresh recomputation': (
        one("        ok = decision == 'ESCALATE' and override_arm in listed and",
            "        ok = override_arm in PREREG_R['arm_order'] and"), OVERRIDE),
    'S10 (a2) drops the "> p0_final_usage_max" condition': (
        one("high[l] > low[l] and high[l] > sel['p0_final_usage_max']", "high[l] > low[l]"), SELECT),
    'S11 usage_sub feeds I1': (
        one("    collapsed = all(s in full and max(full[s]) <= sel['i1_usage_max'] for s in sel['usage_log_steps'])",
            "    collapsed = all(max(r['usage_sub']) <= sel['i1_usage_max'] for r in probe['diagnostics'])"), I1),
    'S12 I2 reads the total loss log instead of the recon log': (
        one("probe['recon_log'] if s == sel['i2_step']", "probe['loss_log'] if s == sel['i2_step']"), I2),
    'S13 INVALID_CONTROL ignores a non-complete P0': (
        one("    out['valid'] = out['complete'] and out['loss'] and out['usage'] and out['final']",
            "    out['valid'] = out['loss'] and out['usage'] and out['final']"), SELECT),
    'S14 preflight exceeds changes the decision': (
        one("    return {'decision': decision, 'escalate_listed_arms': listed,",
            "    if pre is not None and pre['exceeds']:\n        decision = 'STOP'\n"
            "    return {'decision': decision, 'escalate_listed_arms': listed,"), PREFLIGHT),
    # --- extras -----------------------------------------------------------------------------------------------
    'X1 control valid when any K reproduces': (
        one("    if not all(c['valid'] for c in control.values()):", "    if not any(c['valid'] for c in control.values()):"),
        SELECT),
    'X2 P0 usage slack 0': (one("'p0_usage_slack': 1,", "'p0_usage_slack': 0,"), SELECT + CONST),
    'X3 P0 loss tolerance 1e-2': (one("'p0_loss_tol': 1e-3,", "'p0_loss_tol': 1e-2,"), SELECT + CONST),
    'X4 P0 final usage max 4': (one("'p0_final_usage_max': 3,", "'p0_final_usage_max': 4,"), SELECT + CONST),
    'X5 I1 early window < 3000': (
        one("if r['step'] <= sel['i1_max_step']]", "if r['step'] < sel['i1_max_step']]"), I1),
    'X6 I1 dead fraction > instead of >=': (
        one("max(record['dead_relu']) >= sel['i1_dead_frac']", "max(record['dead_relu']) > sel['i1_dead_frac']"), I1),
    'X7 I1 z std <= instead of <': (
        one("record['z_std_mean'] < sel['i1_z_std_min']", "record['z_std_mean'] <= sel['i1_z_std_min']"), I1),
    'X8 I1 REFUTED ignores usage at 2000': (
        one("and first in full and max(full[first]) <= sel['i1_usage_max']", ""), I1),
    'X9 I2 tolerance <= instead of <': (
        one("/ sst < sel['i2_rel_tol']", "/ sst <= sel['i2_rel_tol']"), I2),
    'X10 (a2) rise uses >= instead of >': (
        one("high[l] > low[l] and", "high[l] >= low[l] and"), SELECT),
    'X11 (b) R2 >= 0 instead of > 0': (one("    out['b'] = final['r2'] > 0", "    out['b'] = final['r2'] >= 0"), SELECT),
    'X12 (c) non-strict residual decrease': (
        one("out['c'] = all(x > y for x, y in zip(norms, norms[1:]))",
            "out['c'] = all(x >= y for x, y in zip(norms, norms[1:]))"), SELECT),
    'X13 ESCALATE candidates ignore (c)': (
        one("CANDIDATE = ('a2', 'b', 'c')", "CANDIDATE = ('a2', 'b')"), SELECT),
    'X14 ESCALATE candidates ignore (b)': (
        one("CANDIDATE = ('a2', 'b', 'c')", "CANDIDATE = ('a2', 'c')"), SELECT),
    'X15 override accepts a blank source': (
        one(" and bool(override_source.strip())\n        if not ok:", "\n        if not ok:"), OVERRIDE),
    'X16 override accepts a lone option': (
        one("    if (override_arm is None) != (override_source is None):", "    if False:"), OVERRIDE),
    'X17 preflight boundary >= instead of >': (
        one("> sel['preflight_limit_seconds'] for v in VARIANTS)", ">= sel['preflight_limit_seconds'] for v in VARIANTS)"),
        PREFLIGHT),
    'X18 preflight compares with the T1 P0 only': (
        one("by[(v, arm)]['ms_per_step'] / by[(v, 'p0')]['ms_per_step']",
            "by[(v, arm)]['ms_per_step'] / by[('T1', 'p0')]['ms_per_step']"), PREFLIGHT),
    'X19 (a1) reads the last training log point instead of the canonical usage': (
        one("    used, norms = final['canonical_usage'], final['residual_norms']\n    low, high",
            "    used, norms = _usage_at(probe, 30000), final['residual_norms']\n    low, high"), SELECT),
    'X20 probe set ignores the expected config sha': (
        one("    if config_sha256 is not None and probes[0]['config_sha256'] != config_sha256:", "    if False:"), SCHEMAS),
    'X21 probe set allows different commits': (
        one("SHARED_KEYS = ('config_sha256', 'code_commit', 'content_file_sha256',",
            "SHARED_KEYS = ('config_sha256', 'content_file_sha256',"), SCHEMAS),
    'X22 SST divided by n-1': (
        one("    return float(((data - data.mean(0)) ** 2).sum()) / len(data)",
            "    return float(((data - data.mean(0)) ** 2).sum()) / (len(data) - 1)"), SST_I4),
    'X23 I4 range exclusive': (
        one("lo <= float(norms.mean()) <= hi", "lo < float(norms.mean()) < hi"), SST_I4),
    'X24 I4 ignores the zero-norm count': (
        one(" and int((norms == 0).sum()) == catalog_zero_norm_rows", ""), SST_I4),
    'X25 schema accepts uppercase arms': (
        one("or p['arm'] not in ARMS", "or p['arm'].lower() not in ARMS"), SCHEMAS),
    'X26 non-complete probe may carry a final': (
        one("    if complete != (p['final'] is not None):", "    if False:"), SCHEMAS),
    'X27 decision allows a preflight for STOP': (
        one("    if decision in order:\n        _check_preflight(d['preflight'])\n    elif d['preflight'] is not None:",
            "    if decision in order:\n        _check_preflight(d['preflight'])\n    elif False:"), SCHEMAS),
    'X28 override does not keep the listed arms': (
        one("'escalate_listed_arms': list(listed)}", "'escalate_listed_arms': []}"), OVERRIDE),
}

REQUIRED = [f'S{i}' for i in range(1, 15)]


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


if __name__ == '__main__':
    unittest.main()
