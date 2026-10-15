"""P-HP-02 mutation tests: each deliberately wrong implementation must be caught by its named guard.

Same pattern as tests/test_php01_mutations.py: the guard passes on the real code, raises AssertionError under
``mock.patch.object`` on a php02 module-level seam, and passes again after the revert. Only the php02 namespace is patched;
the php01 module is never touched.
"""
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_feat02_mutations import detected  # noqa: E402
import test_php02_run as tr  # noqa: E402


class RegistryMutationTests(unittest.TestCase):
    def test_p1_temperature_compared_with_a_tolerance(self):
        """0.0500001 would be accepted as 0.05."""
        from vkgr import retrieval_php02_run as p2
        detected(self, p2, '_temperature_ok', lambda value, expected: math.isclose(value, expected, rel_tol=1e-4), [tr.guard_rejects_near_miss_temperature])

    def test_p2_equivalence_knobs_pass_without_synthetic(self):
        """T010 / T015 would become re-runnable real runs (their H1 / F4 runs already exist)."""
        from vkgr import retrieval_php02_run as p2
        detected(self, p2, '_equivalence_allowed', lambda synthetic: True, [tr.guard_equivalence_knobs_are_synthetic_only])

    def test_p3_a_third_temperature_is_registered(self):
        """A stray T006 / 0.06 entry in the real table."""
        from vkgr import retrieval_php02_run as p2
        detected(self, p2, 'TEMPERATURE_KNOBS', {'T005': 0.05, 'T007': 0.07, 'T006': 0.06}, [tr.guard_registry_is_exactly_two_real_temperatures])

    def test_p4_foreign_schema_accepted(self):
        """A php01 / FEAT-02 schema or task id slips through."""
        from vkgr import retrieval_php02_run as p2
        detected(self, p2, '_schema_ok', lambda value: True, [tr.guard_cross_schema_rejected])


class RunnerMutationTests(unittest.TestCase):
    def test_p5_train_step_reads_the_anchor_temperature(self):
        from vkgr import retrieval_feat02_run as fr, retrieval_php02_run as p2
        detected(self, p2, '_step', lambda model, optimizer, batch, data, stats, epoch, config, rng, pool, frequency:
                 fr.train_step(model, optimizer, batch, data, stats, epoch, {**config, 'temperature': p2.ANCHOR_VALUES['temperature']}, rng, pool, frequency),
                 [tr.guard_first_step_loss_uses_config_temperature_p02])

    def test_p6_binding_keeps_the_php01_task(self):
        from vkgr import retrieval_php01_run as php, retrieval_php02_run as p2
        detected(self, p2, 'binding_for', lambda data, config, rows, queries, model, device, inputs_binding:
                 php.binding_for3(data, config, rows, queries, model, device, inputs_binding), [tr.guard_manifest_binding_is_php02])


if __name__ == '__main__':
    unittest.main()
