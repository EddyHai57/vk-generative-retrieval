"""FEAT-04b mutation tests (M56-M59, M83 in FEAT-002; later features append the rest of M41-M86): each deliberately wrong implementation must be
caught by its named guard. ``detected`` (single patch) comes from tests/test_feat02_mutations.py and is imported, never copied; ``detected_multi``
stacks several patches (M70a needs two at once).
Every guard is first run on the unmodified code (must pass), then under the patch (must raise AssertionError), then again after the revert (must pass).
"""
import contextlib
import sys
import unittest
from pathlib import Path
from unittest import mock
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_feat02_mutations import detected  # noqa: E402,F401
import test_feat04b_stats as ts  # noqa: E402


def detected_multi(case, patches, guards):
    """``patches`` is a list of ``(module_or_class, name, replacement)`` applied together."""
    for guard in guards:
        guard()
    with contextlib.ExitStack() as stack:
        for module, name, replacement in patches:
            stack.enter_context(mock.patch.object(module, name, replacement))
        for guard in guards:
            with case.assertRaises(AssertionError, msg=f'{[p[1] for p in patches]}: {guard.__name__} did not catch the mutation'):
                guard()
    for guard in guards:
        guard()


def reference_aggregate(first='min', count='all', zero='keep'):
    """A wrong re-implementation of ``item_stats.aggregate_rows`` with one switch turned: first_week from the last week (``first='max'``),
    exposure counted on like rows only (``count='likes'``), timespent == 0 rows dropped (``zero='drop'``) or refused (``zero='raise'``)."""
    from vkgr import item_stats as s

    def aggregate(partitions, catalog, durations):
        acc = {item: [0, None, 0, 0] for item in catalog}
        for week, events in partitions:
            if not 0 <= week <= s.STAT_CUTOFF_WEEK:
                raise ValueError('non-train week')
            for event in events:
                if event['timespent'] == 0 and zero == 'drop':
                    continue
                if event['timespent'] == 0 and zero == 'raise':
                    raise ValueError('timespent must be positive')
                entry = acc[event['item_id']]
                if count == 'all' or event['like']:
                    entry[0] += 1
                if first == 'max' or entry[1] is None:
                    entry[1] = week
                entry[2] += event['like']
                entry[3] += s.ratio_pm(event['timespent'], durations[event['item_id']])
        return acc
    return aggregate


class StatsMutationTests(unittest.TestCase):
    def test_m56_cutoff_relaxed_to_week_25(self):
        """The constructor accepts the validation week (a leak): the cutoff constant is read at call time."""
        from vkgr import item_stats
        detected(self, item_stats, 'STAT_CUTOFF_WEEK', 25, [ts.guard_stats_cutoff, ts.guard_stats_formula])

    def test_m57_first_week_takes_the_last_week(self):
        from vkgr import item_stats
        detected(self, item_stats, 'aggregate_rows', reference_aggregate(first='max'), [ts.guard_stats_formula])

    def test_m58_exposure_counts_like_rows_only(self):
        from vkgr import item_stats
        detected(self, item_stats, 'aggregate_rows', reference_aggregate(count='likes'), [ts.guard_stats_formula])

    def test_m59_ratio_neither_capped_nor_integer(self):
        from vkgr import item_stats
        detected(self, item_stats, 'ratio_pm', lambda timespent, duration: 1000 * timespent / duration, [ts.guard_stats_ratio_integer])
        detected(self, item_stats, 'ratio_pm', lambda timespent, duration: 1000 * timespent // duration, [ts.guard_stats_ratio_integer])    # integer but uncapped

    def test_m83_timespent_zero_rows_dropped_or_refused(self):
        from vkgr import item_stats
        detected(self, item_stats, 'aggregate_rows', reference_aggregate(zero='drop'), [ts.guard_stats_timespent_zero])
        detected(self, item_stats, 'aggregate_rows', reference_aggregate(zero='raise'), [ts.guard_stats_timespent_zero])

    def test_reference_aggregate_with_no_switch_equals_the_real_one(self):
        """The mutant factory is honest: with every switch off it reproduces the real aggregation (so a detection is due to the switch)."""
        from vkgr import item_stats
        self.assertEqual(reference_aggregate()(ts.world(), ts.CATALOG, ts.DURATIONS), item_stats.aggregate_rows(ts.world(), ts.CATALOG, ts.DURATIONS))

    def test_detected_multi_stacks_patches(self):
        from vkgr import item_stats
        detected_multi(self, [(item_stats, 'STAT_CUTOFF_WEEK', 25), (item_stats, 'aggregate_rows', reference_aggregate(first='max'))],
                       [ts.guard_stats_cutoff, ts.guard_stats_formula])


if __name__ == '__main__':
    unittest.main()
