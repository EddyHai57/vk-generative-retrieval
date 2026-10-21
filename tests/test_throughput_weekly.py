import unittest
from pathlib import Path
from vkgr.throughput_weekly import derive_weeks
from vkgr.throughput_config import Engineering


def parse_week(pair):
    week,rows=pair
    return [(week,i,row)for i,row in enumerate(rows)]


def stateful(rows):
    past={};out=[]
    for w,i,(u,item)in rows:
        out.append((w,i,u,tuple(past.get(u,[])),item));past.setdefault(u,[]).append(item)
    return out


class WeekTests(unittest.TestCase):
    def test_week_order_physical_rows_and_cross_week_history(self):
        weeks=[(0,[(1,3),(2,8),(1,4)]),(1,[(1,5)]),(2,[(2,9),(1,6)])]
        expected=stateful([r for w in weeks for r in parse_week(w)])
        for n in (1,2,8):
            self.assertEqual(derive_weeks(weeks,parse_week,stateful,Engineering(week_workers=n,prefetch=2)),expected)

    def test_duplicate_or_out_of_order_week_rejected(self):
        for weeks in ([(1,[]),(0,[])],[(0,[]),(0,[])],[(26,[])]):
            with self.assertRaises(ValueError):derive_weeks(weeks,parse_week,stateful,Engineering())
        with self.assertRaises(ValueError):derive_weeks([(0,Path('/synthetic/holdout/file'))],parse_week,stateful,Engineering())
        with self.assertRaises(ValueError):derive_weeks([(0,Path('/synthetic/week26/file'))],parse_week,stateful,Engineering())
        with self.assertRaises(ValueError):derive_weeks([(0,[])],parse_week,stateful,Engineering(week_workers=2,max_in_flight_bytes=4096),result_size_hint=lambda _:0)
        with self.assertRaises(MemoryError):derive_weeks([(0,[])],parse_week,stateful,Engineering(memory_limit_bytes=1))
        with self.assertRaises(ValueError):derive_weeks([(0,[])],parse_week,stateful,Engineering(),result_size_hint=lambda _:True)

    def test_test_path_components_are_rejected(self):
        for forbidden in (Path('/synthetic/test'), 'relative/test/part', r'C:\synthetic\test\part', '/synthetic/test.parquet'):
            with self.subTest(forbidden=forbidden), self.assertRaises(ValueError):
                derive_weeks([(0, forbidden)],parse_week,stateful,Engineering())


if __name__=='__main__':unittest.main()
