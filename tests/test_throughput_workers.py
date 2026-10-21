import random
import time
import unittest
from vkgr.throughput_workers import ordered_map, assemble_rows, cursor_rows
from vkgr.throughput_config import Engineering
from test_throughput_sequences import EVENTS, REFS
from vkgr.throughput_sequences import SequenceStore


def delayed(value):
    time.sleep((3-value%3)*.005)
    return value*value


def stochastic(value):
    return random.random()


def expanded(value):
    return b'x'*4096


class WorkerTests(unittest.TestCase):
    def test_output_bound_is_enforced_not_only_input_bytes(self):
        with self.assertRaises((MemoryError,ValueError)):
            list(ordered_map(expanded,[1],workers=2,max_in_flight_bytes=2048,result_size_hint=lambda _:64))
    def test_prefetch_does_not_advance_live_cursor_future_rng(self):
        class Cursor:
            def __init__(self):self.offset=0;self.size=3;self.rng=random.Random(42);self.order=[2,0,1]
            def next(self,n):
                end=min(self.size,self.offset+n);out=self.order[self.offset:end];self.offset=end;self.rng.random();return out
        store=SequenceStore.from_events(EVENTS,REFS)
        for n in (1,2,8):
            live,old=Cursor(),Cursor()
            for rows in cursor_rows(store,live,Engineering(assembly_workers=n,prefetch=2),batch_size=2):
                ids=old.next(2)
                self.assertEqual([r['example_id']for r in rows],ids)
                self.assertEqual(live.offset,old.offset);self.assertEqual(live.rng.getstate(),old.rng.getstate())

    def test_worker_counts_order_and_parent_rng(self):
        before=random.getstate()
        for n in (1,2,8):
            self.assertEqual(list(ordered_map(delayed,range(9),workers=n,prefetch=2)),[i*i for i in range(9)])
        self.assertEqual(before,random.getstate())

    def test_sequence_assembly_workers_and_tail(self):
        s=SequenceStore.from_events(EVENTS,REFS)
        batches=[[2,0],[1]]
        for n in (1,2,8):
            self.assertEqual(list(assemble_rows(s,batches,Engineering(assembly_workers=n,prefetch=2))),[s.rows(x)for x in batches])

    def test_random_worker_rejected(self):
        with self.assertRaises(ValueError):list(ordered_map(stochastic,[1],workers=2))


if __name__=='__main__':unittest.main()
