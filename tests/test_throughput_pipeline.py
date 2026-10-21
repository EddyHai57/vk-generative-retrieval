import json
from pathlib import Path
import tempfile
import unittest
from dataclasses import replace
import numpy as np
from vkgr.throughput_config import Engineering
from vkgr.throughput_pipeline import Toolchain
from test_throughput_sequences import EVENTS,REFS
from vkgr.throughput_sequences import position,SequenceStore


def columns():
    return dict(users=np.array([e['user_id']for e in EVENTS]),items=np.array([e['item_id']for e in EVENTS]),
        positions=np.array([position(e['position'])for e in EVENTS]),likes=np.array([e['like']for e in EVENTS]),
        side=np.array([e['side']for e in EVENTS]),ctx=np.array([e['ctx']for e in EVENTS]),
        references=[(u,position(p))for u,p in REFS])


class PipelineTests(unittest.TestCase):
    def test_disabled_returns_original_and_records_actual_config(self):
        with tempfile.TemporaryDirectory()as t:
            root=Path(t).resolve();identity={'seed':42,'data_sha':'fixture'}
            pipeline=Toolchain(Engineering(),identity,root/'engineering.json')
            original=object()
            self.assertIs(pipeline.examples(original,None,run_id='off'),original)
            receipt=json.loads((root/'engineering.json').read_text())
            self.assertEqual(receipt['science'],identity)
            self.assertEqual(receipt['engineering']['sha256'],Engineering().identity())
            with self.assertRaises(FileExistsError):Toolchain(Engineering(),identity,root/'engineering.json')

    def test_mmap_requires_sequence_instead_of_silent_ignore(self):
        with tempfile.TemporaryDirectory()as t:
            with self.assertRaises(ValueError):
                Toolchain(Engineering(use_mmap=True),{},Path(t)/'bad.json')
            self.assertFalse((Path(t)/'bad.json').exists())

    def test_mmap_switch_uses_new_exclusive_store_and_preserves_rows(self):
        with tempfile.TemporaryDirectory()as t:
            root=Path(t).resolve();cfg=Engineering(sequence_enabled=True,use_mmap=True,temp_dir=str(root/'tmp'))
            pipeline=Toolchain(cfg,{},root/'engineering.json')
            store=pipeline.examples(None,columns(),run_id='fixture')
            self.assertEqual(store.rows([2,0]),SequenceStore.from_events(EVENTS,REFS).rows([2,0]))
            self.assertIsNotNone(store.root)


if __name__=='__main__':unittest.main()
