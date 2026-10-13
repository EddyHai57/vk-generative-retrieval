"""Actual optimizer/RNG/data-cursor checkpoint resume equivalence."""
import unittest,tempfile
from pathlib import Path
import torch
import random
import numpy as np
from vkgr import training
from vkgr.tiger import SIDTokens,TigerModel
from test_tiger import catalog


class TrainingTests(unittest.TestCase):
    def setup_model(self):
        training.seed_all(42);model=TigerModel(10,16,4,1,1,32)
        return model,torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)

    def test_interruption_resume_exact_parameters_optimizer_rng_cursor(self):
        rows=[{'history':[10],'target':20},{'history':[20,30],'target':40},{'history':[30],'target':10}]
        c=SIDTokens(catalog());m,o=self.setup_model();initial={k:v.clone() for k,v in m.state_dict().items()};s=training.ShuffledCursor(len(rows),42)
        for _ in range(5): training.train_step(m,o,c,rows,s,2)
        expected={k:v.clone() for k,v in m.state_dict().items()};self.assertTrue(any(not torch.equal(v,initial[k]) for k,v in expected.items()));nextindices=s.next(2);next_random=torch.rand(3)
        m,o=self.setup_model();s=training.ShuffledCursor(len(rows),42)
        for _ in range(2): training.train_step(m,o,c,rows,s,2)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'ours.pt';binding={'config':{'seed':42},'inputs':{'toy':'a'*64}}
            sha=training.save_checkpoint(p,m,o,s,2,binding,'2026-10-06T20:00:00+00:00')
            m2,o2=self.setup_model();s2=training.ShuffledCursor(len(rows),42)
            value=training.load_checkpoint(p,sha,m2,o2,s2,binding)
            self.assertEqual(value['step'],2);self.assertEqual(value['started_at_utc'],'2026-10-06T20:00:00+00:00')
            for _ in range(3): training.train_step(m2,o2,c,rows,s2,2)
            for k,v in expected.items(): self.assertTrue(torch.equal(v,m2.state_dict()[k]),k)
            self.assertEqual(nextindices,s2.next(2));self.assertTrue(torch.equal(next_random,torch.rand(3)))
            with self.assertRaisesRegex(ValueError,'binding'): training.load_checkpoint(p,sha,m2,o2,s2,{'config':{},'inputs':{}})
            with self.assertRaisesRegex(ValueError,'hash'): training.load_checkpoint(p,'b'*64,m2,o2,s2,binding)
            with self.assertRaises(FileExistsError): training.save_checkpoint(p,m,o,s,2,binding,value['started_at_utc'])

    def test_deadline_preserves_first_start_and_rejects_naive_time(self):
        self.assertEqual(training.stop_reason('2026-10-06T20:00:00+00:00','2026-10-07T02:16:21+00:00',2700,now='2026-10-06T20:46:00+00:00'),'time_limit')
        self.assertEqual(training.stop_reason('2026-10-06T20:00:00+00:00','2026-10-06T20:10:00+00:00',2700,now='2026-10-06T20:11:00+00:00'),'deadline')
        with self.assertRaises(ValueError): training.stop_reason('2026-10-06T20:00:00','2026-10-07T02:16:21+00:00',2700)

    def test_checkpoint_restores_advanced_global_rngs_after_disturbance(self):
        m,o=self.setup_model();s=training.ShuffledCursor(3,42)
        torch.rand(17);np.random.random(19);[random.random() for _ in range(23)]
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'rng.pt';binding={'toy':'rng'}
            sha=training.save_checkpoint(p,m,o,s,0,binding,'2026-10-06T20:00:00+00:00')
            expected=(torch.rand(5),np.random.random(5),[random.random() for _ in range(5)])
            m2,o2=self.setup_model();s2=training.ShuffledCursor(3,999)
            torch.rand(31);np.random.random(37);[random.random() for _ in range(41)]
            training.load_checkpoint(p,sha,m2,o2,s2,binding)
            self.assertTrue(torch.equal(expected[0],torch.rand(5)))
            self.assertTrue(np.array_equal(expected[1],np.random.random(5)))
            self.assertEqual(expected[2],[random.random() for _ in range(5)])
