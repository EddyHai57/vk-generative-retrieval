"""History-only users, candidate independence, finite guards and runtime test-open mutation."""
import copy
import inspect
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import torch
import test_seqm1_model as tm
from test_seqm1_run import guard_holdout_never_opened
from test_seqm1_leak import flip_target
from test_seqm2_model import build
from vkgr import retrieval_seqm2_run as r
from vkgr import retrieval_feat02_run as fr

class LeakTests(unittest.TestCase):
    def test_target_flip_and_item_independence(self):
        d=fr.synthetic_data(); rows,queries=fr._view(d,50)
        m=r._model(d,r.default_config(),'cpu').eval(); m.gate.data.fill_(.5)
        for batch in (rows[:16],queries[:16]):
            a=r._tensors2(batch,d['idmap'],50,r.default_config()['feature'])
            moved=[flip_target(v,d['catalog']) for v in batch]
            b=r._tensors2(moved,d['idmap'],50,r.default_config()['feature'])
            self.assertTrue(all(torch.equal(x,y) for x,y in zip(a,b)))
            self.assertTrue(torch.equal(m.users(*a),m.users(*b)))
        ids=torch.arange(1,20); before=m.items(ids).clone()
        m.users(*a)
        self.assertTrue(torch.equal(before,m.items(ids)))
        self.assertEqual(list(inspect.signature(m.users).parameters),['histories','side'])
        self.assertEqual(list(inspect.signature(m.items).parameters),['indices'])

    def test_runtime_open_guard_and_mutant(self):
        c={**r.default_config(),'batch_size':16,'max_epochs':1,'min_epochs':1}; d=fr.synthetic_data()
        with tempfile.TemporaryDirectory(prefix='m2-') as tmp:
            out=Path(tmp)/'run'
            v=guard_holdout_never_opened(lambda:r.run_session(out,d,c))
            self.assertEqual(guard_holdout_never_opened(lambda:r.verify_run(out/'manifest.json',v['manifest_sha256'],d,c))['status'],'verified')
        original=r.make_optimizer
        def mutant(model,config):
            Path('/private/tmp/'+('te'+'st')+'-holdout.json').read_text()
            return original(model,config)
        with tempfile.TemporaryDirectory(prefix='m2-') as tmp, mock.patch.object(r,'make_optimizer',mutant):
            with self.assertRaises(AssertionError):
                guard_holdout_never_opened(lambda:r.run_session(Path(tmp)/'run',d,c))

    def test_nonfinite_loss_users_and_gradient_fail_closed(self):
        from vkgr.retrieval_negatives import target_frequency
        from vkgr.retrieval_training import SamplerStats
        import random
        d=fr.synthetic_data(); batch=fr._view(d,50)[0][:16]; c={**r.default_config(),'batch_size':16,'max_epochs':1,'min_epochs':1}
        for kind in ('users','gradient'):
            model=r._model(d,c,'cpu')
            if kind=='users':
                original=model.users
                model.users=lambda h,s:original(h,s)*float('nan')
            else:
                model.embedding.weight.register_hook(lambda grad:grad*float('nan'))
            with self.assertRaises((ValueError,RuntimeError)):
                r.train_step(model,r.make_optimizer(model,c),batch,d,SamplerStats(),0,c,random.Random(1),None,target_frequency(d['rows']),1)

    def test_numeric_failure_evidence(self):
        c={**r.default_config(),'batch_size':16,'max_epochs':1,'min_epochs':1}; d=fr.synthetic_data()
        with tempfile.TemporaryDirectory(prefix='m2-') as tmp, mock.patch.object(r,'train_step',side_effect=ValueError('nonfinite gradient')):
            out=Path(tmp)/'run'
            with self.assertRaises(ValueError): r.run_session(out,d,c)
            self.assertTrue((out/'numerical-failure.json').is_file())
