"""M2 assignment, exact retrieval and complete synthetic replay."""
import math
import random
import tempfile
import unittest
from pathlib import Path
import torch
from vkgr import retrieval_feat02_run as fr
class RunTests(unittest.TestCase):
    def test_assignment_oracle(self):
        from vkgr.retrieval_seqm2_run import multi_interest_loss
        u=torch.tensor([[[1.,0.],[0.,1.],[-1.,0.],[0.,-1.]]],requires_grad=True)
        items=torch.eye(2); labels=torch.tensor([0]); mask=torch.ones(1,2,dtype=torch.bool)
        losses=torch.tensor([math.log1p(math.exp(-1)),math.log1p(math.e),math.log1p(math.e),math.log1p(math.exp(-1))])
        weights=torch.softmax(torch.tensor([10.,0.,-10.,0.]),0)
        loss,skip,_=multi_interest_loss(u,items,labels,mask,1.,torch.zeros(2),100)
        self.assertAlmostEqual(float(loss.detach()),float((weights*losses).sum()),places=6)
        loss,_,diag=multi_interest_loss(u,items,labels,mask,1.,torch.zeros(2),101)
        self.assertAlmostEqual(float(loss.detach()),float(losses[0]),places=6)
        self.assertEqual(diag['assignment_counts'],[1,0,0,0]); self.assertEqual(skip,0)
        tie=u.detach()[:,0:1].repeat(1,4,1)
        self.assertEqual(multi_interest_loss(tie,items,labels,mask,1.,None,101)[2]['assignment_counts'],[1,0,0,0])
        mask[:,1]=False
        self.assertEqual(multi_interest_loss(u,items,labels,mask,1.,None,1)[:2],(None,1))
    def test_zero_loss_stats(self):
        from vkgr import retrieval_seqm2_run as r
        from vkgr.retrieval_negatives import masked_ce_logq,target_frequency
        from vkgr.retrieval_training import SamplerStats
        d=fr.synthetic_data(); c={**r.default_config(),'batch_size':16,'max_epochs':2,'min_epochs':2}
        m=r._model(d,c,'cpu'); batch=fr._view(d,50)[0][:16]
        cols,labels,mask,neg,pool=r.b1._columns(batch,d,c,random.Random(1),None)
        h,s=fr._tensors2(batch,d['idmap'],50,c['feature'])
        q=r.b1._log_q(batch,neg,pool,c,target_frequency(d['rows']))
        u=m.users(h,s); items=m.items(cols)
        a=masked_ce_logq(u[:,0],items,labels,mask,.1,q)[0]
        b=r.multi_interest_loss(u,items,labels,mask,.1,q,1)[0]
        self.assertLessEqual(abs(float((a-b).detach())),1e-6)
        b.backward()
        self.assertGreater(float(m.gate.grad.abs()),0)
        for name,param in m.named_parameters():
            if name.split('.')[0] in ('order_table','layer','final_norm','attn_proj','attn_w'):
                self.assertEqual(int(torch.count_nonzero(param.grad)),0,name)
        stats=SamplerStats()
        rec=r.train_step(m,r.make_optimizer(m,c),batch,d,stats,0,c,random.Random(1),None,target_frequency(d['rows']),1)
        self.assertEqual(stats.drawn,16); self.assertEqual(rec['drawn'],16)
    def test_exact_ties(self):
        import numpy as np
        from vkgr.retrieval_seqm2_run import exact_max_topk
        u=np.array([[[1.,0.],[0.,1.],[-1.,0.],[0.,-1.]]],dtype='float32')
        items=np.array([[1.,0.],[0.,1.],[1.,0.],[-1.,0.],[0.,-1.]],dtype='float32')
        pred,win=exact_max_topk(u,items,[9,2,8,7,3],5,item_block=2)
        self.assertEqual(pred,[[2,3,7,8,9]]); self.assertEqual(win,[1])
    def test_run_verify_and_tamper(self):
        from vkgr import retrieval_seqm2_run as r
        d=fr.synthetic_data(); c={**r.default_config(),'batch_size':16,'max_epochs':2,'min_epochs':2}
        with self.assertRaises(ValueError): r.validate_config({**r.default_config(),'arm':'mean'})
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'run'; v=r.run_session(out,d,c)
            self.assertEqual(r.verify_run(out/'manifest.json',v['manifest_sha256'],d,c)['status'],'verified')
            self.assertIn('top1_winning_head_counts',v['evaluations'][0]['diagnostics'])
            self.assertIn('hard_window',v['evaluations'][0]['diagnostics'])
            self.assertIn('dependency_sha256',v['binding'])
            self.assertTrue((out/'predictions-best-g0.json').is_file())
            v.pop('manifest_sha256')
            v['evaluations'][v['best']['epoch']-1]['diagnostics']['branch_ratio_median']+=.1
            sha=r.b1._write_json_atomic(out/'manifest.json',v)
            with self.assertRaises(ValueError): r.verify_run(out/'manifest.json',sha,d,c)

class ContractTests(unittest.TestCase):
    def test_three_frozen_configs(self):
        import json
        from vkgr import retrieval_seqm2_run as r
        root=Path(__file__).resolve().parents[1]/'configs'/'seqm2'
        configs=[json.loads((root/f'p-seqm2-k4-s{seed}.json').read_text()) for seed in (42,2024,2025)]
        for seed,c in zip((42,2024,2025),configs):
            r.validate_config(c); self.assertEqual(c['seed'],seed)
        projected=[{k:v for k,v in c.items() if k not in ('seed','run_id')} for c in configs]
        self.assertTrue(all(c==projected[0] for c in projected))
    def test_logq_mask_and_detached_gradient(self):
        from vkgr.retrieval_seqm2_run import multi_interest_loss
        import torch.nn.functional as F
        u=torch.randn(3,4,5,generator=torch.Generator().manual_seed(7),requires_grad=True)
        items=torch.randn(4,5,generator=torch.Generator().manual_seed(8))
        labels=torch.tensor([0,1,2]); mask=torch.tensor([[1,0,1,1],[0,1,0,0],[1,1,1,0]],dtype=torch.bool)
        q=torch.tensor([-.4,-1.2,-2.3,-.7])
        loss,skips,_=multi_interest_loss(u,items,labels,mask,.1,q,1)
        got=torch.autograd.grad(loss,u)[0]
        v=F.normalize(items,dim=-1)
        oracle=[]
        for i in (0,2):
            normalized=F.normalize(u[i],dim=-1)
            weights=torch.softmax((normalized*v[labels[i]]).sum(-1)/.1,0).detach()
            per=[]
            for j in range(4):
                scores=normalized[j]@v.T/.1-q
                per.append(-scores[labels[i]]+torch.logsumexp(scores[mask[i]],0))
            oracle.append((weights*torch.stack(per)).sum())
        expected=torch.stack(oracle).mean()
        grad=torch.autograd.grad(expected,u)[0]
        self.assertEqual(skips,1); self.assertTrue(torch.allclose(got,grad,atol=1e-6,rtol=1e-5))
        self.assertTrue(torch.equal(got[1],torch.zeros_like(got[1])))
    def test_hard_window_zero_head_is_diagnostic(self):
        from vkgr import retrieval_seqm2_run as r
        records=[{'loss':1.,'assignment':{'optimizer_step':step,'assignment_counts':[2,0,0,0]}} for step in (100,101,200,201)]
        diag=r.hard_window(records)
        self.assertEqual(diag['head_counts'],[4,0,0,0]); self.assertEqual(diag['effective_interests'],1)
        self.assertTrue(diag['diagnostic_only'])

def control_audit_fixture(data_hashes=None):
    """Synthetic metadata only; no archived predictions or control data are read."""
    keys=('rows_sha256','queries_sha256','catalog_sha256','support_sha256','content_sha256')
    hashes=data_hashes or {k:'a'*64 for k in keys}
    controls={}; ledger={}
    for seed in (42,2024,2025):
        rid=f'p-seqm1-mean-s{seed}'; mp=f'/controls/{rid}/manifest.json'; pp=f'/controls/{rid}/predictions.json'
        controls[rid]={'run_id':rid,'seed':seed,'summary':{'run_id':rid},'verified':True,
            'manifest_path':mp,'manifest_sha256':'b'*64,'predictions_path':pp,'predictions_sha256':'c'*64,
            'source':{'git_commit':'d'*40,'files':{'src/vkgr/retrieval_seqm1.py':'e'*64}},
            'data_hashes':dict(hashes),'arm':'mean','best_epoch':3,
            'history_manifest_path':'/controls/history-manifest.json','history_manifest_sha256':'1'*64,
            'query_file_path':'/controls/validation-l50.json','query_file_sha256':'2'*64,
            'inputs':{k:('1'*64 if k=='history_manifest_sha256' else '5'*64) for k in
                ('prepared_sha','query_sha','catalog_sha','history_manifest_sha256','side_manifest_sha256','inputs_sha256')},
            'verifier':{'verified':True,'jobs':[{'verified':True,'run_id':rid+'-verify',
                'terminal':{'status':'succeeded','exit_code':0},
                'job_receipt':{'path':f'/controls/{rid}/verify.claimed.json','sha256':'3'*64,'target_binding_matches':True},
                'records':[{'status':'verified'}],
                'evidence':[{'path':f'/controls/{rid}/verify.stdout.log','sha256':'4'*64}]}]}}
        ledger.update({mp:'b'*64,pp:'c'*64,f'/controls/{rid}/verify.claimed.json':'3'*64,
            f'/controls/{rid}/verify.stdout.log':'4'*64})
        ledger.update({'/controls/history-manifest.json':'1'*64,'/controls/validation-l50.json':'2'*64})
    return {'schema':'seqm2-m1-controls-v1','seeds':[42,2024,2025],'controls':controls,'read_artifacts_sha256':ledger}

class ControlBindingTests(unittest.TestCase):
    def test_real_binding_requires_valid_audit_and_current_data(self):
        import copy
        import hashlib,json
        from unittest import mock
        from vkgr import retrieval_seqm2_run as r
        d=fr.synthetic_data(); c=r.default_config(); rows,qs=fr._view(d,50); m=r._model(d,c,'cpu')
        with mock.patch.object(r.fr,'binding_for2',return_value={k:'a'*64 for k in ('rows_sha256','queries_sha256','catalog_sha256','support_sha256','content_sha256')}):
            with self.assertRaises(ValueError): r.binding_for4({**d,'synthetic':False},c,rows,qs,m,'cuda',{})
            audit=control_audit_fixture()
            pin=hashlib.sha256(json.dumps(audit,sort_keys=True,allow_nan=False).encode()).hexdigest()
            inputs={**audit['controls']['p-seqm1-mean-s42']['inputs'],'m1_control_audit':{'sha256':pin,'audit':audit}}
            got=r.binding_for4({**d,'synthetic':False},c,rows,qs,m,'cuda',inputs)
            self.assertEqual(got['m1_control_audit_sha256'],pin)
            altered=copy.deepcopy(audit); altered['controls']['p-seqm1-mean-s42']['summary']['extra']='changed'
            changed=hashlib.sha256(json.dumps(altered,sort_keys=True,allow_nan=False).encode()).hexdigest()
            newer=r.binding_for4({**d,'synthetic':False},c,rows,qs,m,'cuda',{**inputs,'m1_control_audit':{'sha256':changed,'audit':altered}})
            self.assertNotEqual(r.b1._digest(got),r.b1._digest(newer))
            with self.assertRaises(ValueError):
                r.binding_for4({**d,'synthetic':False},c,rows,qs,m,'cuda',{**inputs,'m1_control_audit':{'sha256':pin,'audit':altered}})
            broken=copy.deepcopy(audit); broken['controls']['p-seqm1-mean-s42']['data_hashes']['queries_sha256']='f'*64
            badpin=hashlib.sha256(json.dumps(broken,sort_keys=True,allow_nan=False).encode()).hexdigest()
            with self.assertRaises(ValueError):
                r.binding_for4({**d,'synthetic':False},c,rows,qs,m,'cuda',{**inputs,'m1_control_audit':{'sha256':badpin,'audit':broken}})

    def test_control_metadata_and_ledger_reject_mutation(self):
        import copy
        from vkgr import retrieval_seqm2_run as r
        good=control_audit_fixture(); self.assertEqual(r.validate_control_audit(good),good)
        changes=[
            lambda a:a['controls'].pop('p-seqm1-mean-s2025'),
            lambda a:a['controls']['p-seqm1-mean-s42'].update(seed=2024),
            lambda a:a['controls']['p-seqm1-mean-s42'].update(verified=False),
            lambda a:a['controls']['p-seqm1-mean-s42'].update(predictions_sha256='f'*64),
            lambda a:a['controls']['p-seqm1-mean-s42']['source'].update(git_commit='bad'),
            lambda a:a['controls']['p-seqm1-mean-s42']['source'].update(files={}),
            lambda a:a['controls']['p-seqm1-mean-s42']['data_hashes'].pop('rows_sha256'),
            lambda a:a['read_artifacts_sha256'].clear(),
            lambda a:a.update(schema='other'),
            lambda a:a['controls']['p-seqm1-mean-s42'].update(arm='bi'),
            lambda a:a['controls']['p-seqm1-mean-s42'].update(best_epoch=0),
            lambda a:a['controls']['p-seqm1-mean-s42']['inputs'].pop('prepared_sha'),
            lambda a:a['controls']['p-seqm1-mean-s42']['verifier']['jobs'][0]['job_receipt'].update(target_binding_matches=False),
            lambda a:a['controls']['p-seqm1-mean-s42']['verifier']['jobs'][0].update(run_id='other-verify'),
            lambda a:a['controls']['p-seqm1-mean-s42']['verifier']['jobs'][0].update(records=[{'status':'succeeded'}]),
            lambda a:a['controls']['p-seqm1-mean-s42']['verifier']['jobs'][0]['evidence'][0].update(sha256='f'*64)]
        for edit in changes:
            bad=copy.deepcopy(good); edit(bad)
            with self.assertRaises(ValueError): r.validate_control_audit(bad)

    def test_real_cli_control_audit_pin_drift_rejected(self):
        import json,os
        from unittest import mock
        from vkgr import retrieval_seqm2_run as r
        from vkgr.data_audit import file_sha256
        with tempfile.TemporaryDirectory(prefix='m2-control-') as tmp:
            p=Path(tmp)/'audit.json'; p.write_text(json.dumps(control_audit_fixture(),sort_keys=True,allow_nan=False))
            with mock.patch.dict(os.environ,{'VKGR_M2_CONTROL_AUDIT':str(p),'VKGR_M2_CONTROL_AUDIT_SHA':file_sha256(p)}):
                binding=r.load_control_binding({'base':'keep'})
                self.assertEqual(binding['base'],'keep')
                self.assertEqual(binding['m1_control_audit']['sha256'],file_sha256(p))
                p.write_text(p.read_text()+' ')
                with self.assertRaises(ValueError): r.load_control_binding({})
            with mock.patch.dict(os.environ,{},clear=True):
                with self.assertRaises(ValueError): r.load_control_binding({})

    def test_cli_rejects_drift_before_train_entry(self):
        import json,os,io,contextlib
        from unittest import mock
        import test_seqm1_run as tr1
        from vkgr import retrieval_seqm2_run as r
        from vkgr.data_audit import file_sha256
        with tempfile.TemporaryDirectory(prefix='m2-cli-control-') as tmp:
            root=Path(tmp)
            spec=root/'inputs.json'; spec.write_text(json.dumps(tr1.CLITests.SPEC))
            cfg=root/'config.json'; cfg.write_text(json.dumps(r.default_config()))
            audit=root/'audit.json'; audit.write_text(json.dumps(control_audit_fixture(),sort_keys=True,allow_nan=False))
            pin=file_sha256(audit); audit.write_text(audit.read_text()+' ')
            argv=['train','--inputs',str(spec),'--inputs-sha',file_sha256(spec),
                '--config',str(cfg),'--config-sha',file_sha256(cfg),'--output',str(root/'run')]
            with mock.patch.dict(os.environ,{'VKGR_M2_CONTROL_AUDIT':str(audit),'VKGR_M2_CONTROL_AUDIT_SHA':pin}), \
                    mock.patch('torch.cuda.is_available',return_value=True), \
                    mock.patch.object(r.b1,'resource_guard',return_value=lambda:None), \
                    mock.patch.object(r.b1,'real_inputs',return_value=({},None,{})), \
                    mock.patch.object(r.fr,'apply_feat02',return_value=({},{})), \
                    mock.patch.object(r,'run_session',side_effect=AssertionError('must reject before train')), \
                    contextlib.redirect_stderr(io.StringIO()) as errors:
                self.assertEqual(r.main(argv),2)
                self.assertIn('external hash mismatch',errors.getvalue())
            self.assertFalse((root/'run').exists())
