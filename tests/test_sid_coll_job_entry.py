"""Unified stage evidence and SGNS verification negative oracles."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import subprocess
from unittest.mock import patch
import unittest
import numpy as np
from scripts import sid_coll_job_entry as entry
from vkgr.sid_coll_sgns import fit_sgns
from vkgr.data_audit import file_sha256

class EntryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ids=np.arange(1,7,dtype=np.uint32)
        cls.hist={2:[1,2]*40,1:[3,4]*40,3:[5]}
        cls.content=np.random.default_rng(19).normal(size=(6,64)).astype('float32')
        cls.vectors,cls.audit=fit_sgns(cls.ids,cls.hist,cls.content,engineering=True)

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve()
        self.spec={'schema':'coll-stage-spec-v1','variant':'C1','engineering':True,'commit':'a'*40,
                   'source_files':{'scripts/sid_coll_job_entry.py':'b'*64},'refs':{},
                   'stages':{'input_verify':{'run_id':'synthetic-verify','parameters':{}}}}

    def producer(self):
        return entry.write_c1_producer(self.root/'producer',self.ids,self.hist,self.content,
                   self.vectors,self.audit,self.spec,{'raw-train': 'c'*64})

    def verify(self,pin):
        return entry.verify_c1_producer(self.root/'producer'/'producer-manifest.json',pin,
                   self.ids,self.hist,self.content,self.spec,{'raw-train':'c'*64})

    def test_dynamic_refs_bind_frozen_producer_and_static_cannot_override(self):
        artifact=self.root/'vector.npy';artifact.write_bytes(b'fixed synthetic')
        pin=file_sha256(artifact)
        receipt={'schema':'coll-stage-receipt-v1','stage':'input','status':'produced','run_id':'producer1',
          'variant':'C1','binding':{'commit':'a'*40,'spec_sha256':'d'*64},
          'proofs':{'vectors':{'path':str(artifact),'sha256':pin}}}
        previous=self.root/'receipt.json';previous.write_text(json.dumps(receipt))
        spec=copy.deepcopy(self.spec);spec['refs']={'vectors':{'path':str(artifact),'sha256':'DYNAMIC',
           'from':{'receipt_path':str(previous),'stage':'input','run_id':'producer1','spec_sha256':'d'*64,'proof':'vectors'}}}
        resolved=entry.resolve_refs(spec,{'vectors':pin})
        self.assertEqual(resolved['refs']['vectors'],receipt['proofs']['vectors'])
        self.assertEqual(spec['refs']['vectors']['sha256'],'DYNAMIC')
        with self.assertRaises(ValueError):entry.resolve_refs(spec,{'vectors':'0'*64})
        with self.assertRaises(ValueError):entry.resolve_refs(resolved,{'vectors':pin})
        receipt['binding']['commit']='e'*40;previous.write_text(json.dumps(receipt))
        with self.assertRaises(ValueError):entry.resolve_refs(spec,{'vectors':pin})

    def test_c2_pending_source_lock_is_new_and_parent_immutable(self):
        template={'schema':'coll-c2-source-lock-v1','train_sequence_sha256':'PENDING_TRAIN_SEQUENCE','checkpoint_sha256':'e'*64}
        actual=entry.materialize_source_lock(template,'f'*64,'d'*64,self.root)
        self.assertEqual(json.loads(actual.read_text())['train_sequence_sha256'],'d'*64)
        self.assertEqual(template['train_sequence_sha256'],'PENDING_TRAIN_SEQUENCE')
        with self.assertRaises(FileExistsError):entry.materialize_source_lock(template,'f'*64,'d'*64,self.root)
        bad={**template,'train_sequence_sha256':'UNKNOWN'}
        with self.assertRaises(ValueError):entry.materialize_source_lock(bad,'f'*64,'d'*64,self.root)

    def raw_fixture(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        records=[]
        for week in range(25):
            relative=f'subsamples/ur0.01_ir0.01/train/week_{week:02d}.parquet'
            path=self.root/relative;path.parent.mkdir(parents=True,exist_ok=True)
            users=[];items=[]
            if week==0:
                for u,events in self.hist.items():users.extend([u]*len(events));items.extend(events)
            pq.write_table(pa.table({'user_id':pa.array(users,type=pa.uint32()),'item_id':pa.array(items,type=pa.uint32())}),path)
            records.append({'path':relative,'sha256':file_sha256(path),'size_bytes':path.stat().st_size})
        audit=self.root/'train-audit.json';audit.write_text(json.dumps({'train_files':records}))
        plan=self.root/'s0-plan.json';plan.write_text(json.dumps({'raw_root':str(self.root),'train_audit':entry._proof(audit)}))
        catalog=self.root/'catalog.json';catalog.write_text(json.dumps(self.ids.astype(int).tolist()))
        content=self.root/'content.npy';np.save(content,np.vstack([np.zeros((1,64),dtype='float32'),self.content]))
        return {'s0_plan':entry._proof(plan),'catalog':entry._proof(catalog),'content':entry._proof(content)}

    def test_input_verify_queue_stage_end_to_end_and_output_exclusion(self):
        spec=copy.deepcopy(self.spec);spec['refs']=self.raw_fixture()
        spec['commit']=subprocess.run(['git','-C',str(entry.ROOT),'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
        spec['source_files']={entry.SCRIPT:file_sha256(entry.ROOT/entry.SCRIPT)}
        spec['stages']={'input':{'run_id':'synthetic-input','parameters':{}},'input_verify':{'run_id':'synthetic-verify','parameters':{}}}
        path=self.root/'frozen.json';path.write_text(json.dumps(spec))
        produced=entry.execute('input',path,file_sha256(path),self.root/'input-out')
        self.assertEqual(produced['status'],'produced')
        spec['refs']['producer_manifest']=produced['proofs']['producer_manifest']
        # Different stage spec is independently frozen; producer identity is source+data, not this spec SHA.
        verify_path=self.root/'verify-spec.json';verify_path.write_text(json.dumps(spec))
        verified=entry.execute('input_verify',verify_path,file_sha256(verify_path),self.root/'verify-out')
        self.assertEqual(verified['status'],'verified')
        self.assertEqual(verified['raw_proof']['gradient_replay'],False)
        self.assertEqual(verified['binding']['commit'],spec['commit'])
        from vkgr.sid_coll_run import expected_config
        fit_cfg=self.root/'fit-config.json';fit_cfg.write_text(json.dumps(expected_config('C1')))
        spec['refs'].update({'input_manifest':verified['proofs']['input_manifest'],'vectors':produced['proofs']['vectors'],'fit_config':entry._proof(fit_cfg)})
        spec['stages']={'sid_fit':{'run_id':'synthetic-sid-fit','parameters':{}}}
        fit_spec=self.root/'sid-fit-spec.json';fit_spec.write_text(json.dumps(spec))
        fitted=entry.execute('sid_fit',fit_spec,file_sha256(fit_spec),self.root/'sid-fit')
        spec['refs']['sid_manifest']=fitted['proofs']['sid_manifest']
        spec['stages']={'sid_verify':{'run_id':'synthetic-sid-verify','parameters':{}}}
        sid_verify_spec=self.root/'sid-verify-spec.json';sid_verify_spec.write_text(json.dumps(spec))
        sid_verified=entry.execute('sid_verify',sid_verify_spec,file_sha256(sid_verify_spec),self.root/'sid-verify')
        self.assertTrue(sid_verified['raw_proof']['mapping_replay'])
        self.assertIn('mapping_diagnostics',sid_verified['proofs'])

        with self.assertRaises(FileExistsError):entry.execute('input_verify',verify_path,file_sha256(verify_path),self.root/'verify-out')

    def frozen_spec(self,stage,refs,params=None):
        spec=copy.deepcopy(self.spec);spec['refs']=refs
        spec['commit']=subprocess.run(['git','-C',str(entry.ROOT),'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
        spec['source_files']={entry.SCRIPT:file_sha256(entry.ROOT/entry.SCRIPT)}
        spec['stages']={stage:{'run_id':'synthetic-'+stage,'parameters':params or {}}}
        path=self.root/(stage+'-spec.json');path.write_text(json.dumps(spec));return path

    def test_input_preflight_is_readonly_no_training_or_checkpoint_load(self):
        refs=self.raw_fixture();path=self.frozen_spec('input_preflight',refs)
        with patch('vkgr.sid_coll_sgns.fit_sgns',side_effect=AssertionError('training forbidden')), patch('torch.load',side_effect=AssertionError('checkpoint forbidden')):
            receipt=entry.execute('input_preflight',path,file_sha256(path),self.root/'pf-out')
        self.assertEqual(receipt['status'],'verified')
        self.assertFalse(receipt['raw_proof']['training'])
        self.assertFalse(receipt['raw_proof']['checkpoint_read'])
        self.assertEqual(receipt['raw_proof']['kind'],'read_only_input_preflight')
        self.assertEqual(len(receipt['raw_proof']['source_physical_sha256']),29)

    def test_parent_helper_exact_statuses_and_parameter_adapter(self):
        from scripts import sid_coll_cuda_check as helper
        contracts={'preflight':('preflight','preflight_ok'),'cuda':('rehearse','cuda_engineering_verified'),'dry':('dry','dry_run_ok')}
        for stage,(method,status) in contracts.items():
            with self.subTest(stage=stage):
                path=self.frozen_spec(stage,{}, {'root':str(self.root),'seed':42})
                seen=[]
                def forward(spec,out):
                    seen.append((spec['parameters'],out.name));return {'status':status}
                with patch.object(helper,method,side_effect=forward):
                    result=entry.execute(stage,path,file_sha256(path),self.root/(stage+'-ok'))
                self.assertEqual(result['status'],'verified')
                self.assertEqual(seen,[({'root':str(self.root),'seed':42},'detail')])
                with patch.object(helper,method,return_value={'status':'verified'}):
                    with self.assertRaises(ValueError):entry.execute(stage,path,file_sha256(path),self.root/(stage+'-bad'))

    def test_actual_seed_config_sha_binding_at_train_boundary(self):
        from vkgr import generative_coll_run as gr
        cfg=gr.default_config();cfg_path=self.root/'g-config.json';cfg_path.write_text(json.dumps(cfg))
        dummy=self.root/'dummy.json';dummy.write_text('{}')
        refs={role:entry._proof(dummy) for role in ('prepared','history','sid_manifest','content','input_manifest','vectors')}
        refs['config42']=entry._proof(cfg_path)
        path=self.frozen_spec('train',refs,{'root':str(self.root),'seed':42})
        def fake_train(*args,**kwargs):
            self.assertEqual(args[-2]['seed'],42)
            args[-1].mkdir();(args[-1]/'manifest.json').write_text(json.dumps({'status':'complete'}))
            return {'status':'complete'}
        with patch.object(gr,'run',side_effect=fake_train):
            result=entry.execute('train',path,file_sha256(path),self.root/'train-ok')
        self.assertEqual(result['binding']['config_sha256'],file_sha256(cfg_path))
        cfg['seed']=2024;cfg_path.write_text(json.dumps(cfg));refs['config42']=entry._proof(cfg_path)
        path=self.frozen_spec('train',refs,{'root':str(self.root),'seed':42})
        with patch.object(gr,'run',side_effect=AssertionError('must reject before training')):
            with self.assertRaises(ValueError):entry.execute('train',path,file_sha256(path),self.root/'train-bad')

    def test_raw_reader_retains_actual_user_identity_and_25_hashes(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        records=[]
        for week in range(25):
            relative=f'subsamples/ur0.01_ir0.01/train/week_{week:02d}.parquet'
            path=self.root/relative;path.parent.mkdir(parents=True,exist_ok=True)
            values={'user_id':[9,2,9,2],'item_id':[1,3,2,4]} if week==0 else {'user_id':[],'item_id':[]}
            pq.write_table(pa.table({k:pa.array(v,type=pa.uint32()) for k,v in values.items()}),path)
            records.append({'path':relative,'sha256':file_sha256(path),'size_bytes':path.stat().st_size})
        histories,physical=entry.load_histories({'raw_root':str(self.root)},{'train_files':records},self.ids)
        self.assertEqual(histories,{9:[1,2],2:[3,4]})
        self.assertEqual(len(physical),25)
        records[24]['sha256']='0'*64
        with self.assertRaises(ValueError):entry.load_histories({'raw_root':str(self.root)},{'train_files':records},self.ids)

    def test_verified_manifest_and_no_retraining(self):
        p=self.producer();accepted,proof=self.verify(file_sha256(p))
        self.assertEqual(accepted['status'],'accepted')
        self.assertEqual(accepted['schema'],'coll-input-v1')
        self.assertEqual(proof['status'],'verified')
        self.assertIn('health_recomputed',proof['checks'])
        self.assertFalse(proof['gradient_replay'])

    def test_pinned_producer_tamper_rejected(self):
        p=self.producer();pin=file_sha256(p)
        p.write_bytes(p.read_bytes()+b' ')
        with self.assertRaises(ValueError):self.verify(pin)

    def test_sgns_contract_mutations_rejected(self):
        for key,value in [('loss_reduction','mean'),('momentum',.1),('gradient_clipping',1.)]:
            with self.subTest(key=key):
                audit=copy.deepcopy(self.audit);audit['training_contract'][key]=value
                with self.assertRaises(ValueError):entry.verify_c1_audit(self.ids,self.hist,self.content,self.vectors,audit,engineering=True)
        for key,value in [('epochs_completed',4),('processed_positive_pairs',1)]:
            audit=copy.deepcopy(self.audit);audit[key]=value
            with self.assertRaises(ValueError):entry.verify_c1_audit(self.ids,self.hist,self.content,self.vectors,audit,engineering=True)
        audit=copy.deepcopy(self.audit);audit['learning_rates']['final']=.025
        with self.assertRaises(ValueError):entry.verify_c1_audit(self.ids,self.hist,self.content,self.vectors,audit,engineering=True)
        audit=copy.deepcopy(self.audit);audit['health']['difference']=-1
        with self.assertRaises(ValueError):entry.verify_c1_audit(self.ids,self.hist,self.content,self.vectors,audit,engineering=True)

    def test_bad_fallback_and_formal_small_catalog_rejected(self):
        wrong=self.vectors.copy();wrong[4]=np.roll(wrong[4],1)
        with self.assertRaises(ValueError):entry.verify_c1_audit(self.ids,self.hist,self.content,wrong,self.audit,engineering=True)
        with self.assertRaises(ValueError):entry.verify_c1_audit(self.ids,self.hist,self.content,self.vectors,self.audit,engineering=False)

    def test_source_and_checkpoint_boundary(self):
        p=self.producer();m=json.loads(p.read_text());m['checkpoint_sha256']='f'*64
        p.write_text(json.dumps(m))
        with self.assertRaises(ValueError):self.verify(file_sha256(p))
        m['checkpoint_sha256']=None;m['commit']='e'*40;p.write_text(json.dumps(m))
        with self.assertRaises(ValueError):self.verify(file_sha256(p))

    def test_exclusive_output_and_safe_ref(self):
        self.producer()
        with self.assertRaises(FileExistsError):self.producer()
        payload=self.root/'payload.json';payload.write_text('{}')
        self.assertEqual(entry.checked_ref({'path':str(payload),'sha256':file_sha256(payload)}),payload)
        with self.assertRaises(ValueError):entry.checked_ref({'path':str(payload),'sha256':'0'*64})
        forbidden=self.root/'old.pt';forbidden.write_bytes(b'not opened')
        with self.assertRaises(ValueError):entry.checked_ref({'path':str(forbidden),'sha256':file_sha256(forbidden)})

if __name__=='__main__':unittest.main()
