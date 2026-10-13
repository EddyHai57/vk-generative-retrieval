"""Real adapter boundaries: local controlled inputs never authorize a server job."""
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import tempfile
import time
import unittest

try:
    from vkgr import retrieval_sasrec_remote as remote
except ImportError:
    remote = None

class PresenceTests(unittest.TestCase):
    def test_real_adapter_exists(self):
        self.assertIsNotNone(remote, 'real remote adapter missing; synthetic operation is insufficient')

@unittest.skipIf(remote is None, 'real adapter missing')
class RemoteBoundaryTests(unittest.TestCase):
    def test_synthetic_manifest_cannot_authorize_remote(self):
        from vkgr.retrieval_sasrec_operation import synthetic_operation_manifest
        for flag in (True, False):
            value = synthetic_operation_manifest('pilot')
            value['synthetic_local'] = flag
            with self.assertRaises(ValueError):
                remote.validate_remote_manifest(value)

    def test_hash_and_symlink_drift_fail_before_read_consumer(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve(); p=root/'input.json';p.write_bytes(b'{"x":1}')
            pin=hashlib.sha256(p.read_bytes()).hexdigest()
            self.assertEqual(remote.read_bound(p,pin,1024),b'{"x":1}')
            p.write_bytes(b'{"x":2}')
            with self.assertRaises(ValueError):remote.read_bound(p,pin,1024)
            link=root/'link';link.symlink_to(p)
            with self.assertRaises(ValueError):remote.read_bound(link,pin,1024)
            with self.assertRaises(ValueError):remote.read_bound(p,hashlib.sha256(p.read_bytes()).hexdigest(),2)

    def test_immutable_output_cannot_overwrite_prior_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder).resolve()/'receipt.json'
            first=remote.write_once(p,{'state':'first'})
            with self.assertRaises(FileExistsError):remote.write_once(p,{'state':'second'})
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),first)
            self.assertEqual(json.loads(p.read_bytes()),{'state':'first'})

class BudgetTests(unittest.TestCase):
    @unittest.skipIf(remote is None,'real adapter missing')
    def test_attempts_do_not_reset_clock_or_disk(self):
        budget=remote.Budget('pilot',started_at=100,deadline=14500,disk_cap=20*1024**3,attempts=1)
        self.assertEqual(budget.remaining_seconds(400),14100)
        next_budget=budget.next_attempt(now=400,reason='transport_before_step0',optimizer_steps=0)
        self.assertEqual(next_budget.started_at,100)
        self.assertEqual(next_budget.deadline,14500)
        self.assertEqual(next_budget.remaining_seconds(400),14100)
        with self.assertRaises(ValueError):next_budget.next_attempt(now=410,reason='transport_before_step0',optimizer_steps=0)
        for reason,steps in [('quality',0),('resource_stop',0),('transport_before_step0',1),('nonfinite',0)]:
            with self.assertRaises(ValueError):budget.next_attempt(now=400,reason=reason,optimizer_steps=steps)

    @unittest.skipIf(remote is None,'real adapter missing')
    def test_atomic_old_new_and_verifier_reserves_must_fit(self):
        b=remote.Budget('formal',started_at=100,deadline=50500,disk_cap=32*1024**3,attempts=1)
        b.require_stage(now=200,stage_seconds=10,finish_seconds=20,used_bytes=100,next_atomic_bytes=40,closure_bytes=10,free_fractions={'ram':.3,'vram':.3,'disk':.3})
        for changes in [{'now':50475},{'used_bytes':32*1024**3-49},{'free_fractions':{'ram':.19,'vram':.3,'disk':.3}},{'stage_seconds':float('nan')}]:
            kwargs=dict(now=200,stage_seconds=10,finish_seconds=20,used_bytes=100,next_atomic_bytes=40,closure_bytes=10,free_fractions={'ram':.3,'vram':.3,'disk':.3});kwargs.update(changes)
            with self.assertRaises(ValueError):b.require_stage(**kwargs)

    @unittest.skipIf(remote is None,'real adapter missing')
    def test_authorized_ceiling_cannot_be_expanded(self):
        for args in [('pilot',100,14501,20*1024**3),('formal',100,50501,32*1024**3),('pilot',100,14500,20*1024**3+1)]:
            with self.assertRaises(ValueError):remote.Budget(args[0],started_at=args[1],deadline=args[2],disk_cap=args[3],attempts=1)

@unittest.skipIf(remote is None,'real adapter missing')
class ActualMeasurementTests(unittest.TestCase):
    def test_measurement_runs_work_and_hashes_actual_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();artifact=root/'actual.bin';events=[]
            def work():artifact.write_bytes(b'actual computation');return 17
            result,measurement=remote.measure_action('checkpoint_write',work,artifacts=[artifact],sink=events.append)
            self.assertEqual(result,17)
            self.assertEqual(measurement['bytes'],18)
            self.assertGreaterEqual(measurement['seconds'],0)
            self.assertEqual(measurement['artifacts'][0]['sha256'],hashlib.sha256(b'actual computation').hexdigest())
            self.assertEqual(events,[measurement])
            with self.assertRaises(ValueError):remote.measure_action('train_step',{'seconds':1,'bytes':1},artifacts=[],sink=events.append)

    def test_failed_work_never_emits_success_measurement(self):
        events=[]
        def work():raise RuntimeError('actual failure')
        with self.assertRaises(RuntimeError):remote.measure_action('train_step',work,artifacts=[],sink=events.append)
        self.assertEqual(events,[])

if __name__=='__main__':unittest.main()

@unittest.skipIf(remote is None,'real adapter missing')
class ModelArtifactTests(unittest.TestCase):
    def test_best_checkpoint_keeps_early_epoch_and_replays_fresh_once(self):
        import torch
        from vkgr.retrieval_sasrec import SASRecIDTwoTower
        from vkgr import retrieval_sasrec_plan as plan
        from vkgr.retrieval_training import EpochCursor
        torch.manual_seed(42)
        model=SASRecIDTwoTower(torch.zeros((5,64)),[1,2,3,4]);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4)
        rows=[{'history':[1,2],'target':3,'past_likes':[1,2]},{'history':[2,1],'target':4,'past_likes':[1,2]}]
        batch_rows=[{'epoch':e,'epoch_end':True,'rows':rows} for e in range(1,5)]
        slot=next(s for s in plan.FORMAL_SLOTS if s['slot_id']=='P4H-S02')
        calls=[]
        def evaluate(current,epoch):
            current.eval();calls.append(epoch)
            with torch.no_grad():x=current.items(torch.tensor([1,2,3,4])).cpu().numpy().tobytes()
            return {'epoch':epoch,'metrics':{'all':{'recall@50':.5 if epoch==1 else .4,'ndcg@50':.2}},'prediction_sha256':hashlib.sha256(x).hexdigest()}
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();state=remote.train_model(model,optimizer,batch_rows,{1:1,2:2,3:3,4:4},slot,evaluate=evaluate,output=root,binding={'input_sha256':'a'*64},gate=lambda:None)
            self.assertEqual(state['public']['selection']['best_epoch'],1)
            self.assertEqual(state['public']['completed_epochs'],4)
            self.assertNotEqual(state['public']['checkpoint_roles']['best'],state['public']['checkpoint_roles']['last'])
            calls.clear()
            index_pin=hashlib.sha256((root/'model-artifacts.json').read_bytes()).hexdigest()
            receipt=remote.verify_model(root,model,batch_rows,{1:1,2:2,3:3,4:4},slot,evaluate=evaluate,binding={'input_sha256':'a'*64},gate=lambda:None,artifact_index_sha256=index_pin)
            self.assertEqual(calls,[1]);self.assertEqual(receipt['status'],'completed+verified')
            self.assertEqual(receipt['optimizer_steps_during_verify'],0)
            checkpoint=root/'state.pt';checkpoint.write_bytes(b'broken')
            with self.assertRaises(ValueError):remote.verify_model(root,model,batch_rows,{1:1,2:2,3:3,4:4},slot,evaluate=evaluate,binding={'input_sha256':'a'*64},gate=lambda:None,artifact_index_sha256=index_pin)

    def test_train_gate_failure_does_not_create_success_artifacts(self):
        import torch
        from vkgr.retrieval_sasrec import SASRecIDTwoTower
        from vkgr import retrieval_sasrec_plan as plan
        model=SASRecIDTwoTower(torch.zeros((5,64)),[1,2,3,4]);optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
        slot=next(s for s in plan.FORMAL_SLOTS if s['slot_id']=='P4H-S02')
        def fail():raise ValueError('stale G feed')
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve()
            with self.assertRaises(ValueError):remote.train_model(model,optimizer,[],{1:1},slot,evaluate=lambda *_:None,output=root,binding={'input_sha256':'a'*64},gate=fail)
            self.assertEqual(list(root.iterdir()),[])


def real_manifest_fixture(root):
    from vkgr import retrieval_sasrec_plan as plan
    roles=copy.deepcopy(plan.synthetic_pilot_manifest()[0]['roles'])
    for row in roles:row['path']=str(root/'inputs'/(row['role']+'.json'))
    value={'schema_id':'p4h-real-operation-v1','operation_id':'p4h-pilot-attempt01','stage':'pilot','roles':roles,'code_root':str(root/'code'),'python':str(root/'runtime'/'python'),'source_pins':{'src/vkgr/retrieval_sasrec.py':'301b7086af2a2ada58e28c4440d18c880bdd50ddf7ccafc19fd470dcfb80f00c'},'python_sha256':'a'*64,'output_root':str(root/'output'),'control_root':str(root/'control'),'authorization':{'path':str(root/'authorization.json'),'sha256':'b'*64,'issued_at':100,'expires_at':220,'stages':['pilot']},'queue':{'kind':'pilot','started_at':100,'deadline':14500,'disk_cap':20*1024**3,'attempts':1},'limits':{'sample_seconds':5,'seal_seconds':60,'signal_grace_seconds':10,'stop_reserve_bytes':1048576,'stage_seconds':60,'finish_seconds':240,'next_atomic_bytes':1048576},'g_feed':{'path':str(root/'g-feed-descriptor.json'),'sha256':'c'*64},'slot':None,'predecessor':None}

    value['queue'].update({'started_monotonic':1000.,'deadline_monotonic':15400.,'host_boot_id':'original-boot','receipt_path':str(root/'queue.json'),'receipt_sha256':'0'*64})
    return value

@unittest.skipIf(remote is None,'real adapter missing')
class RealManifestTests(unittest.TestCase):
    def test_valid_real_manifest_has_static_no_path_gate(self):
        value=real_manifest_fixture(Path('/private/tmp/real-operation'))
        self.assertEqual(remote.validate_remote_manifest(value)['stage'],'pilot')

    def test_bad_role_lineage_or_unregistered_slot_rejects_before_path(self):
        value=real_manifest_fixture(Path('/private/tmp/real-operation'))
        for change in ('validation','extra_role','synthetic','unknown_stage','root_overlap','nonfinite_deadline'):
            row=copy.deepcopy(value)
            if change=='validation':row['roles'][5]['lineage']['split']='validation'
            elif change=='extra_role':row['roles'].append(copy.deepcopy(row['roles'][0]))
            elif change=='synthetic':row['synthetic_local']=False
            elif change=='unknown_stage':row['stage']='test'
            elif change=='root_overlap':row['control_root']=row['output_root']+'/nested'
            else:row['queue']['deadline']=float('nan')
            with self.subTest(change=change),self.assertRaises(ValueError):remote.validate_remote_manifest(row)

    def test_live_approval_binds_all_limits_and_queue_identity(self):
        value=real_manifest_fixture(Path('/private/tmp/real-operation'))
        permit=remote.make_admission(value,manifest_sha256='d'*64,start_sha256='e'*64,g_receipt_sha256='f'*64,now=101)
        remote.validate_admission(value,permit,manifest_sha256='d'*64,start_sha256='e'*64,g_receipt_sha256='f'*64,now=102)
        for field,changed in [('manifest_sha256','0'*64),('start_sha256','0'*64),('g_receipt_sha256','0'*64),('deadline',14501),('disk_cap',20*1024**3+1)]:
            p=copy.deepcopy(permit);p[field]=changed
            with self.assertRaises(ValueError):remote.validate_admission(value,p,manifest_sha256='d'*64,start_sha256='e'*64,g_receipt_sha256='f'*64,now=102)
        with self.assertRaises(ValueError):remote.validate_admission(value,permit,manifest_sha256='d'*64,start_sha256='e'*64,g_receipt_sha256='f'*64,now=221)

@unittest.skipIf(remote is None,'real adapter missing')
class ProcessIdentityTests(unittest.TestCase):
    def test_only_registered_live_p_group_can_be_signalled(self):
        leader={'pid':40,'ppid':10,'pgid':40,'sid':40,'birth':100,'state':'S'}
        child={'pid':41,'ppid':40,'pgid':40,'sid':40,'birth':101,'state':'S'}
        self.assertEqual(remote.authorize_group_signal(leader,[leader,child],[leader,child],controller_pgid=10,protected_pgids={90}),40)
        for observed,registered,controller,protected in [
            ([{**leader,'birth':999},child],[leader,child],10,{90}),
            ([leader,child,{**child,'pid':42,'birth':102}],[leader,child],10,{90}),
            ([leader,child],[leader,child],40,{90}),
            ([leader,child],[leader,child],10,{40}),
            ([child],[leader,child],10,{90}),
        ]:
            with self.assertRaises(ValueError):remote.authorize_group_signal(leader,observed,registered,controller_pgid=controller,protected_pgids=protected)

    def test_proc_parser_preserves_birth_with_parentheses_in_comm(self):
        fields=['S','10','40','40']+['0']*15+['12345']+['0']*8
        value=remote.parse_proc_stat('40 (python (worker)) '+' '.join(fields))
        self.assertEqual(value,{'pid':40,'ppid':10,'pgid':40,'sid':40,'birth':12345,'state':'S'})

    def test_descendant_inventory_does_not_assume_same_group_means_owned(self):
        leader={'pid':40,'ppid':10,'pgid':40,'sid':40,'birth':100,'state':'S'}
        child={'pid':41,'ppid':40,'pgid':41,'sid':41,'birth':101,'state':'S'}
        grandchild={'pid':42,'ppid':41,'pgid':42,'sid':42,'birth':102,'state':'S'}
        stranger={'pid':90,'ppid':1,'pgid':40,'sid':40,'birth':999,'state':'S'}
        self.assertEqual({r['pid'] for r in remote.descendants(leader,[leader,child,grandchild,stranger])},{40,41,42})

@unittest.skipIf(remote is None,'real adapter missing')
class SafetyFeedTests(unittest.TestCase):
    def fixture(self):
        runs=[{'run_id':'core-02','start_sha256':'a'*64}]
        payload={'observed_at':100.,'runs':[{'run_id':'core-02','start_sha256':'a'*64,'status':'RUNNING','owner_live':True}],'free_fractions':{'vram':.5,'ram':.5,'disk':.5},'gpu_pids':[40],'allowed_gpu_pids':[40],'protected_pgids':[40],'p_writer_count':0}
        return runs,payload
    def test_unknown_stale_wrong_g_identity_and_unknown_gpu_writer_stop(self):
        runs,payload=self.fixture();self.assertTrue(remote.validate_safety_payload(payload,runs,now=101,max_age=30))
        for change in ('stale','identity','owner','unknown','gpu','resource','duplicate_p'):
            p=copy.deepcopy(payload)
            if change=='stale':p['observed_at']=60
            elif change=='identity':p['runs'][0]['start_sha256']='b'*64
            elif change=='owner':p['runs'][0]['owner_live']=False
            elif change=='unknown':p['runs'][0]['status']='UNKNOWN'
            elif change=='gpu':p['gpu_pids'].append(999)
            elif change=='resource':p['free_fractions']['ram']=.19
            else:p['p_writer_count']=1
            with self.subTest(change=change),self.assertRaises(ValueError):remote.validate_safety_payload(p,runs,now=101,max_age=30)

    def test_consumer_verifies_payload_digest_not_only_receipt_hash(self):
        runs,payload=self.fixture()
        receipt={'payload_sha256':hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()}
        self.assertTrue(remote.check_payload_binding(receipt,payload))
        payload['gpu_pids'].append(999)
        with self.assertRaises(ValueError):remote.check_payload_binding(receipt,payload)

@unittest.skipIf(remote is None,'real adapter missing')
class ReadonlyObserverTests(unittest.TestCase):
    def test_observation_derives_g_identity_from_bound_disk_and_proc(self):
        from unittest.mock import patch
        leader={'pid':40,'ppid':1,'pgid':40,'sid':40,'birth':100,'state':'S'}
        gpu={'pid':41,'ppid':40,'pgid':41,'sid':41,'birth':101,'state':'R'}
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();pins={'root':str(root),'run_id':'core-02'}
            records={'controller.py':b'controller-source','approval.json':b'{}','input.json':b'{"registration":{"run_id":"core-02"}}','start.json':None}
            for name,data in records.items():
                if data is None:continue
                (root/name).write_bytes(data);pins[name.split('.')[0]+'_sha256']=hashlib.sha256(data).hexdigest()
            start={k:pins[k] for k in ('controller_sha256','approval_sha256','input_sha256')}
            blob=json.dumps(start).encode();(root/'start.json').write_bytes(blob);pins['start_sha256']=hashlib.sha256(blob).hexdigest()
            (root/'owner.json').write_text(json.dumps({'identity':{k:leader[k] for k in ('pid','pgid','sid','birth')},'start_sha256':pins['start_sha256']}));(root/'terminal.json').write_bytes(b'')
            with patch.object(remote,'linux_processes',return_value=[leader,gpu]),patch.object(remote,'linux_resources',return_value=({'vram':.5,'ram':.5,'disk':.5},[41])),patch.object(remote.time,'time',return_value=100.):
                result=remote.observe_g([pins],allowed_p=[])
                self.assertEqual(result['runs'][0]['status'],'RUNNING');self.assertEqual(result['allowed_gpu_pids'],[40,41]);self.assertEqual(result['protected_pgids'],[40,41])
                remote.validate_safety_payload(result,[pins],now=100,max_age=30)
                (root/'controller.py').write_bytes(b'changed')
                with self.assertRaises(ValueError):remote.observe_g([pins],allowed_p=[])

@unittest.skipIf(remote is None,'real adapter missing')
class FeedJournalTests(unittest.TestCase):
    def test_feed_uses_immutable_receipts_and_atomic_head_with_real_payload_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve()/'feed';identity={'pid':40,'start_ticks':100,'executable_sha256':'a'*64,'start_receipt_sha256':'b'*64}
            jobs={'job_id_hash':'c'*64,'start_identity_hash':'d'*64}
            journal=remote.FeedJournal(root,identity,jobs)
            try:
                payload={'observed_at':100,'value':'actual-observation'}
                first=journal.publish(payload,now=100,status='OPEN')
                second=journal.publish(payload,now=105,status='OPEN')
                self.assertEqual(second['value']['previous_receipt_sha256'],first['sha256'])
                self.assertEqual(second['value']['seq'],1)
                self.assertEqual(json.loads((root/'HEAD.json').read_bytes())['receipt_sha256'],second['sha256'])
                self.assertEqual(json.loads((root/('payload-'+first['value']['payload_sha256']+'.json')).read_bytes()),payload)
                saved=(root/first['name']).read_bytes()
                journal.publish(payload,now=110,status='OWNER_LOST')
                self.assertEqual((root/first['name']).read_bytes(),saved)
                with self.assertRaises(ValueError):journal.publish(payload,now=115,status='OPEN')
                with self.assertRaises(FileExistsError):remote.FeedJournal(root,identity,jobs)
            finally:journal.close()

@unittest.skipIf(remote is None,'real adapter missing')
class AdoptionIdentityTests(unittest.TestCase):
    def test_known_descendant_reparent_does_not_change_birth_identity(self):
        leader={'pid':40,'ppid':10,'pgid':40,'sid':40,'birth':100,'state':'S'}
        child={'pid':41,'ppid':40,'pgid':40,'sid':40,'birth':101,'state':'S'}
        self.assertEqual(remote.authorize_group_signal(leader,[leader,{**child,'ppid':10}],[leader,child],controller_pgid=10,protected_pgids={90}),40)

@unittest.skipIf(remote is None,'real adapter missing')
class FeedConsumerTests(unittest.TestCase):
    def test_disk_feed_payload_chain_freshness_and_rollback_are_enforced(self):
        from unittest.mock import patch
        runs,payload=SafetyFeedTests().fixture()
        identity={'pid':40,'start_ticks':100,'executable_sha256':'a'*64,'start_receipt_sha256':'b'*64};jobs={'job_id_hash':'c'*64,'start_identity_hash':'d'*64}
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();journal=remote.FeedJournal(root/'feed',identity,jobs)
            try:
                first=journal.publish(payload,now=100,status='OPEN');journal.descriptor['initial_no_other_writer_inventory_sha256']='e'*64
                desc=root/'descriptor.json';pin=remote.write_once(desc,journal.descriptor)
                reader=remote.SafetyFeedReader(desc,pin,runs)
                with patch.object(remote,'prove_publisher_runtime',return_value=True):
                    self.assertEqual(reader.observe(now=101)['receipt_sha256'],first['sha256'])
                    payload={**payload,'observed_at':105.};second=journal.publish(payload,now=105,status='OPEN')
                    self.assertEqual(reader.observe(now=106)['receipt_sha256'],second['sha256'])
                    (journal.root/'HEAD.json').write_text(json.dumps({'seq':0,'receipt_name':first['name'],'receipt_sha256':first['sha256']}))
                    with self.assertRaises(ValueError):reader.observe(now=107)
                    (journal.root/'HEAD.json').write_text(json.dumps({'seq':1,'receipt_name':second['name'],'receipt_sha256':second['sha256']}))
                    with self.assertRaises(ValueError):reader.observe(now=140)
                    path=journal.root/('payload-'+second['value']['payload_sha256']+'.json');path.chmod(0o600);path.write_text('{}')
                    with self.assertRaises(ValueError):reader.observe(now=107)
            finally:journal.close()

@unittest.skipIf(remote is None,'real adapter missing')
class TrainingRoleTests(unittest.TestCase):
    def test_stripped_roles_reconstruct_full_past_likes_and_preserve_support(self):
        prepared=[];watch=[]
        for step in range(1,62):
            history=list(range(max(1,step-49),step+1));positions=[i-1 for i in history]
            prepared.append({'user_id':7,'history':history,'history_positions':positions,'target':step+1,'target_position':step})
            watch.append({'example_id':step-1,'history':history,'history_positions':positions})
        catalog=list(range(1,65));identity={'items':[{'item_id':item,'index':item,'supported':item!=64} for item in catalog],'pad':0,'start':1}
        value=remote.compile_training_roles(prepared,watch,{'item_ids':catalog},identity)
        self.assertEqual(value['rows_by_arm']['like'][-1]['past_likes'],list(range(1,62)))
        self.assertIn(63,value['supported_indices']);self.assertNotIn(64,value['supported_indices'])
        self.assertEqual(value['rows_by_arm']['observed'][-1]['target'],62)
        changed=copy.deepcopy(watch);changed[0]['history_positions']=[1]
        with self.assertRaises(ValueError):remote.compile_training_roles(prepared,changed,{'item_ids':catalog},identity)

    def test_runtime_batches_are_finite_identical_draws_for_two_arms(self):
        like=[{'history':[1],'target':2,'past_likes':[1]},{'history':[2],'target':3,'past_likes':[1,2]}]
        observed=[{**r,'history':[9]+r['history']} for r in like]
        a=remote.finite_batches(like,batch_size=2,seed=42);b=remote.finite_batches(observed,batch_size=2,seed=42)
        self.assertEqual(len(a),30);self.assertEqual([x['rows'][0]['target'] for x in a],[x['rows'][0]['target'] for x in b])
        self.assertTrue(all(x['epoch_end'] for x in a))
        with self.assertRaises(ValueError):remote.finite_batches(like,batch_size=1,seed=999)

@unittest.skipIf(remote is None,'real adapter missing')
class RoleReaderTests(unittest.TestCase):
    def test_roles_read_same_sha_bound_bytes_and_reject_extra_supervision(self):
        from vkgr import retrieval_sasrec_plan as plan
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();(root/'inputs').mkdir();value=real_manifest_fixture(root)
            for role in value['roles']:
                if role['role']=='prepared-train-rows':payload=[{'user_id':7,'history':[10],'history_positions':[0],'target':11,'target_position':1}]
                elif role['role']=='watch-train':payload=[{'example_id':0,'history':[10],'history_positions':[0]}]
                elif role['role']=='train-prefix-target-free-sidecar':payload=[{'query_id':0,'history':[10]}]
                else:payload={field:None for field in role['schema_fields']}
                p=Path(role['path']);blob=json.dumps(payload).encode();p.write_bytes(blob);role['sha256']=hashlib.sha256(blob).hexdigest();role['size']=len(blob)
            payloads,opened=remote.read_roles(value)
            self.assertEqual(len(opened),15);self.assertEqual(payloads['prepared-train-rows'][0]['target'],11)
            side=next(r for r in value['roles'] if r['role']=='train-prefix-target-free-sidecar');p=Path(side['path']);blob=b'[{"query_id":0,"history":[10],"target":11}]';p.write_bytes(blob);side['sha256']=hashlib.sha256(blob).hexdigest();side['size']=len(blob)
            with self.assertRaises(ValueError):remote.read_roles(value)

@unittest.skipIf(remote is None,'real adapter missing')
class LatchCleanupTests(unittest.TestCase):
    def test_identity_probe_failure_closes_unreleased_child_without_science(self):
        import os
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();marker=root/'science-ran';seen=[]
            def failed_probe(pid):seen.append(pid);raise ValueError('birth identity unavailable')
            with self.assertRaises(ValueError):remote.start_latched_child('from pathlib import Path;Path('+repr(str(marker))+').write_text("bad")',root/'stdout',root/'stderr',identity_probe=failed_probe)
            self.assertFalse(marker.exists());self.assertEqual(len(seen),1)
            with self.assertRaises(ProcessLookupError):os.kill(seen[0],0)

    def test_science_executes_only_after_explicit_latch_release(self):
        import os
        from vkgr.retrieval_sasrec_operation import process_identity
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();marker=root/'science-ran'
            handle=remote.start_latched_child('from pathlib import Path;Path('+repr(str(marker))+').write_text("done")',root/'stdout',root/'stderr',identity_probe=process_identity)
            try:
                self.assertFalse(marker.exists());handle['process'].stdin.write(b'1');handle['process'].stdin.flush();handle['process'].stdin.close()
                self.assertEqual(handle['process'].wait(timeout=5),0);self.assertEqual(marker.read_text(),'done')
            finally:
                handle['stdout'].close();handle['stderr'].close()
                if handle['pidfd'] is not None:os.close(handle['pidfd'])

@unittest.skipIf(remote is None,'real adapter missing')
class MonotonicBudgetTests(unittest.TestCase):
    def test_new_controller_uses_original_host_monotonic_deadline(self):
        queue={'kind':'pilot','started_at':100.,'deadline':14500.,'disk_cap':20*1024**3,'attempts':1,'started_monotonic':1000.,'deadline_monotonic':15400.,'host_boot_id':'original-boot'}
        budget=remote.MonotonicBudget(queue,monotonic=lambda:1300.,boot_id='original-boot')
        self.assertEqual(budget.remaining_seconds(400.),14100.)
        with self.assertRaises(ValueError):budget.remaining_seconds(100.)
        with self.assertRaises(ValueError):remote.MonotonicBudget(queue,monotonic=lambda:1300.,boot_id='changed-boot')
        changed={**queue,'deadline_monotonic':16000.}
        with self.assertRaises(ValueError):remote.MonotonicBudget(changed,monotonic=lambda:1300.,boot_id='original-boot')

@unittest.skipIf(remote is None,'real adapter missing')
class RealQueueBindingTests(unittest.TestCase):
    def test_static_gate_requires_frozen_monotonic_queue_receipt(self):
        value=real_manifest_fixture(Path('/private/tmp/real-operation'))
        remote.validate_remote_manifest(value)
        for key in ('started_monotonic','deadline_monotonic','host_boot_id','receipt_path','receipt_sha256'):
            row=copy.deepcopy(value);row['queue'].pop(key)
            with self.assertRaises(ValueError):remote.validate_remote_manifest(row)
        changed=copy.deepcopy(value);changed['queue']['deadline_monotonic']+=1
        with self.assertRaises(ValueError):remote.validate_remote_manifest(changed)

@unittest.skipIf(remote is None,'real adapter missing')
class VerifierAuthorityTests(unittest.TestCase):
    def test_verifier_rejects_index_without_closed_predecessor_external_pin(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();p=root/'model-artifacts.json';p.write_text('{}')
            with self.assertRaises(ValueError):remote.verify_model(root,None,[],{},None,evaluate=lambda *_:None,binding={},gate=lambda:None,artifact_index_sha256='a'*64)
            with self.assertRaises(ValueError):remote.verify_model(root,None,[],{},None,evaluate=lambda *_:None,binding={},gate=lambda:None)

@unittest.skipIf(remote is None,'real adapter missing')
class OwnedTreeTests(unittest.TestCase):
    def test_adopted_detached_descendants_are_owned_but_neighbor_is_not(self):
        leader={'pid':40,'ppid':10,'pgid':40,'sid':40,'birth':100,'state':'S'}
        child={'pid':41,'ppid':40,'pgid':41,'sid':41,'birth':101,'state':'S'}
        neighbor={'pid':90,'ppid':10,'pgid':90,'sid':90,'birth':900,'state':'S'}
        table=[leader,child,neighbor]
        tracker=remote.OwnedTree(10,10,baseline_children={90},protected_pgids={90},table=lambda:table)
        tracker.remember(leader);self.assertEqual({r['pid'] for r in tracker.collect()},{40,41})
        table[:]=[{**child,'ppid':10},neighbor]
        self.assertEqual([r['pid'] for r in tracker.signal_plan()],[41])
        self.assertFalse(tracker.closed())
        table[:]=[neighbor];self.assertTrue(tracker.closed())
        table[:]=[{**leader,'birth':999},neighbor]
        with self.assertRaises(ValueError):tracker.signal_plan()

    def test_unknown_member_or_protected_group_prevents_all_signals(self):
        leader={'pid':40,'ppid':10,'pgid':40,'sid':40,'birth':100,'state':'S'}
        stranger={'pid':91,'ppid':1,'pgid':40,'sid':40,'birth':999,'state':'S'}
        table=[leader,stranger]
        tracker=remote.OwnedTree(10,10,baseline_children=set(),protected_pgids={90},table=lambda:table);tracker.remember(leader)
        with self.assertRaises(ValueError):tracker.signal_plan()
        table[:]=[leader]
        tracker.protected_pgids.add(40)
        with self.assertRaises(ValueError):tracker.signal_plan()

class FakeKernel:
    def __init__(self,rows):self.rows={r['pid']:dict(r) for r in rows};self.fds={1:40};self.sent=[];self.next_fd=2
    def processes(self):return list(self.rows.values())
    def identity(self,pid):return self.rows.get(pid)
    def open_pidfd(self,row):
        fd=self.next_fd;self.next_fd+=1;self.fds[fd]=row['pid'];return fd
    def signal(self,fd,sig):
        pid=self.fds[fd];self.sent.append(pid)
        if pid in self.rows:self.rows[pid]['state']='Z'
    def reap(self,row):
        if self.rows.get(row['pid'],{}).get('state')=='Z':self.rows.pop(row['pid']);return {'pid':row['pid'],'birth':row['birth'],'wait_status':0}
    def close_fd(self,fd):self.fds.pop(fd,None)

class FakeWaitingProcess:
    def __init__(self,kernel):self.pid=40;self.kernel=kernel;self.returncode=None
    def poll(self):
        if self.pid not in self.kernel.rows or self.kernel.rows[self.pid]['state']=='Z':self.returncode=-15;self.kernel.rows.pop(self.pid,None)
        return self.returncode
    def wait(self,timeout):
        import subprocess
        if self.poll() is None:raise subprocess.TimeoutExpired('owned leaf',timeout)
        return self.returncode

@unittest.skipIf(remote is None,'real adapter missing')
class KernelLeaseTests(unittest.TestCase):
    def test_escalation_closes_only_owned_tree_and_actual_waits(self):
        import io
        leader={'pid':40,'ppid':10,'pgid':40,'sid':40,'birth':100,'state':'S'}
        child={'pid':41,'ppid':40,'pgid':41,'sid':41,'birth':101,'state':'S'}
        neighbor={'pid':90,'ppid':10,'pgid':90,'sid':90,'birth':900,'state':'S'}
        kernel=FakeKernel([leader,child,neighbor]);process=FakeWaitingProcess(kernel)
        handle={'process':process,'identity':leader,'pidfd':1,'stdout':io.BytesIO(),'stderr':io.BytesIO(),'released':True}
        lease=remote.KernelLease(handle,kernel=kernel,owner_pid=10,controller_pgid=10,baseline_children={90},protected_pgids={90})
        with tempfile.TemporaryDirectory() as folder:
            from unittest.mock import patch
            clock=[0.]
            def advance(seconds):clock[0]+=seconds
            # This fake-kernel test verifies escalation, not filesystem latency.
            # A real fsync can exceed its 40ms synthetic cleanup envelope.
            with patch.object(remote.time,'monotonic',side_effect=lambda:clock[0]),patch.object(remote.time,'sleep',side_effect=advance):
                result=lease.close(Path(folder).resolve()/'stop.json',manifest_sha256='a'*64,seal_seconds=.01,signal_grace_seconds=.01)
            self.assertEqual(result['state'],'CLOSED');self.assertEqual(result['actual_worker_exit'],-15)
            self.assertEqual(set(kernel.sent),{40,41});self.assertIn(90,kernel.rows);self.assertNotEqual(kernel.rows[90]['state'],'Z')
            self.assertEqual(kernel.fds,{})

    def test_birth_drift_never_signals_reused_pid(self):
        import io
        leader={'pid':40,'ppid':10,'pgid':40,'sid':40,'birth':100,'state':'S'}
        kernel=FakeKernel([leader]);handle={'process':FakeWaitingProcess(kernel),'identity':leader,'pidfd':1,'stdout':io.BytesIO(),'stderr':io.BytesIO(),'released':True}
        lease=remote.KernelLease(handle,kernel=kernel,owner_pid=10,controller_pgid=10,baseline_children=set(),protected_pgids={90})
        kernel.rows[40]['birth']=999
        with tempfile.TemporaryDirectory() as folder:
            result=lease.close(Path(folder).resolve()/'stop.json',manifest_sha256='a'*64,seal_seconds=.01,signal_grace_seconds=.01)
            self.assertEqual(result['state'],'FAILED_UNSAFE');self.assertEqual(kernel.sent,[])

@unittest.skipIf(remote is None,'real adapter missing')
class RuntimeGateTests(unittest.TestCase):
    def test_gate_reserves_atomic_peak_and_stops_on_stop_file_or_stale_feed(self):
        from unittest.mock import patch
        runs,payload=SafetyFeedTests().fixture();identity={'pid':40,'start_ticks':100,'executable_sha256':'a'*64,'start_receipt_sha256':'b'*64};jobs={'job_id_hash':'c'*64,'start_identity_hash':'d'*64}
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();(root/'control').mkdir();journal=remote.FeedJournal(root/'feed',identity,jobs)
            try:
                journal.publish(payload,now=100,status='OPEN');journal.descriptor['initial_no_other_writer_inventory_sha256']='e'*64
                desc=root/'descriptor.json';pin=remote.write_once(desc,journal.descriptor);reader=remote.SafetyFeedReader(desc,pin,runs)
                budget=remote.MonotonicBudget({'kind':'pilot','started_at':100.,'deadline':14500.,'disk_cap':20*1024**3,'attempts':1,'started_monotonic':1000.,'deadline_monotonic':15400.,'host_boot_id':'boot'},monotonic=lambda:1001.,boot_id='boot')
                limits={'stage_seconds':60,'finish_seconds':120,'next_atomic_bytes':1048576,'stop_reserve_bytes':1048576}
                gate=remote.RuntimeGate(root,root/'control'/'stop.json',budget,reader,limits,clock=lambda:101.)
                with patch.object(remote,'prove_publisher_runtime',return_value=True):
                    self.assertEqual(gate.check()['receipt_sha256'],journal.previous)
                    (root/'control'/'stop.json').write_text('{}')
                    with self.assertRaises(ValueError):gate.check()
                with self.assertRaises(ValueError):remote.queue_tree_usage(root,max_nodes=1)
            finally:journal.close()

@unittest.skipIf(remote is None,'real adapter missing')
class TrainPrefixCompilerTests(unittest.TestCase):
    def test_capacity_queries_replay_only_train_prefixes_including_single_like_users(self):
        from vkgr.retrieval_data import prefix_examples
        events=[{'user_id':7,'item_id':10,'like':True},{'user_id':8,'item_id':11,'like':True},{'user_id':7,'item_id':12,'like':False},{'user_id':7,'item_id':11,'like':True},{'user_id':8,'item_id':12,'like':False}]
        examples=list(prefix_examples([(0,events)]))
        result=remote.compile_train_prefixes([(0,events)],examples,[10,11,12])
        self.assertEqual(result['watch_train'],[{'example_id':0,'history':[10,12],'history_positions':[[0,0],[0,2]]}])
        self.assertEqual(result['sidecar'],[{'query_id':14,'history':[10,11]},{'query_id':15,'history':[10,12,11]},{'query_id':16,'history':[11]},{'query_id':17,'history':[11,12]}])
        self.assertEqual(result['support'],[10,11])
        with self.assertRaises(ValueError):remote.compile_train_prefixes([(25,events)],examples,[10,11,12])
        self.assertEqual(remote.encode_position([24,123]),24*2**32+123)
        with self.assertRaises(ValueError):remote.encode_position([25,0])
        with self.assertRaises(ValueError):remote.encode_position([0,2**32])


class RealIOIntegrationTests(unittest.TestCase):
    def test_materialized_roles_bind_original_mapping_support_and_raw_replay(self):
        from vkgr.retrieval_data import prefix_examples
        events=[{'user_id':1,'item_id':10,'like':True},{'user_id':1,'item_id':20,'like':False},{'user_id':1,'item_id':30,'like':True}]
        examples=list(prefix_examples([(0,events)]))
        original={'items':[10,20,30],'pad':0,'start':1,'supported_indices':[1,3]}
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();destination=root/'roles'
            result=remote.materialize_training_inputs(examples,[10,20,30],original,[10,30],[(0,events)],destination)
            self.assertEqual(result['catalog'],[10,20,30]);self.assertEqual(result['supported_indices'],[1,3])
            self.assertEqual(result['rows_by_arm']['observed'][0]['history'],[10,20])
            self.assertEqual(result['rows_by_arm']['like'][0]['past_likes'],[10])
            self.assertEqual(json.loads((destination/'train-prefix-target-free-sidecar.json').read_bytes()),[{'query_id':2,'history':[10,30]},{'query_id':3,'history':[10,20,30]}])
            with self.assertRaises(ValueError):remote.materialize_training_inputs(examples,[10,20,30],{**original,'supported_indices':[1,2,3]},[10,30],[(0,events)],root/'bad')
            self.assertFalse((root/'bad').exists())
            changed=copy.deepcopy(events);changed[2]['item_id']=20
            with self.assertRaises(ValueError):remote.materialize_training_inputs(examples,[10,20,30],original,[10,30],[(0,changed)],root/'badraw')
            self.assertFalse((root/'badraw').exists())

    def test_source_binding_rejects_missing_runtime_member_before_import(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();pins={}
            for name in remote.RUNTIME_SOURCE_MEMBERS:
                p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'# controlled source\n');pins[name]=hashlib.sha256(p.read_bytes()).hexdigest()
            manifest={'code_root':str(root),'source_pins':pins}
            remote.verify_source_tree(manifest)
            broken=copy.deepcopy(manifest);broken['source_pins'].pop('src/vkgr/retrieval_training.py')
            with self.assertRaises(ValueError):remote.verify_source_tree(broken)
            (root/'src/vkgr/retrieval_training.py').write_bytes(b'drift')
            with self.assertRaises(ValueError):remote.verify_source_tree(manifest)

    def test_cgroup_visible_memory_uses_tightest_ancestor(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();child=root/'child';child.mkdir()
            (root/'memory.max').write_text('1000');(root/'memory.current').write_text('850')
            (child/'memory.max').write_text('500');(child/'memory.current').write_text('100')
            self.assertAlmostEqual(remote.visible_ram_fraction(child,root,.7),.15)
            (root/'memory.max').write_text('max');(child/'memory.max').write_text('max')
            with self.assertRaises(ValueError):remote.visible_ram_fraction(child,root,.7)

    def test_publisher_serves_real_journal_and_expires_without_g_mutation(self):
        from unittest.mock import patch
        runs,payload=SafetyFeedTests().fixture()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve()/'feed'
            def identity(pid):return {'pid':pid,'start_ticks':100,'executable_sha256':'a'*64,'start_receipt_sha256':'b'*64}
            def inventory(descriptor):return {'inventory_sha256':'c'*64}
            with patch.object(remote,'linux_publisher_identity',side_effect=identity),patch.object(remote,'linux_writer_inventory',side_effect=inventory),patch.object(remote,'observe_g',return_value=payload):
                result=remote.serve_publisher(root,runs,deadline=time.time()+.08,interval=.02,allowed_p=[])
            self.assertEqual(result['status'],'CLOSED')
            descriptor=json.loads((root/'descriptor.json').read_bytes());head=json.loads((root/'HEAD.json').read_bytes())
            last=json.loads((root/head['receipt_name']).read_bytes())
            self.assertEqual(last['status'],'CLOSED');self.assertGreater(head['seq'],0)
            self.assertEqual(descriptor['initial_no_other_writer_inventory_sha256'],hashlib.sha256((root/'initial-writer-inventory.json').read_bytes()).hexdigest())


class ActualWorkerIntegrationTests(unittest.TestCase):
    def test_actual_pilot_reloads_tensor_and_scores_complete_catalog_without_targets(self):
        catalog=list(range(1,85));mapping={i:i for i in catalog}
        rows=[{'example_id':j,'user_id':j,'history':[1,2],'history_positions':[0,1],'target':3+j%80,'target_position':2+j,'past_likes':[1,2]} for j in range(512)]
        data={'catalog':catalog,'idmap':mapping,'supported_indices':list(range(1,85)),'rows_by_arm':{'like':rows,'observed':rows},'sidecar':[{'query_id':0,'history':[1,2]},{'query_id':1,'history':[2,1]}]}
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder).resolve();calls=[]
            result=remote.run_pilot_worker(data,output,binding={'source':'a'*64},gate=lambda:calls.append(True),device='cpu',steps=1)
            self.assertEqual(result['status'],'pilot_worker_completed+unverified')
            self.assertEqual(set(result['arms']),{'like','observed'})
            for arm,row in result['arms'].items():
                self.assertEqual(row['optimizer_steps'],1)
                self.assertEqual(set(row['measurements']),{'train_step','draw_replay','epoch_export_exact','target_free_top80','final_verify_export','final_verify_exact','checkpoint_write'})
                self.assertEqual(row['top80_count'],[80,80])
                self.assertTrue((output/(arm+'-state.pt')).is_file())
                self.assertEqual(row['fresh_export_matches'],True)
                import torch
                checkpoint=torch.load(output/(arm+'-state.pt'),weights_only=True,map_location='cpu')
                self.assertTrue(checkpoint['optimizer']['state'])
                self.assertGreater(row['measurements']['epoch_export_exact']['bytes'],0)
                self.assertEqual(json.loads((output/(arm+'-capacity-predictions.json')).read_bytes()),json.loads((output/(arm+'-capacity-fresh-predictions.json')).read_bytes()))
                self.assertEqual(set(checkpoint['best_model']),set(checkpoint['model']))
            self.assertGreater(len(calls),7)
            self.assertNotIn('quality_metrics',result)

    def test_formal_evaluation_includes_oov_denominator_and_persists_predictions(self):
        import torch
        from vkgr.retrieval_sasrec import SASRecIDTwoTower
        torch.manual_seed(42);model=SASRecIDTwoTower(torch.zeros((85,64)),list(range(1,85)))
        queries=[{'query_id':0,'user_id':0,'history':[1,2],'history_positions':[0,1],'target':3,'oov':False},{'query_id':1,'user_id':1,'history':[2,1],'history_positions':[0,1],'target':999,'oov':True}]
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve()
            result=remote.evaluate_formal(model,1,list(range(1,85)),queries,{i:i for i in range(1,85)},list(range(1,85)),root,gate=lambda:None)
            self.assertEqual(set(result),{'epoch','metrics','prediction_sha256'})
            audit=json.loads((root/'epoch-1-metrics.json').read_bytes())
            self.assertEqual(audit['metrics']['all']['count'],2)
            self.assertEqual(audit['metrics']['oov']['count'],1)
            self.assertEqual(audit['metrics']['oov']['recall@50'],0.)
            self.assertEqual(hashlib.sha256((root/'epoch-1-predictions.json').read_bytes()).hexdigest(),result['prediction_sha256'])


class ControllerIntegrationTests(unittest.TestCase):
    def test_bootstrap_rejects_unbound_source_before_creating_roots(self):
        fixture=real_manifest_fixture(Path('/private/tmp/real-operation'))
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();fixture['code_root']=str(root/'code');fixture['output_root']=str(root/'out');fixture['control_root']=str(root/'ctl')
            with self.assertRaises(ValueError):remote.prepare_controller(fixture,'a'*64,[])
            self.assertFalse((root/'out').exists());self.assertFalse((root/'ctl').exists())

    def test_common_roles_cannot_claim_a_different_code_or_runtime(self):
        fixture=real_manifest_fixture(Path('/private/tmp/real-operation'))
        contents={'code-archive':{'members':{},'source_sha256':'a'*64},'source-fingerprint':{'members':{},'source_sha256':'a'*64},'python-executable-receipt':{'python_version':'3.11.16','executable_sha256':'b'*64}}
        with self.assertRaises(ValueError):remote.verify_common_bindings(fixture,contents)

    def test_missing_pidfd_closes_unreleased_leaf_and_no_science_runs(self):
        import io
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();marker=root/'science'
            handle=remote.start_latched_child('open('+repr(str(marker))+',"w").write("unsafe")',root/'stdout',root/'stderr',identity_probe=lambda pid:{'pid':pid,'pgid':pid,'sid':pid,'birth':1,'ppid':10})
            if handle['pidfd'] is not None:
                __import__('os').close(handle['pidfd']);handle['pidfd']=None
            with self.assertRaises(ValueError):remote.KernelLease(handle,kernel=None,owner_pid=10,controller_pgid=10,baseline_children=set(),protected_pgids=set())
            self.assertIsNotNone(handle['process'].poll());self.assertFalse(marker.exists())
            self.assertTrue(handle['stdout'].closed);self.assertTrue(handle['stderr'].closed)

    def test_cli_exposes_real_entry_and_rejects_synthetic_before_output(self):
        import subprocess,sys
        result=subprocess.run([sys.executable,'-B','-m','vkgr.retrieval_sasrec_remote','--help'],capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,0)
        self.assertIn('prepare-controller',result.stdout);self.assertIn('serve-publisher',result.stdout);self.assertIn('worker',result.stdout)


class ClosedPilotIntegrationTests(unittest.TestCase):
    def test_actual_pilot_closed_seal_and_independent_tensor_verify_reject_tamper(self):
        from unittest.mock import patch
        catalog=list(range(1,85));mapping={i:i for i in catalog}
        rows=[{'example_id':j,'user_id':j,'history':[1,2],'history_positions':[0,1],'target':3+j%80,'target_position':2+j,'past_likes':[1,2]} for j in range(512)]
        data={'catalog':catalog,'idmap':mapping,'supported_indices':list(range(1,85)),'rows_by_arm':{'like':rows,'observed':rows},'sidecar':[{'query_id':0,'history':[1,2]},{'query_id':1,'history':[2,1]}]}
        binding={'source':'a'*64}
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();out=root/'out';out.mkdir();ctl=root/'ctl';ctl.mkdir()
            remote.run_pilot_worker(data,out,binding=binding,gate=lambda:None,device='cpu',steps=1,capacity_per_arm=1)
            value={'operation_id':'p4h-pilot-test','stage':'pilot','output_root':str(out),'control_root':str(ctl),'roles':[],'limits':{'seal_seconds':1},'queue':{'disk_cap':20*1024**3,'deadline':time.time()+100,'attempts':1}}
            start={'output_identity':{'device':out.stat().st_dev,'inode':out.stat().st_ino},'control_identity':{'device':ctl.stat().st_dev,'inode':ctl.stat().st_ino},'opened_roles':[]}
            closure={'state':'CLOSED','actual_worker_exit':0,'owned_sessions':[40],'owned_identities':[{'pid':40,'pgid':40,'sid':40,'birth':100}],'elapsed_seconds':.01,'signals_sent':0}
            with patch.object(remote,'linux_processes',return_value=[]):
                terminal,sealed=remote.seal_completed_operation(value,'a'*64,start,'b'*64,{'permit':True},closure,[{'queue_bytes':100}],1.,'c'*64)
            self.assertEqual(sealed['state'],'CLOSED');self.assertEqual(terminal['status'],'completed+unverified')
            self.assertEqual({r['role'] for r in sealed['artifact_hashes']},{'pilot-capacity-measurement-index','pilot-resource-ledger','pilot-stage-event-ledger','pilot-opened-artifact-ledger','pilot-artifact-index'})
            verified=remote.verify_actual_pilot(out,terminal,data,data['sidecar'],gate=lambda:None,device='cpu',binding=binding)
            self.assertEqual(verified['status'],'pilot_completed+verified');self.assertEqual(verified['optimizer_steps_during_verify'],0)
            (out/'like-state.pt').write_bytes(b'tampered')
            with self.assertRaises(ValueError):remote.verify_actual_pilot(out,terminal,data,data['sidecar'],gate=lambda:None,device='cpu',binding=binding)

    def test_no_seal_when_descendant_remains_or_actual_wait_is_missing(self):
        from unittest.mock import patch
        closure={'state':'CLOSED','actual_worker_exit':None,'owned_sessions':[40],'owned_identities':[],'elapsed_seconds':1.}
        with self.assertRaises(ValueError):remote.seal_completed_operation({},'a'*64,{},'b'*64,{},closure,[],1.,'c'*64)


class RealPreparationTests(unittest.TestCase):
    def test_prepare_input_refuses_unknown_producer_or_holdout_before_any_path(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve()
            for stage,pin in [('pilot','a'*64),('test','7f3ccd70c02157542e94f41f39e62988fbd650b4be8976455ff20cfe1e4394d2')]:
                spec={'stage':stage,'prepared':str(root/'missing'),'prepared_sha256':pin,'raw_root':str(root/'raw'),'output':str(root/'out'),'validation':None}
                with self.assertRaises(ValueError):remote.prepare_actual_inputs(spec)
                self.assertFalse((root/'out').exists())

    def test_positive_common_binding_uses_actual_uv_lock_and_package_origin(self):
        import platform,sys
        root=Path(remote.__file__).resolve().parents[2];pins={}
        for name in remote.RUNTIME_SOURCE_MEMBERS+('uv.lock','configs/p4h-sasrec-v2.json','docs/p4h_sasrec_protocol_v2.md'):
            pins[name]=hashlib.sha256((root/name).read_bytes()).hexdigest()
        source={'members':pins,'source_sha256':hashlib.sha256(remote.canonical(pins)).hexdigest()}
        executable=hashlib.sha256(Path(sys.executable).resolve().read_bytes()).hexdigest()
        value={'code_root':str(root),'source_pins':pins,'python_sha256':executable,'g_feed':{'sha256':'e'*64}}
        contents={'code-archive':source,'source-fingerprint':source,'python-executable-receipt':{'python_version':platform.python_version(),'executable_sha256':executable},'controller-executable':{'path_sha256':hashlib.sha256(str(Path(remote.__file__).resolve()).encode()).hexdigest(),'source_sha256':pins['src/vkgr/retrieval_sasrec_remote.py']},'dependency-lock':{'lock_sha256':pins['uv.lock'],'dependencies':{'numpy':'1.26.4'}},'p4h-config':{'config_sha256':pins['configs/p4h-sasrec-v2.json']},'p4h-protocol':{'protocol_sha256':pins['docs/p4h_sasrec_protocol_v2.md']},'p4h-plan':{'formal_slots':remote.plan.FORMAL_SLOTS,'task_ids':remote.plan.TASK_IDS,'plan_sha256':'ea3f7566750c4fad13dfba80fd03a5f8fdf7e81c4c68aa8912b09b3e0e732121'},'g-feed-descriptor':{'descriptor_sha256':'e'*64},'cooperative-stop-control-schema':{'request_schema':'p4h-real-stop-request-v1','receipt_schema':'p4h-real-closure-v1'}}
        self.assertEqual(remote.verify_common_bindings(value,contents)['source_sha256'],source['source_sha256'])


class BoundedSamplingTests(unittest.TestCase):
    def test_worker_checks_stop_and_clock_between_bounded_resource_samples(self):
        from unittest.mock import patch
        runs,payload=SafetyFeedTests().fixture();identity={'pid':40,'start_ticks':100,'executable_sha256':'a'*64,'start_receipt_sha256':'b'*64};jobs={'job_id_hash':'c'*64,'start_identity_hash':'d'*64}
        now=[101.]
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();(root/'control').mkdir();journal=remote.FeedJournal(root/'feed',identity,jobs)
            try:
                journal.publish(payload,now=100,status='OPEN');journal.descriptor['initial_no_other_writer_inventory_sha256']='e'*64
                desc=root/'descriptor.json';pin=remote.write_once(desc,journal.descriptor);reader=remote.SafetyFeedReader(desc,pin,runs)
                budget=remote.MonotonicBudget({'kind':'pilot','started_at':100.,'deadline':14500.,'disk_cap':20*1024**3,'attempts':1,'started_monotonic':1000.,'deadline_monotonic':15400.,'host_boot_id':'boot'},monotonic=lambda:now[0]+900,boot_id='boot')
                limits={'stage_seconds':60,'finish_seconds':120,'next_atomic_bytes':1048576,'stop_reserve_bytes':1048576,'sample_seconds':5}
                gate=remote.RuntimeGate(root,root/'control'/'stop.json',budget,reader,limits,clock=lambda:now[0])
                with patch.object(remote,'prove_publisher_runtime',return_value=True):
                    gate.check();now[0]=102.;gate.check();self.assertEqual(len(gate.samples),1)
                    now[0]=131.
                    with self.assertRaises(ValueError):gate.check()
                    now[0]=102.;(root/'control'/'stop.json').write_text('{}')
                    with self.assertRaises(ValueError):gate.check()
            finally:journal.close()


class ReviewFixTests(unittest.TestCase):
    def test_queue_cannot_leave_outputs_feed_or_code_outside_charged_tree(self):
        value=real_manifest_fixture(Path('/private/tmp/real-operation'))
        for key in ('output_root','control_root','code_root'):
            wrong=copy.deepcopy(value);wrong[key]='/private/tmp/outside-'+key
            with self.subTest(key=key),self.assertRaises(ValueError):remote.validate_remote_manifest(wrong)
        wrong=copy.deepcopy(value);wrong['g_feed']['path']='/private/tmp/outside-feed.json'
        with self.assertRaises(ValueError):remote.validate_remote_manifest(wrong)

    def test_external_authority_rejects_fresh_queue_reset_and_repeated_attempt(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();value=real_manifest_fixture(root)
            auth={'queue':copy.deepcopy(value['queue']),'attempt':{'slot_id':'pilot','number':1,'reason':'initial','previous_terminal':None}}
            remote.verify_queue_authority(value,auth)
            wrong=copy.deepcopy(value);wrong['queue']['started_at']=200;wrong['queue']['deadline']=14600
            with self.assertRaises(ValueError):remote.verify_queue_authority(wrong,auth)
            remote.reserve_queue_attempt(value,auth,'a'*64)
            wrong=copy.deepcopy(value);wrong['operation_id']='p4h-pilot-duplicate';wrong['output_root']=str(root/'other-output')
            with self.assertRaises(ValueError):remote.reserve_queue_attempt(wrong,auth,'b'*64)

    def test_running_g_without_terminal_is_safe_but_dead_owner_is_unknown(self):
        from unittest.mock import patch
        leader={'pid':40,'ppid':1,'pgid':40,'sid':40,'birth':100,'state':'S'}
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();fixed={'root':str(root),'run_id':'core-02'}
            for name,data in [('controller.py',b'controller'),('approval.json',b'{}'),('input.json',b'{"registration":{"run_id":"core-02"}}')]:
                (root/name).write_bytes(data);fixed[name.split('.')[0]+'_sha256']=hashlib.sha256(data).hexdigest()
            start={k:fixed[k] for k in ('controller_sha256','approval_sha256','input_sha256')};remote.write_once(root/'start.json',start);fixed['start_sha256']=hashlib.sha256((root/'start.json').read_bytes()).hexdigest()
            remote.write_once(root/'owner.json',{'identity':{k:leader[k] for k in ('pid','pgid','sid','birth')},'start_sha256':fixed['start_sha256']})
            with patch.object(remote,'linux_processes',return_value=[leader]),patch.object(remote,'linux_resources',return_value=({'ram':.5,'vram':.5,'disk':.5},[])):
                self.assertEqual(remote.observe_g([fixed],allowed_p=[])['runs'][0]['status'],'RUNNING')
                (root/'terminal.json').write_bytes(b' '*65536)
                self.assertEqual(remote.observe_g([fixed],allowed_p=[])['runs'][0]['status'],'RUNNING')
            with patch.object(remote,'linux_processes',return_value=[]),patch.object(remote,'linux_resources',return_value=({'ram':.5,'vram':.5,'disk':.5},[])):
                with self.assertRaises(ValueError):remote.observe_g([fixed],allowed_p=[])

    def test_publisher_stop_is_birth_bound_and_closes_before_deadline(self):
        from unittest.mock import patch
        runs,payload=SafetyFeedTests().fixture();identity={'pid':40,'start_ticks':100,'executable_sha256':'a'*64,'start_receipt_sha256':'b'*64}
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve()/'feed'
            def observe(*args,**kwargs):
                if (root/'descriptor.json').exists() and not (root/'stop.json').exists():
                    pin=hashlib.sha256((root/'descriptor.json').read_bytes()).hexdigest();remote.write_once(root/'stop.json',{'schema_id':'p4h-real-publisher-stop-v1','descriptor_sha256':pin,'publisher_process':identity})
                return payload
            before=time.monotonic()
            with patch.object(remote,'linux_publisher_identity',return_value=identity),patch.object(remote,'linux_writer_inventory',return_value={'inventory_sha256':'c'*64}),patch.object(remote,'observe_g',side_effect=observe):
                result=remote.serve_publisher(root,runs,deadline=time.time()+3,interval=.01,allowed_p=[])
            self.assertEqual(result['status'],'CLOSED');self.assertEqual(json.loads((root/'publisher-closure.json').read_bytes())['reason'],'cooperative_stop')
            self.assertEqual(json.loads((root/'publisher-closure.json').read_bytes())['descriptor_sha256'],hashlib.sha256((root/'descriptor.json').read_bytes()).hexdigest())

    def test_finish_reserve_must_cover_seal_all_graces_probe_and_final_writes(self):
        value=real_manifest_fixture(Path('/private/tmp/real-operation'));value['limits'].update(finish_seconds=1,seal_seconds=120,signal_grace_seconds=30)
        with self.assertRaises(ValueError):remote.validate_remote_manifest(value)

    def test_pilot_verifier_rejects_current_binding_different_from_sealed_worker(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();worker={'binding':{'old':'a'*64},'query_count':1,'query_identity_sha256':hashlib.sha256(b'[0]').hexdigest(),'arms':{}}
            remote.write_once(root/'pilot-worker.json',worker);rec={'role':'pilot-worker.json','path':str(root/'pilot-worker.json'),'sha256':hashlib.sha256((root/'pilot-worker.json').read_bytes()).hexdigest(),'size':(root/'pilot-worker.json').stat().st_size}
            terminal={'private_artifacts':[rec],'artifact_hashes':[]}
            with self.assertRaises(ValueError):remote.verify_actual_pilot(root,terminal,{},[{'query_id':0,'history':[1]}],gate=lambda:None,device='cpu',binding={'new':'b'*64})

    def test_cleanup_existing_stop_still_reaps_owned_process(self):
        import io
        leader={'pid':40,'ppid':10,'pgid':40,'sid':40,'birth':100,'state':'S'};kernel=FakeKernel([leader]);process=FakeWaitingProcess(kernel)
        handle={'process':process,'identity':leader,'pidfd':1,'stdout':io.BytesIO(),'stderr':io.BytesIO(),'released':True}
        lease=remote.KernelLease(handle,kernel=kernel,owner_pid=10,controller_pgid=10,baseline_children=set(),protected_pgids=set())
        with tempfile.TemporaryDirectory() as folder:
            stop=Path(folder).resolve()/'stop.json';stop.write_text('{}')
            result=lease.close(stop,manifest_sha256='a'*64,seal_seconds=.01,signal_grace_seconds=.01)
            self.assertEqual(result['state'],'CLOSED');self.assertEqual(result['actual_worker_exit'],-15)


class PredecessorCrossBindingTests(unittest.TestCase):
    def test_closed_receipts_reject_manifest_start_admission_inventory_and_input_mismatch(self):
        closure={'state':'CLOSED','manifest_sha256':'a'*64,'owned_identities':[{'pid':40,'birth':100}]}
        start={'manifest_sha256':'a'*64,'controller':{'pid':10,'birth':90}}
        admission={'manifest_sha256':'a'*64,'start_sha256':hashlib.sha256(remote.canonical(start)).hexdigest()}
        inventory=closure['owned_identities'];inputs=[{'role':'catalog','sha256':'b'*64}]
        sealed={'predecessor_operation_manifest_sha256':'a'*64,'controller_start_receipt_sha256':hashlib.sha256(remote.canonical(start)).hexdigest(),'admission_sha256':hashlib.sha256(remote.canonical(admission)).hexdigest(),'worker_birth_inventory_sha256':hashlib.sha256(remote.canonical(inventory)).hexdigest(),'final_g_receipt_sha256':'c'*64,'input_hashes':inputs}
        terminal={'manifest_sha256':'a'*64,'start_sha256':sealed['controller_start_receipt_sha256'],'admission_sha256':sealed['admission_sha256'],'final_g_receipt_sha256':'c'*64,'input_hashes':inputs}
        value={'roles':[{'role':'catalog','source':'input_hashes','sha256':'b'*64}]}
        self.assertTrue(remote.validate_predecessor_cross_bindings(value,sealed,terminal,closure,start,admission,inventory))
        for key in ('controller_start_receipt_sha256','admission_sha256','worker_birth_inventory_sha256','final_g_receipt_sha256'):
            bad=copy.deepcopy(sealed);bad[key]='d'*64
            with self.subTest(key=key),self.assertRaises(ValueError):remote.validate_predecessor_cross_bindings(value,bad,terminal,closure,start,admission,inventory)
        bad=copy.deepcopy(closure);bad['manifest_sha256']='d'*64
        with self.assertRaises(ValueError):remote.validate_predecessor_cross_bindings(value,sealed,terminal,bad,start,admission,inventory)
        bad=copy.deepcopy(value);bad['roles'][0]['sha256']='d'*64
        with self.assertRaises(ValueError):remote.validate_predecessor_cross_bindings(bad,sealed,terminal,closure,start,admission,inventory)


class RealPrecisionTests(unittest.TestCase):
    def test_real_execution_disables_tf32_and_records_actual_fp32_determinism(self):
        import torch
        prior=torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32=True
            execution=remote.configure_real_torch()
            self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
            self.assertFalse(torch.backends.cudnn.allow_tf32)
            self.assertTrue(torch.are_deterministic_algorithms_enabled())
            self.assertEqual(execution,{'precision':'fp32','tf32':False,'deterministic':True,'cublas_workspace':':4096:8'})
        finally:torch.backends.cuda.matmul.allow_tf32=prior


class ForeignUIDVisibilityTests(unittest.TestCase):
    def test_readonly_uid_scope_receipt_is_birth_uid_and_executable_bound(self):
        import os,sys,subprocess
        from unittest.mock import patch
        row={'pid':900,'ppid':1,'pgid':900,'sid':900,'birth':100,'state':'S'}
        uid=1001 if os.getuid()!=1001 else 1002
        credentials={'uid':[uid]*4,'gid':[uid]*4}
        receipt={'schema_id':'p4h-readonly-uid-fd-v1','observer_uid':uid,'pid':900,'birth':100,'uid':[uid]*4,'python_sha256':hashlib.sha256(Path(sys.executable).resolve().read_bytes()).hexdigest(),'fd_modes':[],'unknown_fds':0}
        with patch.object(remote,'pin_foreign_observer_process',return_value=44),patch.object(remote,'foreign_observer_exited',return_value=False),patch('os.close'),patch.object(remote,'linux_identity',return_value=row),patch.object(remote,'linux_proc_credentials',return_value=credentials),patch('shutil.which',return_value='/usr/bin/setpriv'):
            with patch('subprocess.run',return_value=subprocess.CompletedProcess([],0,json.dumps(receipt).encode(),b'')):
                self.assertEqual(remote.foreign_uid_fd_modes(row,{'feed_root':'/private/tmp/owned-feed','feed_owner_uid':os.getuid()}),[])
            for field,bad in [('observer_uid',os.getuid()),('birth',101),('unknown_fds',1),('python_sha256','a'*64),('fd_modes',['w'])]:
                wrong={**receipt,field:bad}
                with self.subTest(field=field),patch('subprocess.run',return_value=subprocess.CompletedProcess([],0,json.dumps(wrong).encode(),b'')),self.assertRaises(ValueError):
                    remote.foreign_uid_fd_modes(row,{'feed_root':'/private/tmp/owned-feed','feed_owner_uid':os.getuid()})

    def test_same_uid_or_mixed_credentials_never_bypass_unavailable_fd_inventory(self):
        import os
        from unittest.mock import patch
        row={'pid':900,'ppid':1,'pgid':900,'sid':900,'birth':100,'state':'S'}
        for ids in ([os.getuid()]*4,[1001,0,1001,1001]):
            with patch.object(remote,'pin_foreign_observer_process',return_value=44),patch.object(remote,'foreign_observer_exited',return_value=False),patch('os.close'),patch.object(remote,'linux_proc_credentials',return_value={'uid':ids,'gid':[1001]*4}),self.assertRaises(ValueError):
                remote.foreign_uid_fd_modes(row,{'feed_root':'/private/tmp/owned-feed','feed_owner_uid':os.getuid()})


class MatchedCapacityCohortTests(unittest.TestCase):
    def fixture(self):
        return [{'query_id':0,'history':[1]}, {'query_id':1,'history':[1,2]},
                {'query_id':2,'history':[1,2,3]}, {'query_id':3,'history':[1,2,3,4]},
                {'query_id':4,'history':[2,3]}, {'query_id':5,'history':[2,3,4]},
                {'query_id':7,'history':[9]}]

    def test_matched_long_history_cohort_is_train_only_and_order_invariant(self):
        original=self.fixture();before=copy.deepcopy(original)
        cohort=remote.select_capacity_cohort(original,per_arm=2,seed=42,lineage='train')
        self.assertEqual([r['query_id'] for r in cohort['like']],[2,4])
        self.assertEqual([r['query_id'] for r in cohort['observed']],[3,5])
        self.assertEqual(cohort,remote.select_capacity_cohort(list(reversed(original)),per_arm=2,seed=42,lineage='train'))
        self.assertEqual(original,before)

    def test_wrong_lineage_duplicate_target_and_insufficient_pairs_fail(self):
        for lineage in ['validation','test','unknown']:
            with self.assertRaises(ValueError):remote.select_capacity_cohort(self.fixture(),per_arm=2,lineage=lineage)
        for rows in [self.fixture()+[self.fixture()[0]],self.fixture()[:3]+self.fixture()[4:],
                     [{**r,'target':1} for r in self.fixture()],
                     [{**r,'history':[]} for r in self.fixture()]]:
            with self.assertRaises(ValueError):remote.select_capacity_cohort(rows,per_arm=2)
        with self.assertRaises(ValueError):remote.select_capacity_cohort(self.fixture(),per_arm=4)
        with self.assertRaises(ValueError):remote.select_capacity_cohort(self.fixture(),per_arm=True)

    def test_actual_worker_scores_only_own_arm_cohort_and_records_identity(self):
        catalog=list(range(1,85));mapping={i:i for i in catalog}
        rows=[{'example_id':j,'user_id':j,'history':[1,2],'history_positions':[0,1],'target':3+j%80,'target_position':2+j,'past_likes':[1,2]} for j in range(512)]
        sidecar=[{'query_id':0,'history':[1,2]},{'query_id':1,'history':[2,1]}]
        data={'catalog':catalog,'idmap':mapping,'supported_indices':list(range(1,85)),
              'rows_by_arm':{'like':rows,'observed':rows},'sidecar':sidecar}
        with tempfile.TemporaryDirectory() as folder:
            result=remote.run_pilot_worker(data,Path(folder).resolve(),binding={'source':'a'*64},gate=lambda:None,device='cpu',steps=1,capacity_per_arm=1)
            self.assertEqual(result['capacity_per_arm'],1)
            for arm,row in result['arms'].items():
                self.assertEqual(row['capacity_query_count'],1)
                self.assertEqual(row['top80_count'],[80])
                self.assertEqual(row['query_identity_sha256'],remote.sha(remote.canonical([0 if arm=='like' else 1])))


class CapacityPreparationTests(unittest.TestCase):
    def test_cohort_materialization_keeps_complete_training_examples(self):
        from vkgr.retrieval_data import prefix_examples
        events=[{'user_id':1,'item_id':10,'like':True},{'user_id':1,'item_id':20,'like':False},{'user_id':1,'item_id':30,'like':True},{'user_id':2,'item_id':10,'like':True},{'user_id':2,'item_id':20,'like':True}]
        examples=list(prefix_examples([(0,events)]));before=copy.deepcopy(examples)
        original={'items':[10,20,30],'pad':0,'start':1,'supported_indices':[1,2,3]}
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();result=remote.materialize_training_inputs(examples,[10,20,30],original,[10,20,30],[(0,events)],root/'roles',capacity_per_arm=1)
            self.assertEqual(result['sidecar'],[{'query_id':2,'history':[10,30]},{'query_id':3,'history':[10,20,30]}])
            self.assertEqual(len(result['rows_by_arm']['like']),len(examples))
            self.assertEqual(len(result['rows_by_arm']['observed']),len(examples))
            self.assertEqual(examples,before)
            self.assertEqual(json.loads((root/'roles/train-prefix-target-free-sidecar.json').read_bytes()),result['sidecar'])


class ExplicitPilotExtensionTests(unittest.TestCase):
    def fixture(self,root):
        from contextlib import ExitStack
        from unittest.mock import patch
        old=root/'old';old.mkdir();new=old/'extension-03';new.mkdir()
        parent={'kind':'pilot','started_at':100,'deadline':14500,'disk_cap':20*1024**3,'attempts':1,'started_monotonic':100,'deadline_monotonic':14500,'host_boot_id':'boot'}
        def bound(path,value):
            pin=remote.write_once(path,value);return {'path':str(path),'sha256':pin}
        parent_ref=bound(old/'queue.json',parent)
        failures=[]
        for n in (1,2):
            failures.append(bound(old/f'pilot-driver-result-0{n}.json',{'status':'failed+spent','failure':'ValueError: real publisher bootstrap failed','terminal':None,'finished_at':1000+n,'queue':{**parent,'attempts':n},'driver_closure':{'state':'CLOSED','actual_worker_exit':-15}}))
        grant={'schema_id':'pg-r01-human-continue-v1','pilot':{'additional_attempts':1,'cumulative_attempt_cap':3,'additional_wall_seconds':14400,'cumulative_authorized_wall_cap_seconds':28800,'total_disk_bytes':20*1024**3,'preserve_old_spent':True}}
        grant_ref=bound(new/'human-grant.json',grant)
        ext={'schema_id':'p4h-explicit-pilot-extension-v1','parent_queue':parent_ref,'failure_receipts':failures,'human_grant':grant_ref}
        queue={**parent,'started_at':20000,'deadline':34400,'started_monotonic':20000,'deadline_monotonic':34400,'attempts':3,'extension':ext,'receipt_path':str(new/'queue.json'),'receipt_sha256':'a'*64}
        auth={'queue':queue,'attempt':{'slot_id':'pilot','number':3,'reason':'explicit_budget_extension','previous_terminal':None}}
        stack=ExitStack();stack.enter_context(patch.object(remote,'PILOT_RECOVERY_PARENT_SHA256',parent_ref['sha256']));stack.enter_context(patch.object(remote,'PILOT_RECOVERY_FAILURE_SHA256S',tuple(r['sha256'] for r in failures)));stack.enter_context(patch.object(remote,'PILOT_RECOVERY_GRANT_SHA256',grant_ref['sha256']))
        return queue,auth,stack,parent_ref

    def test_extension_requires_pinned_human_grant_and_preserves_parent_and_total_attempts(self):
        with tempfile.TemporaryDirectory() as folder:
            queue,auth,stack,parent=self.fixture(Path(folder).resolve())
            with stack:
                before=Path(parent['path']).read_bytes()
                budget=remote.MonotonicBudget(queue,monotonic=lambda:20001,boot_id='boot')
                self.assertEqual(budget.attempts,3);self.assertEqual(budget.remaining_seconds(20001),14399)
                self.assertEqual(remote.queue_charge_root(queue),Path(parent['path']).parent)
                value={'queue':queue,'slot':None,'stage':'pilot','operation_id':'p4h-recovery','output_root':'/out','control_root':'/ctl'}
                self.assertTrue(remote.verify_queue_authority(value,auth))
                remote.reserve_queue_attempt(value,auth,'b'*64)
                with self.assertRaises(ValueError):remote.reserve_queue_attempt(value,auth,'b'*64)
                self.assertEqual(Path(parent['path']).read_bytes(),before)
                with self.assertRaises(ValueError):budget.next_attempt(now=20002,reason='infrastructure_failure',optimizer_steps=0)

    def test_unpinned_grant_changed_parent_missing_failed_spent_and_fourth_attempt_reject(self):
        with tempfile.TemporaryDirectory() as folder:
            queue,auth,stack,parent=self.fixture(Path(folder).resolve())
            with stack:
                for change in ['wrong_grant','missing_failure','attempt4','earlier_clock']:
                    q=copy.deepcopy(queue)
                    if change=='wrong_grant':q['extension']['human_grant']['sha256']='f'*64
                    elif change=='missing_failure':q['extension']['failure_receipts']=q['extension']['failure_receipts'][:1]
                    elif change=='attempt4':q['attempts']=4
                    else:q['started_at']=100;q['deadline']=14500;q['started_monotonic']=100;q['deadline_monotonic']=14500
                    a={'queue':q,'attempt':{**auth['attempt'],'number':q['attempts']}}
                    with self.assertRaises(ValueError):remote.verify_queue_authority({'queue':q,'slot':None},a)
                Path(parent['path']).write_text('{}')
                with self.assertRaises(ValueError):remote.verify_queue_authority({'queue':queue,'slot':None},auth)


class PreworkerContinuationTests(unittest.TestCase):
    fixture=ExplicitPilotExtensionTests.fixture
    def test_explicit_preworker_grant_preserves_registered_attempt_and_refuses_any_worker(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();queue,auth,stack,parent=self.fixture(root)
            with stack:
                old=Path(parent['path']).parent;ext=Path(queue['receipt_path']).parent
                prior=real_manifest_fixture(old);prior['queue']=queue
                prior['control_root']=str(ext/'prior-control');prior['output_root']=str(ext/'prior-output')
                ctl=Path(prior['control_root']);ctl.mkdir();Path(prior['output_root']).mkdir()
                manifest_pin=remote.write_once(ctl/'operation.json',prior)
                remote.reserve_queue_attempt(prior,auth,manifest_pin)
                controller={'pid':99999991,'pgid':99999991,'sid':99999991,'birth':100,'ppid':99999993,'state':'S'}
                remote.write_once(ctl/'start.json',{'manifest_sha256':manifest_pin,'controller':controller})
                closure={'state':'CLOSED','actual_worker_exit':-15,'owned_sessions':[99999991],'owned_identities':[controller]}
                cpath=ext/'pilot-operation.driver-closure.json';cpin=remote.write_once(cpath,closure)
                publisher={'pid':99999992,'pgid':99999992,'sid':99999992,'birth':101,'ppid':99999993,'state':'S'}
                driver={'status':'failed+spent','queue':queue,'publisher_driver_closure':{'state':'CLOSED','actual_worker_exit':0,'owned_sessions':[99999992],'owned_identities':[publisher]}}
                dpath=ext/'driver-result.json';dpin=remote.write_once(dpath,driver)
                remote.write_once(ext/'driver-intent.json',{'queue':queue,'source_sha256':remote.sha(remote.canonical(prior['source_pins'])),'identity':{'pid':99999993,'pgid':99999993,'sid':99999993,'birth':99,'ppid':1,'state':'S'}})
                feed=ext/'feed';feed.mkdir()
                desc={'feed_root':str(feed),'publisher_process':{'pid':99999992,'start_ticks':101,'executable_sha256':prior['python_sha256'],'start_receipt_sha256':'f'*64}}
                descpin=remote.write_once(feed/'descriptor.json',desc)
                prior['g_feed']={'path':str(feed/'descriptor.json'),'sha256':descpin}
                # Refresh bound manifest/registration for the real fixture feed.
                (ctl/'operation.json').write_bytes(remote.canonical(prior));manifest_pin=remote.operation._file_sha256(ctl/'operation.json')
                (old/'attempt-pilot-3.json').unlink();remote.reserve_queue_attempt(prior,auth,manifest_pin)
                (ctl/'start.json').write_bytes(remote.canonical({'manifest_sha256':manifest_pin,'controller':controller}))
                remote.write_once(feed/'publisher-closure.json',{'status':'CLOSED','reason':'cooperative_stop','descriptor_sha256':descpin,'publisher_process':desc['publisher_process'],'lock_fd_closed':True})
                gpath=ext/'preworker-grant.json';gpin=remote.write_once(gpath,{'human_answer':'允许原窗口内自主修复准入编排','scope':'pre-worker only; oldclock/spent; no scientific retry'})
                apath=old/'attempt-pilot-3.json';before=apath.read_bytes()
                repair={'schema_id':'p4h-explicit-preworker-repair-v1','generation':1,'human_grant':{'path':str(gpath),'sha256':gpin},'previous_attempt':{'path':str(apath),'sha256':remote.operation._file_sha256(apath)},'previous_driver':{'path':str(dpath),'sha256':dpin},'previous_controller_closure':{'path':str(cpath),'sha256':cpin}}
                value=copy.deepcopy(prior);value['operation_id']='p4h-preworker-repair01';value['output_root']=str(ext/'next-output');value['control_root']=str(ext/'next-control')
                authorization={**auth,'preworker_repair':repair}
                with patch.object(remote,'PREWORKER_REPAIR_GRANT_SHA256',gpin,create=True),patch.object(remote,'linux_processes',return_value=[]):
                    remote.write_once(ctl/'worker-start.json',{'worker':'registered'})
                    with self.assertRaises(ValueError):remote.reserve_queue_attempt(value,authorization,'c'*64)
                    (ctl/'worker-start.json').unlink()
                    # Reviewer reproduction: self-reported sessions must not hide
                    # the live bound controller; wrong driver queue/empty pub rejected.
                    bad_closed={**closure,'owned_sessions':[99999994]}
                    bad_driver={**driver,'queue':{**queue,'deadline':queue['deadline']-1},'publisher_driver_closure':{'state':'CLOSED','actual_worker_exit':0}}
                    saved_c=cpath.read_bytes();saved_d=dpath.read_bytes()
                    cpath.write_bytes(remote.canonical(bad_closed));dpath.write_bytes(remote.canonical(bad_driver))
                    tampered=copy.deepcopy(authorization);tampered['preworker_repair']['previous_controller_closure']['sha256']=remote.operation._file_sha256(cpath);tampered['preworker_repair']['previous_driver']['sha256']=remote.operation._file_sha256(dpath)
                    with patch.object(remote,'linux_processes',return_value=[controller]),self.assertRaises(ValueError):remote.reserve_queue_attempt(value,tampered,'c'*64)
                    cpath.write_bytes(saved_c);dpath.write_bytes(saved_d)
                    record=remote.reserve_queue_attempt(value,authorization,'c'*64)
                    self.assertEqual(record['queue'],queue)
                    self.assertEqual(apath.read_bytes(),before)
                    self.assertTrue((old/'attempt-pilot-3-preworker-1.json').is_file())
                    with self.assertRaises(ValueError):remote.reserve_queue_attempt(value,authorization,'c'*64)
                    bad=copy.deepcopy(authorization);bad['preworker_repair']['human_grant']['sha256']='d'*64
                    with self.assertRaises(ValueError):remote.reserve_queue_attempt(value,bad,'d'*64)


class LiveFeedProbeClockTests(unittest.TestCase):
    def test_publication_during_real_probe_uses_elapsed_clock_but_future_receipt_rejects(self):
        from unittest.mock import patch
        runs,payload=SafetyFeedTests().fixture()
        identity={'pid':40,'start_ticks':100,'executable_sha256':'a'*64,'start_receipt_sha256':'b'*64};jobs={'job_id_hash':'c'*64,'start_identity_hash':'d'*64}
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();journal=remote.FeedJournal(root/'feed',identity,jobs)
            try:
                journal.publish(payload,now=100,status='OPEN');journal.descriptor['initial_no_other_writer_inventory_sha256']='e'*64
                desc=root/'descriptor.json';pin=remote.write_once(desc,journal.descriptor);reader=remote.SafetyFeedReader(desc,pin,runs)
                clock=[10.];calls=[0]
                def probe(_):
                    clock[0]+=2.;calls[0]+=1
                    sample={**payload,'observed_at':102.}
                    journal.publish(sample,now=102. if calls[0]==1 else 110.,status='OPEN')
                    return True
                with patch.object(remote.time,'monotonic',side_effect=lambda:clock[0]),patch.object(remote,'prove_publisher_runtime',side_effect=probe):
                    result=reader.observe(now=101.)
                    self.assertEqual(result['receipt_sha256'],journal.previous)
                    with self.assertRaises(ValueError):reader.observe(now=104.)
            finally:journal.close()


class ExplicitClockExtensionTests(unittest.TestCase):
    fixture=ExplicitPilotExtensionTests.fixture
    def test_clock_adds_only_granted_7200_without_rebasing_original_queue_or_attempt(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();q,auth,stack,parent=self.fixture(root)
            with stack:
                base={k:v for k,v in q.items() if k not in ('receipt_path','receipt_sha256')}
                q['receipt_sha256']=remote.write_once(q['receipt_path'],base)
                ext=Path(q['receipt_path']).parent
                grant={'schema_id':'pgr01-human-clock-extension-v1','human_answer':'批准有限追加2h','original_queue':{'path':q['receipt_path'],'sha256':q['receipt_sha256']},'original_started_at':q['started_at'],'original_deadline':q['deadline'],'additional_wall_seconds':7200,'effective_deadline':q['deadline']+7200,'attempt_cap':3,'additional_science_attempts':0,'disk_cap':q['disk_cap'],'preserve_original_bytes_starts_spent':True,'no_science_retry_after_release':True}
                gpath=ext/'clock-human.json';gpin=remote.write_once(gpath,grant)
                cpath=ext/'clock-receipt.json'
                receipt={'schema_id':'p4h-explicit-clock-receipt-v1','original_queue':grant['original_queue'],'human_grant':{'path':str(gpath),'sha256':gpin},'additional_seconds':7200,'original_started_at':q['started_at'],'original_started_monotonic':q['started_monotonic'],'original_deadline':q['deadline'],'original_deadline_monotonic':q['deadline_monotonic'],'effective_deadline':q['deadline']+7200,'effective_deadline_monotonic':q['deadline_monotonic']+7200,'host_boot_id':q['host_boot_id'],'attempts':3,'disk_cap':q['disk_cap']}
                cpin=remote.write_once(cpath,receipt)
                clock={'schema_id':'p4h-explicit-clock-extension-v1','additional_seconds':7200,'original_queue':grant['original_queue'],'human_grant':{'path':str(gpath),'sha256':gpin},'receipt':{'path':str(cpath),'sha256':cpin}}
                before=Path(q['receipt_path']).read_bytes();extended={**q,'clock_extension':clock}
                with patch.object(remote,'CLOCK_EXTENSION_GRANT_SHA256',gpin,create=True):
                    budget=remote.MonotonicBudget(extended,monotonic=lambda:20001,boot_id='boot')
                    self.assertEqual(budget.remaining_seconds(20001),21599)
                    self.assertEqual(budget.deadline_mono,41600)
                    self.assertEqual(budget.attempts,3)
                    self.assertEqual(Path(q['receipt_path']).read_bytes(),before)
                    for change in ('start','amount','grant','disk','attempt'):
                        bad=copy.deepcopy(extended)
                        if change=='start':bad['started_at']+=1;bad['deadline']+=1;bad['started_monotonic']+=1;bad['deadline_monotonic']+=1
                        elif change=='amount':bad['clock_extension']['additional_seconds']=7201
                        elif change=='grant':bad['clock_extension']['human_grant']['sha256']='f'*64
                        elif change=='disk':bad['disk_cap']+=1
                        else:bad['attempts']=4
                        with self.subTest(change=change),self.assertRaises(ValueError):remote.MonotonicBudget(bad,monotonic=lambda:20002,boot_id='boot')


class ClockAuthorityOrderingTests(unittest.TestCase):
    fixture=ExplicitPilotExtensionTests.fixture
    def test_missing_clock_receipt_fails_before_science_bytes_are_parsed(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();q,auth,stack,parent=self.fixture(root)
            with stack:
                value=real_manifest_fixture(Path(parent['path']).parent);value['queue']=q
                ref={'path':q['receipt_path'],'sha256':q['receipt_sha256']}
                q['clock_extension']={'schema_id':'p4h-explicit-clock-extension-v1','additional_seconds':7200,'original_queue':ref,'human_grant':{'path':str(root/'old/extension-03/missing-grant.json'),'sha256':remote.CLOCK_EXTENSION_GRANT_SHA256},'receipt':{'path':str(root/'old/extension-03/missing-clock.json'),'sha256':'f'*64}}
                rows=sorted(value['roles'],key=lambda r:r['role']!='prepared-train-rows');value['roles']=rows
                science=Path(rows[0]['path']);science.parent.mkdir(parents=True,exist_ok=True)
                blob=b'SCIENCE CONTENT MUST NOT BE PARSED BEFORE AUTHORITY'
                science.write_bytes(blob);rows[0].update(sha256=remote.sha(blob),size=len(blob))
                remote.validate_remote_manifest(value)
                # Only the public-source package boundary is stubbed; read_roles
                # is real and would consume/parse the sentinel science bytes.
                with patch.object(remote,'verify_source_tree',return_value={'members':{},'source_sha256':'a'*64}):
                    for entry in (remote.prepare_controller,remote.worker_entry):
                        with self.subTest(entry=entry.__name__),self.assertRaises(FileNotFoundError):entry(value,'a'*64,[])


class ForeignUIDExitRaceTests(unittest.TestCase):
    def test_pidfd_confirmed_exit_is_not_a_live_visibility_failure(self):
        from unittest.mock import patch
        row={'pid':900,'ppid':1,'pgid':900,'sid':900,'birth':100,'state':'S'}
        with patch.object(remote,'pin_foreign_observer_process',return_value=44,create=True),patch.object(remote,'confirm_foreign_observer_exit',return_value=True,create=True),patch.object(remote,'_foreign_uid_fd_modes_live',side_effect=ValueError('observer failed'),create=True) as inspect,patch('os.close') as close:
            self.assertEqual(remote.foreign_uid_fd_modes(row,{}),[])
            inspect.assert_not_called();close.assert_called_once_with(44)

    def test_exit_during_helper_is_confirmed_by_same_pidfd(self):
        from unittest.mock import patch
        row={'pid':900,'birth':100}
        for error in (ValueError('observer failed'),FileNotFoundError('proc gone'),PermissionError('fd gone')):
            with self.subTest(error=type(error).__name__),patch.object(remote,'pin_foreign_observer_process',return_value=44,create=True),patch.object(remote,'confirm_foreign_observer_exit',side_effect=[False,True],create=True),patch.object(remote,'_foreign_uid_fd_modes_live',side_effect=error,create=True),patch('os.close') as close:
                self.assertEqual(remote.foreign_uid_fd_modes(row,{}),[])
                close.assert_called_once_with(44)

    def test_live_visibility_or_writer_failure_is_never_suppressed(self):
        from unittest.mock import patch
        row={'pid':900,'birth':100}
        for error in (ValueError('writer'),FileNotFoundError('incomplete'),PermissionError('inaccessible')):
            with self.subTest(error=type(error).__name__),patch.object(remote,'pin_foreign_observer_process',return_value=44,create=True),patch.object(remote,'confirm_foreign_observer_exit',return_value=False,create=True),patch.object(remote,'_foreign_uid_fd_modes_live',side_effect=error,create=True),patch('os.close') as close,self.assertRaises(type(error)):
                remote.foreign_uid_fd_modes(row,{})
            close.assert_called_once_with(44)

    def test_birth_drift_closes_pin_and_stops(self):
        from unittest.mock import patch
        row={'pid':900,'ppid':1,'pgid':900,'sid':900,'birth':100,'state':'S'}
        with patch('os.pidfd_open',return_value=44,create=True),patch.object(remote,'linux_identity',return_value={**row,'birth':101}),patch('os.close') as close,self.assertRaisesRegex(ValueError,'birth drift'):
            remote.pin_foreign_observer_process(row)
        close.assert_called_once_with(44)

    def test_disappearance_after_pin_requires_kernel_exit_confirmation(self):
        from unittest.mock import patch
        for exited in (True,False):
            with self.subTest(exited=exited),patch('os.pidfd_open',return_value=44,create=True),patch.object(remote,'linux_identity',side_effect=FileNotFoundError()),patch.object(remote,'foreign_observer_exited',return_value=exited),patch('os.close') as close:
                if exited:
                    self.assertEqual(remote.pin_foreign_observer_process({'pid':900}),44)
                    close.assert_not_called()
                else:
                    with self.assertRaises(FileNotFoundError):remote.pin_foreign_observer_process({'pid':900})
                    close.assert_called_once_with(44)

    def test_old_pidfd_exit_never_hides_live_reused_pid(self):
        from unittest.mock import patch
        row={'pid':900,'ppid':1,'pgid':900,'sid':900,'birth':100,'state':'S'}
        with patch.object(remote,'foreign_observer_exited',return_value=True),patch.object(remote,'linux_identity',return_value={**row,'birth':101}),self.assertRaisesRegex(ValueError,'birth drift'):
            remote.confirm_foreign_observer_exit(row,44)

    def test_detected_identity_drift_is_not_an_exit_failure(self):
        from unittest.mock import patch
        row={'pid':900,'birth':100}
        with patch.object(remote,'pin_foreign_observer_process',return_value=44),patch.object(remote,'confirm_foreign_observer_exit',side_effect=[False,True],create=True),patch.object(remote,'foreign_observer_exited',side_effect=[False,True]),patch.object(remote,'_foreign_uid_fd_modes_live',side_effect=remote.ForeignUIDIdentityDrift('birth drift')),patch('os.close'),self.assertRaisesRegex(ValueError,'birth drift'):
            remote.foreign_uid_fd_modes(row,{})


class AutonomousContinuationTests(unittest.TestCase):
    def fixture(self):
        root='/private/tmp/owned-pilot';ref=lambda name,pin:{'path':root+'/'+name,'sha256':pin}
        extension={'schema_id':'p4h-autonomous-pilot-continuation-v1','parent_queue':ref('extension-03/queue.json','6db7532096823748ec9bc26bfd0c6e12a8322b2b68d26c193af55854f4ae6598'),'human_grant':ref('attempt04/autonomous-grant.json','f0d487472254b19087e7144bd57d191ba0c335473150bdd4a46cd65021de186a'),'previous_driver':ref('extension-03/preworker02/driver-result.json','63f98bddecea9aa011388948586dae508e96dcc6c4f3488f08e213cfdb06c150'),'previous_closure':ref('extension-03/preworker02/pilot-operation.driver-closure.json','bcf3bee934caf3e475dbac63ace1a9d71f63d74250864ebfa6e35bbbc446580e')}
        queue={'kind':'pilot','started_at':200.,'deadline':14600.,'disk_cap':20*1024**3,'attempts':4,'started_monotonic':1200.,'deadline_monotonic':15600.,'host_boot_id':'boot','receipt_path':root+'/attempt04/queue.json','receipt_sha256':'a'*64,'extension':extension}
        return queue

    def test_attempt4_needs_exact_direct_human_and_previous_spent_pins(self):
        queue=self.fixture();remote.Budget('pilot',**{k:queue[k] for k in ('started_at','deadline','disk_cap','attempts','extension')})
        for key in ('human_grant','previous_driver','previous_closure','parent_queue'):
            bad=copy.deepcopy(queue);bad['extension'][key]['sha256']='b'*64
            with self.subTest(key=key),self.assertRaises(ValueError):remote.Budget('pilot',**{k:bad[k] for k in ('started_at','deadline','disk_cap','attempts','extension')})
        for attempts in (3,5):
            with self.subTest(attempts=attempts),self.assertRaises(ValueError):remote.Budget('pilot',**{**{k:queue[k] for k in ('started_at','deadline','disk_cap','attempts','extension')},'attempts':attempts})

    def test_invalid_continuation_stops_before_science_reads(self):
        from unittest.mock import patch
        queue=self.fixture()
        with patch.object(remote,'_read_json',side_effect=FileNotFoundError('missing human authority')),self.assertRaises(FileNotFoundError):remote.verify_clock_extension(queue)

    def test_new_queue_has_one_finite_window_and_same_charge_root(self):
        queue=self.fixture();self.assertEqual(remote.queue_charge_root(queue),Path('/private/tmp/owned-pilot'))
        for field,value in (('deadline',14601.),('disk_cap',20*1024**3+1)):
            bad={**queue,field:value}
            with self.subTest(field=field),self.assertRaises(ValueError):remote.Budget('pilot',**{k:bad[k] for k in ('started_at','deadline','disk_cap','attempts','extension')})

    def authority_fixture(self):
        q=self.fixture();ext=q['extension'];previous={**q,'attempts':3,'receipt_path':ext['parent_queue']['path'],'receipt_sha256':ext['parent_queue']['sha256'],'started_at':0.,'started_monotonic':1000.}
        grant={'schema_id':'pgr01-human-autonomous-continue-v1','human_answer':'不用管限制，一直做下去','preserve_old_spent_and_artifact_bytes':True,'operational_envelope':{'next_cumulative_attempt':4,'new_wall_seconds':14400,'total_old_plus_new_disk_bytes':20*1024**3,'resume':0,'new_resources':False,'formal_submit':False}}
        identity={'pid':900,'ppid':899,'sid':900};pubidentity={'pid':901,'ppid':899,'sid':901}
        closed={'state':'CLOSED','actual_worker_exit':-15,'owned_sessions':[900],'owned_identities':[identity]}
        failed={'status':'failed+spent','cleanup_failures':[],'stages':[],'formal_submit':False,'queue':previous,'finished_at':100.,'publisher_driver_closure':{'state':'CLOSED','actual_worker_exit':0,'owned_sessions':[901],'owned_identities':[pubidentity]}}
        values={ext['human_grant']['path']:grant,ext['previous_driver']['path']:failed,ext['previous_closure']['path']:closed}
        return q,values

    def test_actual_authority_requires_failed_closed_and_fresh_absence(self):
        from unittest.mock import patch
        queue,values=self.authority_fixture();reader=lambda path,**kw:values[str(path)]
        with patch.object(remote,'_read_json',side_effect=reader),patch.object(remote,'verify_explicit_extension'),patch.object(remote,'verify_clock_extension'),patch.object(remote,'linux_processes',return_value=[]):
            self.assertEqual(remote.verify_autonomous_continuation(queue)['cumulative_attempts'],4)
        for processes in ([{'pid':999,'sid':900}],[{'pid':899,'sid':999}]):
            with self.subTest(processes=processes),patch.object(remote,'_read_json',side_effect=reader),patch.object(remote,'verify_explicit_extension'),patch.object(remote,'verify_clock_extension'),patch.object(remote,'linux_processes',return_value=processes),self.assertRaisesRegex(ValueError,'not closed'):remote.verify_autonomous_continuation(queue)
        for field,bad in (('status','success'),('cleanup_failures',['unsafe']),('stages',['finished'])):
            altered=copy.deepcopy(values);altered[queue['extension']['previous_driver']['path']][field]=bad
            with self.subTest(field=field),patch.object(remote,'_read_json',side_effect=lambda path,**kw:altered[str(path)]),patch.object(remote,'verify_explicit_extension'),patch.object(remote,'verify_clock_extension'),self.assertRaises(ValueError):remote.verify_autonomous_continuation(queue)


class TransientForeignVisibilityTests(unittest.TestCase):
    def test_one_bounded_retry_requires_complete_same_identity_inventory(self):
        from unittest.mock import patch
        row={'pid':900,'ppid':1,'pgid':900,'sid':900,'birth':100,'state':'S'};credentials={'uid':[1001]*4,'gid':[1001]*4}
        error=remote.ForeignUIDProbeUnavailable('transient',credentials)
        with patch.object(remote,'pin_foreign_observer_process',return_value=44),patch.object(remote,'confirm_foreign_observer_exit',return_value=False),patch.object(remote,'linux_identity',return_value=row),patch.object(remote,'linux_proc_credentials',return_value=credentials),patch.object(remote,'_foreign_uid_fd_modes_live',side_effect=[error,[]]) as probe,patch('time.sleep'),patch('os.close'):
            self.assertEqual(remote.foreign_uid_fd_modes(row,{}),[]);self.assertEqual(probe.call_count,2)

    def test_persistent_invisibility_and_uid_drift_never_pass(self):
        from unittest.mock import patch
        row={'pid':900,'ppid':1,'pgid':900,'sid':900,'birth':100,'state':'S'};credentials={'uid':[1001]*4,'gid':[1001]*4}
        error=remote.ForeignUIDProbeUnavailable('persistent',credentials)
        for actual in (credentials,{'uid':[1002]*4,'gid':[1002]*4}):
            with self.subTest(actual=actual),patch.object(remote,'pin_foreign_observer_process',return_value=44),patch.object(remote,'confirm_foreign_observer_exit',return_value=False),patch.object(remote,'linux_identity',return_value=row),patch.object(remote,'linux_proc_credentials',return_value=actual),patch.object(remote,'_foreign_uid_fd_modes_live',side_effect=error) as probe,patch('time.sleep'),patch('os.close'),self.assertRaises(ValueError):remote.foreign_uid_fd_modes(row,{})
            self.assertLessEqual(probe.call_count,2)

class Attempt05ContinuationTests(unittest.TestCase):
    def test_attempt5_requires_the_exact_failed_attempt4_chain(self):
        q=AutonomousContinuationTests().fixture();ext=q['extension'];root=Path(ext['parent_queue']['path']).parents[1]
        q.update(attempts=5,receipt_path=str(root/'attempt05/queue.json'))
        ext['human_grant']['path']=str(root/'attempt05/autonomous-grant.json')
        ext['parent_queue']={'path':str(root/'attempt04/queue.json'),'sha256':'79639fa09a53fc373cf8ed90495c930132b63f65ba7c8cf548b27a446c669d6d'}
        ext['previous_driver']={'path':str(root/'attempt04/driver-result.json'),'sha256':'c00255718358fea03b24657b608ad5b26466b12b39c53e886bf721f5d9a050f5'}
        ext['previous_closure']={'path':str(root/'attempt04/pilot-operation.driver-closure.json'),'sha256':'a66187c6158a4712ce1ba7d771518048ffc7941a00665524d350335815c477af'}
        remote.Budget('pilot',**{k:q[k] for k in ('started_at','deadline','disk_cap','attempts','extension')});self.assertEqual(remote.queue_charge_root(q),root)
        wrong=copy.deepcopy(q);wrong['extension']['previous_driver']['sha256']=remote.AUTONOMOUS_PREVIOUS_DRIVER_SHA256
        with self.assertRaises(ValueError):remote.Budget('pilot',**{k:wrong[k] for k in ('started_at','deadline','disk_cap','attempts','extension')})

class VisibilityRecheckDriftTests(unittest.TestCase):
    def test_drift_during_sleep_or_after_second_probe_is_never_accepted(self):
        from unittest.mock import patch
        row={'pid':900,'ppid':1,'pgid':900,'sid':900,'birth':100,'state':'S'};old={'uid':[1001]*4,'gid':[1001]*4};new={'uid':[1002]*4,'gid':[1002]*4};error=remote.ForeignUIDProbeUnavailable('transient',old)
        for sequence in ([old,new],[old,old,new]):
            with self.subTest(sequence=sequence),patch.object(remote,'pin_foreign_observer_process',return_value=44),patch.object(remote,'confirm_foreign_observer_exit',return_value=False),patch.object(remote,'linux_identity',return_value=row),patch.object(remote,'linux_proc_credentials',side_effect=sequence),patch.object(remote,'_foreign_uid_fd_modes_live',side_effect=[error,[]]),patch('time.sleep'),patch('os.close'),self.assertRaises(remote.ForeignUIDIdentityDrift):remote.foreign_uid_fd_modes(row,{})

    def test_helper_reported_identity_drift_is_not_retryable_visibility(self):
        import os,subprocess
        from unittest.mock import patch
        row={'pid':900,'ppid':1,'pgid':900,'sid':900,'birth':100,'state':'S'};uid=1001 if os.getuid()!=1001 else 1002;credentials={'uid':[uid]*4,'gid':[uid]*4}
        for message in (b'ValueError: kernel UID/birth mismatch\n',b'ValueError: kernel credential/birth changed\n'):
            with self.subTest(message=message),patch.object(remote,'linux_identity',return_value=row),patch.object(remote,'linux_proc_credentials',return_value=credentials),patch('shutil.which',return_value='/usr/bin/setpriv'),patch('subprocess.run',return_value=subprocess.CompletedProcess([],1,b'',message)),self.assertRaises(remote.ForeignUIDIdentityDrift):remote._foreign_uid_fd_modes_live(row,{'feed_root':'/private/tmp/owned-feed','feed_owner_uid':os.getuid()})

class ForeignProbeDiagnosticTests(unittest.TestCase):
    def test_target_kind_probe_returns_only_bounded_public_kernel_metadata(self):
        from unittest.mock import patch
        row={'pid':900,'ppid':899,'pgid':900,'sid':900,'birth':100,'state':'S'}
        with patch.object(remote,'linux_identity',return_value=row),patch.object(remote,'linux_proc_credentials',return_value={'uid':[1001]*4,'gid':[1001]*4}),patch.object(Path,'read_bytes',return_value=b'p4h-readonly-uid-fd-v1 SECRET_MUST_NOT_ESCAPE'):
            value=remote.foreign_probe_target_diagnostic(row)
            self.assertTrue(value['observer_helper_command']);self.assertEqual(value['identity'],row);self.assertNotIn('SECRET_MUST_NOT_ESCAPE',json.dumps(value))

    def test_target_disappearance_is_diagnostic_unknown_and_not_visibility_acceptance(self):
        from unittest.mock import patch
        with patch.object(remote,'linux_identity',side_effect=FileNotFoundError()):
            value=remote.foreign_probe_target_diagnostic({'pid':900,'birth':100});self.assertTrue(value['target_metadata_unavailable'])

class Attempt06ContinuationTests(unittest.TestCase):
    def test_attempt6_requires_actual_failed5_pins(self):
        q=AutonomousContinuationTests().fixture();ext=q['extension'];root=Path(ext['parent_queue']['path']).parents[1]
        q.update(attempts=6,receipt_path=str(root/'attempt06/queue.json'));ext['human_grant']['path']=str(root/'attempt06/autonomous-grant.json')
        ext['parent_queue']={'path':str(root/'attempt05/queue.json'),'sha256':'8f3e09105a06e6da73f6b2488dfb4ed27c9cb4ca50a4b94045c8a8e05a228721'}
        ext['previous_driver']={'path':str(root/'attempt05/driver-result.json'),'sha256':'89ea255c325b2f913f5df271cf1b46f946037077a012d4aabdbbfc00dcfcf09a'}
        ext['previous_closure']={'path':str(root/'attempt05/pilot-operation.driver-closure.json'),'sha256':'4b8a4c51ee6dca17a5aa42eef06bb5c51ac5ec3a47a384fb0c6fb59851a31446'}
        remote.Budget('pilot',**{k:q[k] for k in ('started_at','deadline','disk_cap','attempts','extension')})

class CommittedFeedSnapshotTests(unittest.TestCase):
    def test_pending_next_receipt_is_not_cached_before_head_commit(self):
        from unittest.mock import patch
        runs,payload=SafetyFeedTests().fixture();identity={'pid':40,'start_ticks':100,'executable_sha256':'a'*64,'start_receipt_sha256':'b'*64};jobs={'job_id_hash':'c'*64,'start_identity_hash':'d'*64}
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();journal=remote.FeedJournal(root/'feed',identity,jobs)
            try:
                first=journal.publish(payload,now=100,status='OPEN');journal.descriptor['initial_no_other_writer_inventory_sha256']='e'*64;desc=root/'descriptor.json';pin=remote.write_once(desc,journal.descriptor);reader=remote.SafetyFeedReader(desc,pin,runs)
                future={**first['value'],'seq':1,'previous_receipt_sha256':first['sha256'],'issued_at':101.,'expires_at':131.,'next_expected_by':146.};fp=remote.sha(remote.canonical(future));name='receipt-000001-'+fp+'.json';path=journal.root/name;path.write_bytes(remote.canonical(future))
                with patch.object(remote,'prove_publisher_runtime',return_value=True):
                    self.assertEqual(reader.observe(now=101)['head_seq'],0);self.assertNotIn(1,reader.cache)
                    path.chmod(0o400);head={'seq':1,'receipt_name':name,'receipt_sha256':fp};(journal.root/'new-head').write_bytes(remote.canonical(head));(journal.root/'new-head').replace(journal.root/'HEAD.json')
                    self.assertEqual(reader.observe(now=102)['head_seq'],1)
                    path.chmod(0o600)
                    with self.assertRaisesRegex(ValueError,'immutable receipt drift'):reader.observe(now=103)
            finally:journal.close()

    def test_head_snapshot_holds_one_inode_across_atomic_replace(self):
        import os
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder).resolve()/'HEAD.json';path.write_bytes(b'{"seq":0}');replacement=Path(folder).resolve()/'next';replacement.write_bytes(b'{"seq":1}');original=os.read;changed=[False]
            def read(fd,count):
                if not changed[0]:replacement.replace(path);changed[0]=True
                return original(fd,count)
            with patch('os.read',side_effect=read):self.assertEqual(remote.read_head_snapshot(path),{'seq':0})

    def test_head_inplace_mutation_is_rejected(self):
        import os
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder).resolve()/'HEAD.json';path.write_bytes(b'{"seq":0}');original=os.read;changed=[False]
            def read(fd,count):
                if not changed[0]:path.write_bytes(b'{"seq":222}');changed[0]=True
                return original(fd,count)
            with patch('os.read',side_effect=read),self.assertRaises(ValueError):remote.read_head_snapshot(path)

class OwnedExitedPinTests(unittest.TestCase):
    def test_kernel_gone_pid_returns_no_signal_handle(self):
        from unittest.mock import patch
        kernel=object.__new__(remote.LinuxKernel);row={'pid':900,'pgid':900,'sid':900,'birth':100}
        with patch('os.pidfd_open',side_effect=ProcessLookupError(),create=True),patch.object(kernel,'identity',return_value=None):self.assertIsNone(kernel.open_pidfd(row))

    def test_exited_held_pidfd_is_confirmed_and_closed(self):
        from unittest.mock import patch
        kernel=object.__new__(remote.LinuxKernel);row={'pid':900,'pgid':900,'sid':900,'birth':100}
        with patch('os.pidfd_open',return_value=44,create=True),patch.object(kernel,'identity',return_value=None),patch.object(remote,'foreign_observer_exited',return_value=True),patch('os.close') as close:self.assertIsNone(kernel.open_pidfd(row))
        close.assert_called_once_with(44)

    def test_gone_pin_does_not_hide_live_reuse_or_unconfirmed_exit(self):
        from unittest.mock import patch
        kernel=object.__new__(remote.LinuxKernel);row={'pid':900,'pgid':900,'sid':900,'birth':100}
        with patch('os.pidfd_open',side_effect=ProcessLookupError(),create=True),patch.object(kernel,'identity',return_value={**row,'birth':101}),self.assertRaises(ValueError):kernel.open_pidfd(row)
        with patch('os.pidfd_open',return_value=44,create=True),patch.object(kernel,'identity',return_value=None),patch.object(remote,'foreign_observer_exited',return_value=False),patch('os.close'),self.assertRaises(ValueError):kernel.open_pidfd(row)

class Attempt07ObservedQuiescenceTests(unittest.TestCase):
    def test_attempt7_needs_exact_late_quiescence_and_failed6_pins(self):
        q=AutonomousContinuationTests().fixture();ext=q['extension'];root=Path(ext['parent_queue']['path']).parents[1]
        q.update(attempts=7,receipt_path=str(root/'attempt07/queue.json'));ext['human_grant']['path']=str(root/'attempt07/autonomous-grant.json')
        ext['parent_queue']={'path':str(root/'attempt06/queue.json'),'sha256':'7a98d94d2b0fa60d9be300b823cabfaded653f6f484153a9832eb028fc76516e'}
        ext['previous_driver']={'path':str(root/'attempt06/driver-result.json'),'sha256':'6e20a00ace4b45bcceff42478e2d385bc2ad8be9465f6cd1d73df98dc6acb5d4'}
        ext['previous_closure']={'path':str(root/'attempt06/pilot-operation.driver-closure.json'),'sha256':'f49e40b3e619634801d87a94c037ead258e2ce789db3def22aec375c3f634cfd'}
        ext['late_quiescence']={'path':str(root/'attempt06/late-publisher-quiescence.json'),'sha256':'18cce7cf2cda93d76e13e3e5a17ff022335346a7f94d7a6264a696102d91eeb9'}
        remote.Budget('pilot',**{k:q[k] for k in ('started_at','deadline','disk_cap','attempts','extension')})
        for change in ('missing','pin','wrong_root'):
            bad=copy.deepcopy(q)
            if change=='missing':del bad['extension']['late_quiescence']
            elif change=='pin':bad['extension']['late_quiescence']['sha256']='f'*64
            else:bad['extension']['late_quiescence']['path']=str(root/'attempt05/late-publisher-quiescence.json')
            with self.subTest(change=change),self.assertRaises(ValueError):remote.Budget('pilot',**{k:bad[k] for k in ('started_at','deadline','disk_cap','attempts','extension')})

class GoneOwnedSignalTests(unittest.TestCase):
    def test_confirmed_gone_child_without_handle_receives_no_signal(self):
        from unittest.mock import Mock
        import types
        row={'pid':900,'pgid':900,'sid':900,'birth':100};lease=object.__new__(remote.KernelLease);lease.fds={};lease.tree=types.SimpleNamespace(signal_plan=lambda:[row]);lease.kernel=types.SimpleNamespace(open_pidfd=lambda r:None,identity=lambda pid:None,signal=Mock())
        self.assertEqual(lease._signal(15),0);lease.kernel.signal.assert_not_called()

class LateQuiescenceValidationTests(unittest.TestCase):
    def fixture(self):
        root=Path('/private/tmp/owned-pilot');q={'started_at':200.,'started_monotonic':1200.};ref=lambda name,pin:{'path':str(root/name),'sha256':pin}
        ext={'schema_id':'p4h-autonomous-pilot-continuation-v1','parent_queue':ref('attempt06/queue.json','7a98d94d2b0fa60d9be300b823cabfaded653f6f484153a9832eb028fc76516e'),'human_grant':ref('attempt07/autonomous-grant.json',remote.AUTONOMOUS_GRANT_SHA256),'previous_driver':ref('attempt06/driver-result.json','6e20a00ace4b45bcceff42478e2d385bc2ad8be9465f6cd1d73df98dc6acb5d4'),'previous_closure':ref('attempt06/pilot-operation.driver-closure.json','f49e40b3e619634801d87a94c037ead258e2ce789db3def22aec375c3f634cfd'),'late_quiescence':ref('attempt06/late-publisher-quiescence.json','18cce7cf2cda93d76e13e3e5a17ff022335346a7f94d7a6264a696102d91eeb9')};q['extension']=ext
        pins={'driver-result.json':ext['previous_driver']['sha256'],'pilot-operation.driver-closure.json':ext['previous_closure']['sha256'],'feed/publisher-closure.json':'149d03ffd2e66b831e1fe223c4ce5769125c7c51cc5be9df461e7052a4f29b1a','feed/descriptor.json':'34d3b0d29bf03a7b699b6503f86df0ec25df93891c419dd3cc2387765a5f5eac','driver-intent.json':'7dc0af92820f57126b9d020ec7d7f110b343ee086d1c13d7accacc34702b3bff'};refs={name:ref('attempt06/'+name,pin) for name,pin in pins.items()}
        publisher={'pid':902,'start_ticks':102};identity={'pid':902,'birth':102,'pgid':902,'sid':902,'ppid':899};pub={'state':'FAILED_UNSAFE','failure':'ProcessLookupError: [Errno 3] No such process','actual_worker_exit':None,'signals_sent':0,'owned_sessions':[902],'owned_identities':[identity]};failed={'publisher_driver_closure':pub,'queue':{'host_boot_id':'boot'},'finished_at':100.};closed={'owned_sessions':[900,901]}
        cert={'schema_id':'p4h-observed-late-publisher-quiescence-v1','previous_records':refs,'original_status_preserved':'failed+spent','original_publisher_closure_preserved':'FAILED_UNSAFE','original_parent_wait':'UNKNOWN_NOT_RECOVERABLE','actual_cooperative_closed':True,'fresh_owned_processes':[],'fresh_gpu_pids':[],'original_bytes_and_spent_preserved':True,'new_science_submit':False,'fresh_owned_sessions_absent':[899,900,901,902],'host_boot_id':'boot','observed_at':150.,'observed_monotonic':1150.}
        receipt={'status':'CLOSED','reason':'cooperative_stop','lock_fd_closed':True,'descriptor_sha256':pins['feed/descriptor.json'],'publisher_process':publisher}
        values={ext['late_quiescence']['path']:cert,refs['feed/descriptor.json']['path']:{'publisher_process':publisher},refs['feed/publisher-closure.json']['path']:receipt,refs['driver-intent.json']['path']:{'identity':{'pid':899,'sid':899}}}
        return q,failed,closed,values

    def test_fixed_late_evidence_preserves_unknown_parent_wait(self):
        from unittest.mock import patch
        q,failed,closed,values=self.fixture()
        with patch.object(remote,'_read_json',side_effect=lambda path,**kw:values[str(path)]),patch.object(remote,'linux_processes',return_value=[]):self.assertTrue(remote.verify_late_publisher_quiescence(q,failed,closed))
        for field,bad in (('original_parent_wait',0),('actual_cooperative_closed',False),('fresh_gpu_pids',[999]),('host_boot_id','different')):
            altered=copy.deepcopy(values);altered[q['extension']['late_quiescence']['path']][field]=bad
            with self.subTest(field=field),patch.object(remote,'_read_json',side_effect=lambda path,**kw:altered[str(path)]),patch.object(remote,'linux_processes',return_value=[]),self.assertRaises(ValueError):remote.verify_late_publisher_quiescence(q,failed,closed)
        with patch.object(remote,'_read_json',side_effect=lambda path,**kw:values[str(path)]),patch.object(remote,'linux_processes',return_value=[{'sid':902}]),self.assertRaises(ValueError):remote.verify_late_publisher_quiescence(q,failed,closed)

class CompleteArtifactDigestConventionTests(unittest.TestCase):
    def test_artifact_set_digest_matches_frozen_plan_newline_convention(self):
        artifacts=[{'role':'pilot-artifact-index','path':'/private/tmp/owned/index.json','sha256':'a'*64,'size':10}]
        expected=hashlib.sha256(remote.plan.canonical_bytes(artifacts)).hexdigest()
        self.assertEqual(remote.complete_artifact_digest(artifacts),expected)
        self.assertNotEqual(expected,remote.sha(remote.canonical(artifacts)))


@unittest.skipIf(remote is None, 'real adapter missing')
class ProcExitRaceTests(unittest.TestCase):
    """P4H-05 S02: /proc entries can vanish between listdir and read (ESRCH)."""
    STAT='{pid} (python) S 1 {pid} {pid} 0 -1 0 0 0 0 0 0 0 0 0 20 0 1 0 {birth} 0 0'

    def fake_proc(self,errors):
        from pathlib import Path as Real
        test=self
        class Entry:
            def __init__(self,name):self.name=name
            def __truediv__(self,part):
                name=self.name
                class Leaf:
                    def read_bytes(self):
                        if name in errors:raise errors[name]
                        return test.STAT.format(pid=name,birth=100+int(name)).encode()
                return Leaf()
        class Root:
            def iterdir(self):return [Entry(n) for n in ('10','11','self')]
        return lambda value='',*a:Root() if value=='/proc' else Real(value,*a)

    def test_esrch_vanished_process_is_skipped(self):
        from unittest.mock import patch
        with patch.object(remote,'Path',self.fake_proc({'10':ProcessLookupError(3,'No such process')})):
            rows=remote.linux_processes()
        self.assertEqual([r['pid'] for r in rows],[11])

    def test_enoent_vanished_process_still_skipped(self):
        from unittest.mock import patch
        with patch.object(remote,'Path',self.fake_proc({'10':FileNotFoundError(2,'gone')})):
            self.assertEqual([r['pid'] for r in remote.linux_processes()],[11])

    def test_other_read_errors_fail_closed(self):
        from unittest.mock import patch
        for error in (PermissionError(13,'denied'),OSError(5,'EIO'),ValueError('stat schema')):
            with self.subTest(error=type(error).__name__),patch.object(remote,'Path',self.fake_proc({'10':error})),self.assertRaises(type(error)):
                remote.linux_processes()

    def writer_fixture(self,folder):
        from pathlib import Path as Real
        base=Real(folder);proc=base/'proc';feed=base/'feed';feed.mkdir();(feed/'lock').write_bytes(b'')
        for pid in ('10','11','12'):
            (proc/pid/'fd').mkdir(parents=True);(proc/pid/'fdinfo').mkdir()
            os.symlink(str(feed/'lock'),proc/pid/'fd'/'3');(proc/pid/'fdinfo'/'3').write_text('pos:\t0\nflags:\t0100001\n')
        descriptor={'feed_root':str(feed),'feed_root_identity':'fixture','publisher_process':{'pid':10,'start_ticks':110,'executable_sha256':'e'*64}}
        rows=[{'pid':int(p),'ppid':1,'pgid':int(p),'sid':int(p),'birth':100+int(p),'state':'S'} for p in ('10','11','12')]
        mapped=lambda value='',*a:proc if value=='/proc' else Real(value,*a)
        identity=lambda pid:{'pid':pid,'start_ticks':100+pid,'executable_sha256':'e'*64 if pid==10 else 'f'*64}
        return descriptor,rows,mapped,proc,identity

    def test_writer_scan_skips_esrch_vanished_fd_set_but_keeps_publisher(self):
        import pathlib
        from unittest.mock import patch
        real_iterdir=pathlib.Path.iterdir;real_readlink=os.readlink
        with tempfile.TemporaryDirectory() as folder:
            descriptor,rows,mapped,proc,identity=self.writer_fixture(os.path.realpath(folder))
            def iterdir(self):
                if self==proc/'11'/'fd':raise ProcessLookupError(3,'No such process')
                return real_iterdir(self)
            def readlink(path,*a):
                if str(path)==str(proc/'12'/'fd'/'3'):raise ProcessLookupError(3,'No such process')
                return real_readlink(path,*a)
            with patch.object(remote,'Path',mapped),patch.object(pathlib.Path,'iterdir',iterdir),patch.object(remote.os,'readlink',readlink),patch.object(remote,'linux_processes',return_value=rows),patch.object(remote,'linux_publisher_identity',side_effect=identity):
                value=remote.linux_writer_inventory(descriptor)
        self.assertEqual(value['expected_writer_count'],1);self.assertEqual(value['unknown_writer_count'],0)

    def test_live_unknown_writer_still_stops_after_race_fix(self):
        import pathlib
        from unittest.mock import patch
        real_iterdir=pathlib.Path.iterdir
        with tempfile.TemporaryDirectory() as folder:
            descriptor,rows,mapped,proc,identity=self.writer_fixture(os.path.realpath(folder))
            def iterdir(self):
                if self==proc/'11'/'fd':raise ProcessLookupError(3,'No such process')
                return real_iterdir(self)
            with patch.object(remote,'Path',mapped),patch.object(pathlib.Path,'iterdir',iterdir),patch.object(remote,'linux_processes',return_value=rows),patch.object(remote,'linux_publisher_identity',side_effect=identity),self.assertRaisesRegex(ValueError,'writer inventory'):
                remote.linux_writer_inventory(descriptor)

    def test_writer_scan_other_errors_fail_closed(self):
        import pathlib
        from unittest.mock import patch
        real_iterdir=pathlib.Path.iterdir
        with tempfile.TemporaryDirectory() as folder:
            descriptor,rows,mapped,proc,identity=self.writer_fixture(os.path.realpath(folder))
            def iterdir(self):
                if self==proc/'11'/'fd':raise OSError(5,'EIO')
                return real_iterdir(self)
            with patch.object(remote,'Path',mapped),patch.object(pathlib.Path,'iterdir',iterdir),patch.object(remote,'linux_processes',return_value=rows[:2]),patch.object(remote,'linux_publisher_identity',side_effect=identity),self.assertRaises(OSError):
                remote.linux_writer_inventory(descriptor)


@unittest.skipIf(remote is None, 'real adapter missing')
class S02AcceptanceRecoveryTests(unittest.TestCase):
    """Read-only acceptance of the pinned S02 model; original failed terminal is never rewritten."""
    R='src/vkgr/retrieval_sasrec_remote.py'

    def fixture(self,folder):
        base=Path(folder);output=base/'out';control=base/'ctl';(output/'model').mkdir(parents=True);(output/'fresh-best').mkdir();control.mkdir()
        refs=copy.deepcopy(remote.S02_RECOVERY_REFS)
        slot=next(s for s in remote.plan.FORMAL_SLOTS if s['slot_id']=='P4H-S02')
        queue={'kind':'formal','attempts':1,'fixture':True}
        common=[{'role':remote.plan.COMMON_ROLES[0],'source':'common','path':'/c','sha256':'c'*64,'size':1}]
        inputs=[{'role':'catalog','source':'input_hashes','path':'/i/catalog','sha256':'1'*64,'size':2}]
        pins={self.R:remote.S02_ORIGINAL_CONSUMER_SHA256,'src/vkgr/retrieval_sasrec.py':'d'*64}
        op={'stage':'formal-train','operation_id':'p4h-main05-formal01-P4H-S02-formal-train','slot':copy.deepcopy(slot),'queue':copy.deepcopy(queue),'predecessor':None,'source_pins':pins,'roles':common+inputs,'output_root':str(output),'control_root':str(control)}
        value={'stage':'formal-train','slot':copy.deepcopy(slot),'queue':copy.deepcopy(queue),'source_pins':{**pins,self.R:'e'*64},'roles':copy.deepcopy(common)+copy.deepcopy(inputs)}
        binding={'input_sha256':{'catalog':'1'*64},'source_sha256':remote.sha(remote.canonical(pins)),'test_gate':'closed'}
        evaluations=[{'epoch':e,'metrics':{'all':{'recall@50':m,'ndcg@50':m}},'prediction_sha256':str(e)*64} for e,m in ((1,.5),(2,.4),(3,.3),(4,.2))]
        selection={'epoch':4,'best_epoch':1,'best_metric':[.5,.5],'bad_epochs':3}
        public={'status':'completed+unverified','stop_reason':'patience','config':copy.deepcopy(slot),'steps':2,'step_records':[{},{}],'evaluations':evaluations,'selection':selection}
        index={'schema_id':'p4h-real-model-artifacts-v1','binding':binding,'slot':copy.deepcopy(slot),'test_gate':'closed','state':{'path':str(output/'model/state.pt'),'sha256':'f'*64,'size':10},'public':public,'science_events':[]}
        refs['original_operation']['path']=str(control/'operation.json');refs['original_model_artifacts']['path']=str(output/'model/model-artifacts.json')
        (output/'model/model-artifacts.json').write_bytes(b'x')
        reg={'manifest_sha256':refs['original_operation']['sha256'],'operation_id':op['operation_id'],'output_root':str(output),'control_root':str(control),'queue':copy.deepcopy(queue),'attempt':{'slot_id':'P4H-S02','number':1,'reason':'initial','previous_terminal':None},'issued_at':1.}
        terminal={'status':'failed+spent','failure':'ValueError: actual worker failed','manifest_sha256':refs['original_operation']['sha256'],'closure_sha256':refs['original_closure']['sha256'],'worker_released':True,'optimizer_steps':None,'queue':copy.deepcopy(queue),'attempt':1,'operation_id':op['operation_id']}
        closure={'state':'CLOSED','actual_worker_exit':1,'signals_sent':0,'manifest_sha256':refs['original_operation']['sha256'],'failure':None,'owned_sessions':[900]}
        driver={'status':'failed+spent','stages':[],'test_read':False,'additional_slots_submitted':False,'cleanup_failures':[],'queue':copy.deepcopy(queue),'publisher_driver_closure':{'state':'CLOSED','actual_worker_exit':0,'owned_sessions':[901]}}
        stderr=b'  File "x", line 1429, in worker_entry\n    fresh=evaluate_formal(model,\n  File "x", line 424, in linux_processes\nProcessLookupError: [Errno 3] No such process\n'
        values={'original_registration':reg,'original_operation':op,'original_terminal':terminal,'original_closure':closure,'original_driver_result':driver,'original_model_artifacts':index}
        auth={'acceptance_recovery':{'schema_id':remote.S02_RECOVERY_SCHEMA,**refs}}
        return {'refs':refs,'values':values,'value':value,'auth':auth,'stderr':stderr,'output':output}

    def call(self,f,processes=()):
        from unittest.mock import patch
        by_path={f['refs'][k]['path']:v for k,v in f['values'].items()}
        def read_json(path,cap=65536,pin=None):
            key=next(k for k,r in f['refs'].items() if r['path']==str(path))
            if pin!=f['refs'][key]['sha256']:raise ValueError('pin')
            return copy.deepcopy(by_path[str(path)])
        def read_bound(path,pin,cap):
            if str(path)==f['refs']['original_worker_stderr']['path']:return f['stderr']
            return b'{}'
        refs=f['refs']
        with patch.object(remote,'S02_RECOVERY_REFS',refs),patch.object(remote,'_read_json',side_effect=read_json),patch.object(remote,'read_bound',side_effect=read_bound),patch.object(remote,'verify_source_tree') as source,patch.object(remote,'linux_processes',return_value=list(processes)):
            result=remote.acceptance_recovery_original(f['value'],f['auth'])
            source.assert_called_once()
            return result

    def test_pinned_complete_original_is_recoverable(self):
        with tempfile.TemporaryDirectory() as folder:
            f=self.fixture(folder);result=self.call(f)
            self.assertEqual(result['model_root'],f['output']/'model')
            self.assertEqual(result['binding'],f['values']['original_model_artifacts']['binding'])
            self.assertEqual(result['index_record']['path'],f['refs']['original_model_artifacts']['path'])

    def test_unpinned_or_altered_authority_rejected_before_read(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as folder:
            f=self.fixture(folder)
            for change in ('schema','path','missing','slot','stage'):
                auth=copy.deepcopy(f['auth']);value=copy.deepcopy(f['value'])
                if change=='schema':auth['acceptance_recovery']['schema_id']='other'
                elif change=='path':auth['acceptance_recovery']['original_terminal']['path']='/other'
                elif change=='missing':del auth['acceptance_recovery']['original_closure']
                elif change=='slot':value['slot']['slot_id']='P4H-S06'
                else:value['stage']='pilot'
                with self.subTest(change=change),patch.object(remote,'S02_RECOVERY_REFS',f['refs']),patch.object(remote,'_read_json') as read,self.assertRaises(ValueError):
                    remote.acceptance_recovery_original(value,auth)
                read.assert_not_called()

    def test_original_runtime_cannot_recover_itself(self):
        with tempfile.TemporaryDirectory() as folder:
            f=self.fixture(folder);f['value']['source_pins'][self.R]=remote.S02_ORIGINAL_CONSUMER_SHA256
            with self.assertRaises(ValueError):self.call(f)

    def test_any_original_or_binding_drift_fails_closed(self):
        cases={
            'terminal_status':lambda v:v['original_terminal'].update(status='completed+unverified'),
            'terminal_steps':lambda v:v['original_terminal'].update(optimizer_steps=0),
            'closure_signal':lambda v:v['original_closure'].update(signals_sent=1),
            'closure_exit':lambda v:v['original_closure'].update(actual_worker_exit=0),
            'driver_stage':lambda v:v['original_driver_result'].update(stages=[{}]),
            'driver_test':lambda v:v['original_driver_result'].update(test_read=True),
            'publisher':lambda v:v['original_driver_result']['publisher_driver_closure'].update(state='FAILED_UNSAFE'),
            'registration':lambda v:v['original_registration']['attempt'].update(number=2),
            'queue':lambda v:v['original_operation']['queue'].update(attempts=2),
            'science_source':lambda v:v['original_operation']['source_pins'].update({'src/vkgr/retrieval_sasrec.py':'0'*64}),
            'binding':lambda v:v['original_model_artifacts']['binding'].update(test_gate='open'),
            'incomplete':lambda v:v['original_model_artifacts']['public'].update(status='running'),
            'stop_reason':lambda v:v['original_model_artifacts']['public'].update(stop_reason='preempted'),
            'selection':lambda v:v['original_model_artifacts']['public']['selection'].update(best_epoch=2),
            'steps':lambda v:v['original_model_artifacts']['public'].update(steps=3),
            'state_path':lambda v:v['original_model_artifacts']['state'].update(path='/elsewhere/state.pt'),
        }
        for name,mutate in cases.items():
            with self.subTest(case=name),tempfile.TemporaryDirectory() as folder:
                f=self.fixture(folder);mutate(f['values'])
                with self.assertRaises(ValueError):self.call(f)

    def test_input_drift_or_live_session_or_partial_export_fails_closed(self):
        for case in ('input','live','sealed','fresh','stderr'):
            with self.subTest(case=case),tempfile.TemporaryDirectory() as folder:
                f=self.fixture(folder);processes=()
                if case=='input':f['value']['roles'][1]['sha256']='2'*64
                elif case=='live':processes=({'sid':900},)
                elif case=='sealed':(f['output']/'sealed.json').write_text('{}')
                elif case=='fresh':(f['output']/'fresh-best'/'result.json').write_text('{}')
                else:f['stderr']=b'ValueError: other failure\n'
                with self.assertRaises(ValueError):self.call(f,processes)

    def test_recovery_registration_is_separate_and_unique(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);value={'stage':'formal-train','queue':{'receipt_path':str(root/'queue.json')},'operation_id':'p4h-r','output_root':'/o','control_root':'/c','slot':{'slot_id':'P4H-S02'}}
            auth={'attempt':{'slot_id':'P4H-S02','number':1,'reason':'initial','previous_terminal':None},'acceptance_recovery':{'fixture':1}}
            target=root/'attempt-P4H-S02-1-acceptance-recovery-1.json'
            with patch.object(remote,'verify_queue_authority'),patch.object(remote,'acceptance_recovery_original') as original,patch.object(remote,'S02_RECOVERY_REGISTRATION',str(target)):
                record=remote.reserve_queue_attempt(value,auth,'a'*64)
                original.assert_called_once_with(value,auth)
                self.assertTrue(target.exists());self.assertFalse((root/'attempt-P4H-S02-1.json').exists())
                self.assertEqual(record['acceptance_recovery'],{'fixture':1})
                with self.assertRaisesRegex(ValueError,'duplicate'):remote.reserve_queue_attempt(value,auth,'b'*64)


@unittest.skipIf(remote is None, 'real adapter missing')
class ProcExitRaceFollowupTests(unittest.TestCase):
    """Review A1/A2: the same vanished-process ESRCH on publisher scan and kernel identity."""
    def test_observe_g_cmdline_esrch_is_vanished_other_errors_stop(self):
        from unittest.mock import patch
        from pathlib import Path as Real
        rows=[{'pid':20,'ppid':1,'pgid':20,'sid':20,'birth':120,'state':'S'},{'pid':21,'ppid':1,'pgid':21,'sid':21,'birth':121,'state':'S'}]
        def mapped(error):
            class Leaf:
                def __init__(self,pid):self.pid=pid
                def __truediv__(self,part):return self
                def read_bytes(self):
                    if self.pid=='20':raise error
                    return b'bash\x00'
            class Proc:
                def __truediv__(self,pid):return Leaf(pid)
            return lambda value='',*a:Proc() if value=='/proc' else Real(value,*a)
        with patch.object(remote,'linux_processes',return_value=rows),patch.object(remote,'linux_resources',return_value=({'ram':.5,'vram':.5,'disk':.5},[])),patch.object(remote,'Path',mapped(ProcessLookupError(3,'No such process'))):
            self.assertEqual(remote.observe_g([],allowed_p=[])['p_writer_count'],0)
        for error in (PermissionError(13,'denied'),OSError(5,'EIO')):
            with self.subTest(error=type(error).__name__),patch.object(remote,'linux_processes',return_value=rows),patch.object(remote,'linux_resources',return_value=({'ram':.5,'vram':.5,'disk':.5},[])),patch.object(remote,'Path',mapped(error)),self.assertRaises(type(error)):
                remote.observe_g([],allowed_p=[])

    def test_kernel_identity_maps_only_vanished_to_none(self):
        from unittest.mock import patch
        kernel=object.__new__(remote.LinuxKernel)
        for error in (FileNotFoundError(2,'gone'),ProcessLookupError(3,'No such process')):
            with self.subTest(error=type(error).__name__),patch.object(remote,'linux_identity',side_effect=error):
                self.assertIsNone(kernel.identity(7))
        for error in (PermissionError(13,'denied'),ValueError('proc stat schema')):
            with self.subTest(error=type(error).__name__),patch.object(remote,'linux_identity',side_effect=error),self.assertRaises(type(error)):
                kernel.identity(7)


@unittest.skipIf(remote is None, 'real adapter missing')
class S02RecoveryBranchTests(unittest.TestCase):
    """Review T1: recovery branches load/seal/origin; never build an optimizer."""
    def state(self):
        import torch
        from vkgr import retrieval_sasrec_experiment as science
        best={'w':torch.ones(2)};last={'w':torch.zeros(2)}
        public={'status':'completed+unverified','checkpoint_roles':{'best':science._tensor_digest(best),'last':science._tensor_digest(last)},'fixture':1}
        return {'public':public,'private':{'best_model_state':best,'model_state':last}}

    def recovery(self,state,binding=None,slot=None):
        index={'state':{'path':'/o/model/state.pt','sha256':'a'*64,'size':10},'slot':slot or {'slot_id':'P4H-S02'},'public':copy.deepcopy(state['public'])}
        return {'index':index,'binding':binding or {'source_sha256':'b'*64}}

    def blob(self,value):
        import io,torch
        buffer=io.BytesIO();torch.save(value,buffer);return buffer.getvalue()

    def test_load_recovered_state_checks_binding_public_and_tensor_digests(self):
        from unittest.mock import patch
        state=self.state();recovery=self.recovery(state);calls=[]
        good={'binding':recovery['binding'],'slot':recovery['index']['slot'],'state':state}
        with patch.object(remote,'read_bound',return_value=self.blob(good)):
            loaded=remote.load_recovered_state(recovery,lambda:calls.append(1))
        self.assertEqual(loaded['public'],state['public']);self.assertGreaterEqual(len(calls),2)
        def bad(kind):
            value=copy.deepcopy(good)
            if kind=='binding':value['binding']={'source_sha256':'c'*64}
            elif kind=='slot':value['slot']={'slot_id':'P4H-S06'}
            elif kind=='public':value['state']['public']['fixture']=2
            elif kind=='best':
                import torch;value['state']['private']['best_model_state']={'w':torch.full((2,),3.)}
            else:value['extra']=1
            return value
        for kind in ('binding','slot','public','best','extra'):
            with self.subTest(kind=kind),patch.object(remote,'read_bound',return_value=self.blob(bad(kind))),self.assertRaises(ValueError):
                remote.load_recovered_state(recovery,lambda:None)

    def test_worker_recovery_branch_never_builds_optimizer_or_trains(self):
        import torch
        from unittest.mock import patch,MagicMock
        from vkgr import retrieval_sasrec,training
        state=self.state();state['public']['selection']={'best_epoch':1};expected={'epoch':1,'metrics':{},'prediction_sha256':'d'*64};state['public']['evaluations']=[expected]
        slot={'slot_id':'P4H-S02','seed':42,'history_source':'like','learning_rate':.001}
        value={'stage':'formal-train','slot':slot,'control_root':'/c','output_root':'/o','queue':{'q':1},'roles':[],'source_pins':{'p':'e'*64},'g_feed':{'path':'/f','sha256':'f'*64},'limits':{},'authorization':{'path':'/auth','sha256':'1'*64}}
        recovery={'binding':{'input_sha256':{},'source_sha256':'2'*64,'test_gate':'closed'}}
        contents={'prepared-train-rows':[],'formal-history-view':[],'catalog':[],'idmap':{},'validation-query-target-free':[{'query_id':'q'}],'validation-target':[{'query_id':'q','target':'x','oov':True}]}
        data={'catalog':['x'],'supported_indices':[1],'rows_by_arm':{'like':[]},'idmap':{}}
        written={}
        def read_json(path,cap=65536,pin=None):return {'acceptance_recovery':{'fixture':1}} if str(path)=='/auth' else {'g_receipt_sha256':'3'*64}
        model=MagicMock();model.to.return_value=model
        patches=[patch.object(remote,n) for n in ('configure_real_torch','validate_remote_manifest','verify_clock_extension','verify_source_tree','verify_common_bindings','validate_admission','RuntimeGate','MonotonicBudget','SafetyFeedReader','queue_charge_root','verify_queue_authority','verify_input_origin','finite_batches')]
        patches+=[patch.object(remote,'read_roles',return_value=(contents,[])),patch.object(remote,'_read_json',side_effect=read_json),patch.object(remote,'acceptance_recovery_original',return_value=recovery),patch.object(remote,'compile_training_roles',return_value=data),patch.object(remote,'load_recovered_state',return_value=state) ,patch.object(remote,'evaluate_formal',return_value=expected),patch.object(remote,'write_once',side_effect=lambda p,v:written.setdefault(str(p),v)),patch.object(remote,'train_model',side_effect=AssertionError('train_model called')),patch.object(torch.optim,'AdamW',side_effect=AssertionError('optimizer built')),patch.object(retrieval_sasrec,'SASRecIDTwoTower',return_value=model),patch.object(training,'seed_all'),patch.object(remote.Path,'mkdir')]
        for p in patches:p.start()
        try:result=remote.worker_entry(value,'4'*64,[])
        finally:
            for p in reversed(patches):p.stop()
        self.assertEqual(result,state['public'])
        self.assertIn('/o/fresh-best/result.json',written);self.assertEqual(written['/o/fresh-best/result.json'],expected)
        self.assertIn('/o/worker-result.json',written)
        model.load_state_dict.assert_called_once_with(state['private']['best_model_state'])

    def test_seal_recovery_binds_original_index_and_marks_zero_recovery_steps(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as folder:
            base=Path(os.path.realpath(folder));output=base/'out';(output/'fresh-best').mkdir(parents=True);original=base/'orig'/'model';original.mkdir(parents=True)
            evaluations=[{'epoch':1,'metrics':{'all':{'recall@50':.5,'ndcg@50':.5}},'prediction_sha256':'5'*64}]
            public={'selection':{'epoch':1,'best_epoch':1,'best_metric':[.5,.5],'bad_epochs':0},'checkpoint_roles':{'best':'6'*64,'last':'7'*64},'steps':2,'evaluations':evaluations,'step_records':[{},{}]}
            index={'public':public,'science_events':[{'kind':'optimizer_step','value':{}}]}
            (original/'model-artifacts.json').write_text(json.dumps(index));(output/'fresh-best'/'result.json').write_text(json.dumps(evaluations[0]))
            record={'role':'model-artifacts','path':str(original/'model-artifacts.json'),'sha256':hashlib.sha256((original/'model-artifacts.json').read_bytes()).hexdigest(),'size':(original/'model-artifacts.json').stat().st_size}
            recovery={'index_record':record,'index':index}
            value={'stage':'formal-train','operation_id':'p4h-recovery','output_root':str(output),'roles':[{'role':'catalog','sha256':'8'*64}],'queue':{'attempts':1},'authorization':{'path':'/auth','sha256':'9'*64},'control_root':'/c'}
            start={'opened_roles':[],'output_identity':{'device':1,'inode':2},'control_identity':{'device':1,'inode':3}}
            closure={'state':'CLOSED','actual_worker_exit':0,'owned_sessions':[1],'owned_identities':[],'signals_sent':0,'elapsed_seconds':1.}
            real=remote._read_json
            def read_json(path,cap=65536,pin=None):return {'acceptance_recovery':{'fixture':1}} if str(path)=='/auth' else real(path,cap=cap,pin=pin)
            with patch.object(remote,'linux_processes',return_value=[]),patch.object(remote,'_read_json',side_effect=read_json),patch.object(remote,'acceptance_recovery_original',return_value=recovery):
                terminal,sealed=remote.seal_completed_operation(value,'a'*64,start,'b'*64,{'p':1},closure,[{'queue_bytes':1}],1.,'c'*64)
            self.assertIn(record,terminal['private_artifacts'])
            self.assertFalse((output/'model').exists())
            ledger=json.loads((output/'formal-stage-event-ledger.json').read_text())
            event=next(e for e in ledger['events'] if e['kind']=='acceptance_recovery')
            self.assertEqual(event['optimizer_steps_during_recovery'],0);self.assertEqual(event['original_terminal_status_preserved'],'failed+spent')
            checkpoint=json.loads((output/'formal-checkpoint-index.json').read_text())
            self.assertEqual(checkpoint['best']['path'],record['path']);self.assertEqual(sealed['state'],'CLOSED')
            tampered={'index_record':{**record,'sha256':'0'*64},'index':index}
            with patch.object(remote,'linux_processes',return_value=[]),patch.object(remote,'_read_json',side_effect=read_json),patch.object(remote,'acceptance_recovery_original',return_value=tampered),self.assertRaises(ValueError):
                remote.seal_completed_operation({**value,'output_root':str(base/'out2')},'a'*64,start,'b'*64,{'p':1},closure,[{'queue_bytes':1}],1.,'c'*64)

    def test_origin_consumer_is_original_only_under_recovery(self):
        from unittest.mock import patch
        from vkgr.retrieval_ablation_run import PREPARED_SHA
        original=remote.S02_ORIGINAL_CONSUMER_SHA256
        producer={'schema_id':'p4h-real-input-producer-v1','prepared_manifest_sha256':PREPARED_SHA,'consumer_source_sha256':original,'test_gate':'closed','prepared_manifest_path':'/p','raw_records':[],'raw_root':'/r','artifacts':{'catalog':{'path':'/i/catalog','sha256':'1'*64,'size':2}}}
        value={'source_pins':{'src/vkgr/retrieval_sasrec_remote.py':'e'*64},'roles':[{'role':'catalog','source':'input_hashes','sha256':'1'*64}],'slot':{'history_source':'like'}}
        auth={'origin':{'path':'/producer','sha256':'2'*64},'input_hashes':{'catalog':'1'*64},'source_sha256':remote.sha(remote.canonical(value['source_pins'])),'acceptance_recovery':{'fixture':1}}
        def read_json(path,cap=65536,pin=None):return copy.deepcopy(producer) if str(path)=='/producer' else {'binding':{'input_records':[]}}
        with patch.object(remote,'_read_json',side_effect=read_json),patch.object(remote,'read_bound'),patch.object(remote,'acceptance_recovery_original',return_value={'operation':{'source_pins':{'src/vkgr/retrieval_sasrec_remote.py':original}}}) as rec:
            self.assertEqual(remote.verify_input_origin(value,auth),producer);rec.assert_called_once_with(value,auth)
        plain={k:v for k,v in auth.items() if k!='acceptance_recovery'}
        with patch.object(remote,'_read_json',side_effect=read_json),patch.object(remote,'read_bound'),self.assertRaisesRegex(ValueError,'consumer'):
            remote.verify_input_origin(value,plain)


# INFRA-01 sub-task B: job-scoped P gate (design §3, review v3 M1/M2/N10).
def job_payload(observed_at=100.,**fractions):
    values={'ram':.5,'vram':.5,'disk':.5,'cpu':.05};values.update(fractions)
    return {'observed_at':observed_at,'runs':[],'free_fractions':values,'gpu_pids':[],'allowed_gpu_pids':[],'protected_pgids':[],'p_writer_count':0,'gate_mode':'job-scoped'}


@unittest.skipIf(remote is None,'real adapter missing')
class JobScopedManifestTests(unittest.TestCase):
    def test_gate_field_selects_mode_and_rejects_bad_envelope(self):
        value=real_manifest_fixture(Path('/private/tmp/real-operation'))
        self.assertEqual(remote.validate_remote_manifest(value)['gate_mode'],'legacy')
        self.assertEqual(remote.validate_remote_manifest({**value,'gate':{'mode':'legacy'}})['gate_mode'],'legacy')
        self.assertEqual(remote.validate_remote_manifest({**value,'gate':copy.deepcopy(remote.NEW_RUN_GATE)})['gate_mode'],'job-scoped')
        self.assertEqual(remote.NEW_RUN_GATE,{'mode':'job-scoped'})
        for bad in ({'mode':'shared'},{'mode':'job-scoped','extra':1},'job-scoped',None,{}):
            with self.subTest(bad=bad),self.assertRaisesRegex(ValueError,'gate mode envelope'):remote.validate_remote_manifest({**value,'gate':bad})


@unittest.skipIf(remote is None,'real adapter missing')
class JobScopedPayloadTests(unittest.TestCase):
    def test_job_scoped_payload_keeps_only_resource_stop(self):
        self.assertTrue(remote.validate_safety_payload(job_payload(),[],now=101,max_age=30,mode='job-scoped'))
        self.assertTrue(remote.validate_safety_payload(job_payload(cpu=None),[],now=101,max_age=30,mode='job-scoped'))
        with self.assertRaisesRegex(ValueError,'shared resource STOP'):remote.validate_safety_payload(job_payload(ram=.19),[],now=101,max_age=30,mode='job-scoped')
        mismatches={'gpu':{**job_payload(),'gpu_pids':[123]},'writers':{**job_payload(),'p_writer_count':3},'allowed':{**job_payload(),'allowed_gpu_pids':[5]},'protected':{**job_payload(),'protected_pgids':[5]},'runs':{**job_payload(),'runs':[{'run_id':'x'}]},'mode':{**job_payload(),'gate_mode':'legacy'},'cpu_range':job_payload(cpu=1.5),'cpu_bool':job_payload(cpu=True)}
        missing_cpu=job_payload();del missing_cpu['free_fractions']['cpu'];mismatches['missing_cpu']=missing_cpu
        missing_mode=job_payload();del missing_mode['gate_mode'];mismatches['missing_mode']=missing_mode
        for name,payload in mismatches.items():
            with self.subTest(name=name),self.assertRaisesRegex(ValueError,'gate mode/payload mismatch'):remote.validate_safety_payload(payload,[],now=101,max_age=30,mode='job-scoped')
        with self.assertRaisesRegex(ValueError,'gate mode/payload mismatch'):remote.validate_safety_payload(job_payload(),[{'run_id':'core-02','start_sha256':'a'*64}],now=101,max_age=30,mode='job-scoped')
        with self.assertRaises(ValueError):remote.validate_safety_payload(job_payload(observed_at=60.),[],now=101,max_age=30,mode='job-scoped')
        runs,legacy=SafetyFeedTests().fixture()
        with self.assertRaises(ValueError):remote.validate_safety_payload(legacy,[],now=101,max_age=30,mode='job-scoped')
        with self.assertRaises(ValueError):remote.validate_safety_payload(job_payload(),runs,now=101,max_age=30)
        self.assertTrue(remote.validate_safety_payload(legacy,runs,now=101,max_age=30,mode='legacy'))
        self.assertFalse(remote.JOB_SCOPED_CPU_ENFORCED)


@unittest.skipIf(remote is None,'real adapter missing')
class ObserveJobTests(unittest.TestCase):
    def test_observe_job_reads_no_g_files_and_never_lists_compute_apps(self):
        import subprocess
        from unittest.mock import patch
        calls=[]
        def run(argv,*a,**k):
            calls.append(list(argv));return subprocess.CompletedProcess(argv,0,'1000, 500\n','')
        with tempfile.TemporaryDirectory() as folder:
            root=Path(os.path.realpath(folder));stat_file=root/'stat';stat_file.write_text('cpu  100 0 100 800 0 0 0 0 0 0\n')
            def advance(_):stat_file.write_text('cpu  150 0 150 900 0 0 0 0 0 0\n')
            sampler=remote.CpuSampler(path=stat_file,sleep=advance)
            with patch('subprocess.run',side_effect=run),patch.object(remote,'current_ram_fraction',return_value=.5),patch.object(remote,'observe_g',side_effect=AssertionError('G observation')):
                payload=remote.observe_job(root,cpu=sampler)
            self.assertTrue(calls);self.assertFalse(any('--query-compute-apps' in a for argv in calls for a in argv))
            self.assertTrue(any(a.startswith('--query-gpu=') for argv in calls for a in argv))
            self.assertEqual(payload['gate_mode'],'job-scoped');self.assertAlmostEqual(payload['free_fractions']['cpu'],.5);self.assertAlmostEqual(payload['free_fractions']['vram'],.5)
            remote.validate_safety_payload(payload,[],now=payload['observed_at'],max_age=30,mode='job-scoped')
            stat_file.write_text('garbage\n')
            with patch('subprocess.run',side_effect=run),patch.object(remote,'current_ram_fraction',return_value=.5):
                self.assertIsNone(remote.observe_job(root,cpu=remote.CpuSampler(path=stat_file,sleep=lambda _:None))['free_fractions']['cpu'])
                self.assertIsNone(remote.observe_job(root,cpu=remote.CpuSampler(path=root/'missing',sleep=lambda _:None))['free_fractions']['cpu'])


@unittest.skipIf(remote is None,'real adapter missing')
class JobScopedWriterInventoryTests(unittest.TestCase):
    def fixture(self,folder):
        return ProcExitRaceTests.writer_fixture(self,folder)

    def run_inventory(self,folder,*,mode,iterdir_map=None,readlink_map=None,extra_rows=(),drop_writers=(),foreign=None):
        import pathlib
        from unittest.mock import patch
        descriptor,rows,mapped,proc,identity=self.fixture(folder)
        for pid in drop_writers:os.unlink(proc/str(pid)/'fd'/'3')
        rows=rows+[{'pid':p,'ppid':1,'pgid':p,'sid':p,'birth':100+p,'state':'S'} for p in extra_rows]
        real_iterdir=pathlib.Path.iterdir;real_readlink=os.readlink
        def iterdir(self):
            for key,value in (iterdir_map or {}).items():
                if self==proc/key/'fd':
                    if isinstance(value,BaseException):raise value
                    return [proc/key/'fd'/str(i) for i in range(value)]
            return real_iterdir(self)
        def readlink(path,*a):
            for key,value in (readlink_map or {}).items():
                if str(path).startswith(str(proc/key/'fd')+'/'):
                    if isinstance(value,BaseException):raise value
                    return value
            return real_readlink(path,*a)
        foreign=foreign or patch.object(remote,'foreign_uid_fd_modes',side_effect=AssertionError('foreign UID probe called'))
        with patch.object(remote,'Path',mapped),patch.object(pathlib.Path,'iterdir',iterdir),patch.object(remote.os,'readlink',readlink),patch.object(remote,'linux_processes',return_value=rows) as table,patch.object(remote,'linux_publisher_identity',side_effect=identity),foreign as probe:
            value=remote.linux_writer_inventory(descriptor,mode=mode) if mode else remote.linux_writer_inventory(descriptor)
        return value,probe,table

    def test_job_scoped_skips_permission_denied_processes_without_foreign_probe(self):
        with tempfile.TemporaryDirectory() as folder:
            value,probe,table=self.run_inventory(os.path.realpath(folder),mode='job-scoped',iterdir_map={'11':PermissionError(13,'denied')},readlink_map={'12':PermissionError(13,'denied')})
        self.assertEqual([r['pid'] for r in value['writable_fd_processes']],[10]);self.assertEqual(value['unknown_writer_count'],0)
        probe.assert_not_called();self.assertEqual(table.call_args.kwargs.get('skip_unparseable'),True)

    def test_legacy_permission_denied_still_uses_foreign_probe_in_both_places(self):
        from unittest.mock import patch
        for where in ('iterdir','readlink'):
            with self.subTest(where=where),tempfile.TemporaryDirectory() as folder:
                args={'iterdir_map':{'11':PermissionError(13,'denied')}} if where=='iterdir' else {'readlink_map':{'11':PermissionError(13,'denied')}}
                value,probe,_=self.run_inventory(os.path.realpath(folder),mode=None,drop_writers=(12,),foreign=patch.object(remote,'foreign_uid_fd_modes',return_value=[]),**args)
                self.assertEqual([c.args[0]['pid'] for c in probe.call_args_list],[11]);self.assertEqual(value['expected_writer_count'],1)

    def test_job_scoped_readable_unknown_writer_still_stops(self):
        with tempfile.TemporaryDirectory() as folder,self.assertRaisesRegex(ValueError,'writer inventory'):
            self.run_inventory(os.path.realpath(folder),mode='job-scoped')

    def test_external_fd_bounds_do_not_stop_job_scoped_but_publisher_bound_does(self):
        with tempfile.TemporaryDirectory() as folder:
            value,_,_=self.run_inventory(os.path.realpath(folder),mode='job-scoped',drop_writers=(11,12),iterdir_map={'11':5000})
            self.assertEqual(value['expected_writer_count'],1)
        with tempfile.TemporaryDirectory() as folder,self.assertRaisesRegex(ValueError,'per-process fd bound'):
            self.run_inventory(os.path.realpath(folder),mode=None,drop_writers=(11,12),iterdir_map={'11':5000})
        many={str(p):4000 for p in range(20,30)};elsewhere={str(p):'/dev/null' for p in range(20,30)}
        with tempfile.TemporaryDirectory() as folder:
            value,_,_=self.run_inventory(os.path.realpath(folder),mode='job-scoped',drop_writers=(11,12),extra_rows=range(20,30),iterdir_map=many,readlink_map=elsewhere)
            self.assertEqual(value['expected_writer_count'],1)
        with tempfile.TemporaryDirectory() as folder,self.assertRaisesRegex(ValueError,'total FD inventory bound'):
            self.run_inventory(os.path.realpath(folder),mode=None,drop_writers=(11,12),extra_rows=range(20,30),iterdir_map=many,readlink_map=elsewhere)
        with tempfile.TemporaryDirectory() as folder,self.assertRaisesRegex(ValueError,'per-process fd bound'):
            self.run_inventory(os.path.realpath(folder),mode='job-scoped',drop_writers=(11,12),iterdir_map={'10':5000})

    def test_process_table_skips_unparseable_foreign_rows_only_on_request(self):
        from pathlib import Path as Real
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as folder:
            proc=Real(os.path.realpath(folder))
            for pid,text in (('10','garbage'),('11',ProcExitRaceTests.STAT.format(pid=11,birth=111)),('12','x'*5000)):
                (proc/pid).mkdir();(proc/pid/'stat').write_text(text)
            (proc/'self').mkdir()
            mapped=lambda value='',*a:proc if value=='/proc' else Real(value,*a)
            with patch.object(remote,'Path',mapped):
                self.assertEqual([r['pid'] for r in remote.linux_processes(skip_unparseable=True)],[11])
                with self.assertRaises(ValueError):remote.linux_processes()
                with self.assertRaises(ValueError):remote.linux_processes(skip_unparseable=True,owned={10})
                with self.assertRaises(ValueError):remote.linux_processes(skip_unparseable=True,owned={12})

    def test_job_scoped_kernel_table_skips_foreign_but_keeps_owned_fail_closed(self):
        from unittest.mock import patch
        kernel=object.__new__(remote.LinuxKernel);kernel.mode='job-scoped';kernel.owned_pids={41}
        with patch.object(remote,'linux_processes',return_value=[]) as table:kernel.processes()
        self.assertEqual(table.call_args.kwargs['skip_unparseable'],True);self.assertIn(41,table.call_args.kwargs['owned']);self.assertIn(os.getpid(),table.call_args.kwargs['owned'])
        legacy=object.__new__(remote.LinuxKernel)
        with patch.object(remote,'linux_processes',return_value=[]) as table:legacy.processes()
        self.assertEqual(table.call_args.kwargs,{})


@unittest.skipIf(remote is None,'real adapter missing')
class JobOwnerTests(unittest.TestCase):
    def test_owner_identity_zombie_absent_or_reparent_stops(self):
        from unittest.mock import patch
        owner={'pid':70,'ppid':1,'pgid':70,'sid':70,'birth':500,'state':'S'}
        with patch.object(remote,'linux_identity',return_value=dict(owner)),patch.object(remote.os,'getppid',return_value=70):
            self.assertTrue(remote.check_job_owner(owner))
        cases={'birth':({**owner,'birth':501},70),'zombie':({**owner,'state':'Z'},70),'ppid':(dict(owner),1),'absent':(FileNotFoundError(2,'gone'),70),'esrch':(ProcessLookupError(3,'gone'),70)}
        for name,(row,ppid) in cases.items():
            effect={'side_effect':row} if isinstance(row,BaseException) else {'return_value':row}
            with self.subTest(name=name),patch.object(remote,'linux_identity',**effect),patch.object(remote.os,'getppid',return_value=ppid),self.assertRaisesRegex(ValueError,'job owner lost'):
                remote.check_job_owner(owner)


@unittest.skipIf(remote is None,'real adapter missing')
class JobScopedPublisherTests(unittest.TestCase):
    identity={'pid':40,'start_ticks':100,'executable_sha256':'a'*64,'start_receipt_sha256':'b'*64}

    def test_job_scoped_envelope_requires_empty_runs_and_allowed_p(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(os.path.realpath(folder))/'feed'
            for runs,allowed,mode in (([{'run_id':'x'}],[],'job-scoped'),([],[self.identity],'job-scoped'),([],[],'shared')):
                with self.subTest(mode=mode,runs=runs,allowed=allowed),self.assertRaisesRegex(ValueError,'publisher pinned envelope'):
                    remote.serve_publisher(root,runs,deadline=time.time()+1,interval=.01,allowed_p=allowed,gate_mode=mode)
                self.assertFalse(root.exists())
            envelope=Path(os.path.realpath(folder))/'publisher.json';envelope.write_bytes(remote.canonical({'root':str(root),'runs':[],'deadline':time.time()+1,'interval':.01,'allowed_p':[],'gate_mode':'shared'}))
            with self.assertRaisesRegex(ValueError,'publisher pinned envelope'):
                remote.main(['serve-publisher','--manifest',str(envelope),'--sha256',hashlib.sha256(envelope.read_bytes()).hexdigest()])
            self.assertFalse(root.exists())

    def test_job_scoped_publisher_self_feeds_valid_receipt_chain(self):
        from unittest.mock import patch
        from vkgr import retrieval_sasrec_plan as plan
        modes=[]
        def inventory(descriptor,mode='legacy'):modes.append(mode);return {'inventory_sha256':'c'*64}
        with tempfile.TemporaryDirectory() as folder:
            root=Path(os.path.realpath(folder))/'feed'
            with patch.object(remote,'linux_publisher_identity',return_value=self.identity),patch.object(remote,'linux_writer_inventory',side_effect=inventory),patch.object(remote,'observe_job',side_effect=lambda disk_root,**k:job_payload(time.time())) as observe,patch.object(remote,'observe_g',side_effect=AssertionError('observe_g called')):
                result=remote.serve_publisher(root,[],deadline=time.time()+.08,interval=.02,allowed_p=[],gate_mode='job-scoped')
            self.assertEqual(result['status'],'CLOSED',result);self.assertEqual(modes,['job-scoped'])
            self.assertEqual(observe.call_args.args[0],root)
            descriptor=json.loads((root/'descriptor.json').read_bytes());plan.validate_feed_descriptor(descriptor)
            self.assertEqual(descriptor['g_job_identity']['job_id_hash'],remote.sha(remote.canonical([])))
            self.assertEqual(json.loads((root/'runs.json').read_bytes()),[])
            receipts=sorted(root.glob('receipt-*.json'));self.assertGreater(len(receipts),1)
            for path in receipts:
                receipt=json.loads(path.read_bytes());payload=json.loads((root/('payload-'+receipt['payload_sha256']+'.json')).read_bytes())
                remote.validate_safety_payload(payload,[],now=payload['observed_at'],max_age=30,mode='job-scoped')

    def test_job_scoped_publisher_start_survives_unreadable_foreign_fd_set(self):
        import pathlib
        from pathlib import Path as Real
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as folder:
            base=Real(os.path.realpath(folder));proc=base/'proc';root=base/'feed';me=str(os.getpid())
            for pid in (me,'11'):(proc/pid/'fd').mkdir(parents=True);(proc/pid/'fdinfo').mkdir()
            os.symlink(str(root/'publisher.lock'),proc/me/'fd'/'3');(proc/me/'fdinfo'/'3').write_text('pos:\t0\nflags:\t0100001\n')
            rows=[{'pid':os.getpid(),'ppid':1,'pgid':1,'sid':1,'birth':100,'state':'S'},{'pid':11,'ppid':1,'pgid':11,'sid':11,'birth':111,'state':'S'}]
            mapped=lambda value='',*a:proc if value=='/proc' else Real(value,*a)
            real_iterdir=pathlib.Path.iterdir
            def iterdir(self):
                if self==proc/'11'/'fd':raise PermissionError(13,'denied')
                return real_iterdir(self)
            identity={'pid':os.getpid(),'start_ticks':100,'executable_sha256':'a'*64,'start_receipt_sha256':'b'*64}
            with patch.object(remote,'Path',mapped),patch.object(pathlib.Path,'iterdir',iterdir),patch.object(remote,'linux_processes',return_value=rows),patch.object(remote,'linux_publisher_identity',return_value=identity),patch.object(remote,'foreign_uid_fd_modes',side_effect=AssertionError('foreign UID probe called')),patch.object(remote,'observe_job',side_effect=lambda disk_root,**k:job_payload(time.time())):
                result=remote.serve_publisher(root,[],deadline=time.time()+.05,interval=.02,allowed_p=[],gate_mode='job-scoped')
            self.assertIsNone(result['failure']);self.assertTrue((root/'initial-writer-inventory.json').exists())
            first=json.loads(next(root.glob('receipt-000000-*.json')).read_bytes());self.assertEqual(first['status'],'OPEN')


@unittest.skipIf(remote is None,'real adapter missing')
class JobScopedRuntimeGateTests(unittest.TestCase):
    def gate(self,root,payload,now,owner=None):
        identity={'pid':40,'start_ticks':100,'executable_sha256':'a'*64,'start_receipt_sha256':'b'*64};jobs={'job_id_hash':'c'*64,'start_identity_hash':'d'*64}
        journal=remote.FeedJournal(root/'feed',identity,jobs)
        journal.publish(payload,now=100,status='OPEN');journal.descriptor['initial_no_other_writer_inventory_sha256']='e'*64
        desc=root/'descriptor.json';pin=remote.write_once(desc,journal.descriptor);reader=remote.SafetyFeedReader(desc,pin,[],mode='job-scoped')
        budget=remote.MonotonicBudget({'kind':'pilot','started_at':100.,'deadline':14500.,'disk_cap':20*1024**3,'attempts':1,'started_monotonic':1000.,'deadline_monotonic':15400.,'host_boot_id':'boot'},monotonic=lambda:now[0]+900,boot_id='boot')
        limits={'stage_seconds':60,'finish_seconds':120,'next_atomic_bytes':1048576,'stop_reserve_bytes':1048576,'sample_seconds':5}
        return journal,remote.RuntimeGate(root,root/'control'/'stop.json',budget,reader,limits,clock=lambda:now[0],mode='job-scoped',owner=owner)

    def test_cached_and_fresh_branches_never_stop_on_cpu(self):
        from unittest.mock import patch
        for cpu in (.05,None):
            with self.subTest(cpu=cpu),tempfile.TemporaryDirectory() as folder:
                root=Path(os.path.realpath(folder));(root/'control').mkdir();now=[101.]
                journal,gate=self.gate(root,job_payload(cpu=cpu),now)
                try:
                    with patch.object(remote,'prove_publisher_runtime',return_value=True):
                        gate.check();now[0]=102.;gate.check()
                    self.assertEqual(len(gate.samples),1)
                    sample=gate.samples[-1];self.assertIn('cpu',sample['free_fractions']);self.assertEqual(sample['free_fractions']['cpu'],cpu)
                    self.assertEqual(set(sample['stage_fractions']),{'ram','vram','disk'})
                finally:journal.close()

    def test_low_vram_stops_and_owner_is_checked_on_fresh_sample(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as folder:
            root=Path(os.path.realpath(folder));(root/'control').mkdir();now=[101.]
            journal,gate=self.gate(root,job_payload(vram=.1),now)
            try:
                with patch.object(remote,'prove_publisher_runtime',return_value=True),self.assertRaisesRegex(ValueError,'shared resource STOP'):gate.check()
            finally:journal.close()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(os.path.realpath(folder));(root/'control').mkdir();now=[101.];owner={'pid':70,'ppid':1,'pgid':70,'sid':70,'birth':500,'state':'S'}
            journal,gate=self.gate(root,job_payload(),now,owner=owner)
            try:
                with patch.object(remote,'prove_publisher_runtime',return_value=True),patch.object(remote,'check_job_owner',return_value=True) as check:
                    gate.check();now[0]=102.;gate.check()
                check.assert_called_once_with(owner)
                now[0]=110.
                with patch.object(remote,'prove_publisher_runtime',return_value=True),patch.object(remote,'check_job_owner',side_effect=ValueError('job owner lost')),self.assertRaisesRegex(ValueError,'job owner lost'):gate.check()
            finally:journal.close()

    def test_job_scoped_reader_rejects_legacy_payload_feed(self):
        from unittest.mock import patch
        runs,legacy=SafetyFeedTests().fixture()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(os.path.realpath(folder));(root/'control').mkdir();now=[101.]
            journal,gate=self.gate(root,legacy,now)
            try:
                with patch.object(remote,'prove_publisher_runtime',return_value=True) as prove,self.assertRaisesRegex(ValueError,'gate mode/payload mismatch'):gate.check()
                self.assertEqual(prove.call_args.kwargs,{'mode':'job-scoped'})
            finally:journal.close()


@unittest.skipIf(remote is None,'real adapter missing')
class PProgressTests(unittest.TestCase):
    def run_stub(self,root,progress_path):
        from unittest.mock import MagicMock,patch
        value=real_manifest_fixture(root);value['stage']='formal-train'
        (root/'output').mkdir();(root/'control').mkdir();(root/'control'/'admission.json').write_text('{}')
        gate=MagicMock();gate.check.return_value={'receipt_sha256':'f'*64,'payload':{'protected_pgids':[]}};gate.samples=[]
        prepared={'gate':gate,'kernel':MagicMock(),'budget':MagicMock(),'reader':MagicMock(),'start':{'output_identity':{'device':1,'inode':2},'control_identity':{'device':1,'inode':3},'controller':{}},'start_sha256':'a'*64,'baseline':set()}
        env={'VKGR_RUN_ID':'p-run-01','VKGR_PROGRESS_PATH':str(progress_path)}
        with patch.dict(os.environ,env),patch.object(remote,'prepare_controller',return_value=prepared),patch.object(remote,'_read_json',return_value={'g_receipt_sha256':'f'*64}),patch.object(remote,'validate_admission',return_value=True),patch.object(remote,'start_latched_child',side_effect=ValueError('stub stop')):
            return remote.run_controller(value,str(root/'manifest.json'),'b'*64,[])

    def test_controller_writes_progress_outside_charged_tree(self):
        with tempfile.TemporaryDirectory() as folder,tempfile.TemporaryDirectory() as other:
            root=Path(os.path.realpath(folder));target=Path(os.path.realpath(other))/'progress.json'
            terminal=self.run_stub(root,target)
            self.assertEqual(terminal['status'],'failed+spent')
            data=target.read_bytes();self.assertLessEqual(len(data),4096);record=json.loads(data)
            self.assertEqual(set(record),{'schema','run_id','line','stage','phase','epoch','max_epochs','completed','total','message','updated_at','pid'})
            self.assertEqual((record['schema'],record['run_id'],record['line'],record['stage'],record['phase']),('vkgr-progress-v1','p-run-01','P','formal-train','sealing'))
            self.assertEqual([record[k] for k in ('epoch','max_epochs','completed','total')],[None]*4);self.assertEqual(record['pid'],os.getpid())

    def test_progress_inside_charged_tree_or_unwritable_never_stops(self):
        import contextlib,io
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as folder:
            root=Path(os.path.realpath(folder));target=root/'progress.json';err=io.StringIO()
            with contextlib.redirect_stderr(err):terminal=self.run_stub(root,target)
            self.assertEqual(terminal['status'],'failed+spent');self.assertFalse(target.exists());self.assertIn('progress write disabled',err.getvalue())
        with tempfile.TemporaryDirectory() as folder:
            target=Path(os.path.realpath(folder))/'p.json';err=io.StringIO()
            writer=remote.ProgressWriter(Path('/private/tmp/charged-elsewhere'),'pilot',environ={'VKGR_RUN_ID':'r','VKGR_PROGRESS_PATH':str(target)})
            with contextlib.redirect_stderr(err),patch.object(remote.os,'replace',side_effect=OSError(28,'full')):writer.write('worker-running')
            writer.write('worker-running');self.assertFalse(target.exists());self.assertIn('progress write disabled',err.getvalue())
            self.assertEqual(list(Path(folder).iterdir()),[])
        for environ in ({'VKGR_RUN_ID':'r'},{'VKGR_PROGRESS_PATH':'/private/tmp/x.json'},{'VKGR_RUN_ID':'r','VKGR_PROGRESS_PATH':'relative.json'},{}):
            with self.subTest(environ=environ),contextlib.redirect_stderr(io.StringIO()):
                writer=remote.ProgressWriter(Path('/private/tmp/charged-elsewhere'),'pilot',environ=environ);writer.write('awaiting-admission');self.assertIsNone(writer.path)
