"""Synthetic metadata-only recovery: no models, artifacts, or queue submissions."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
TOOL = HERE / 'analysis/sid_coll_configsha_handoff.py'

def sha(data): return hashlib.sha256(data).hexdigest()
def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(value, sort_keys=True).encode())
    return {'path': str(path), 'sha256': sha(path.read_bytes())}

class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(TOOL.is_file(), 'metadata-only config SHA handoff tool is missing')
        spec = importlib.util.spec_from_file_location('handoff', TOOL)
        self.m = importlib.util.module_from_spec(spec); spec.loader.exec_module(self.m)
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.p = Path(self.tmp.name).resolve(); self.repo = self.p/'source'; self.repo.mkdir()
        # Use the real frozen controller as an immutable Git object in a synthetic repository.
        raw = subprocess.check_output(['git','-C',str(HERE),'show','11d6768:scripts/sid_coll_ops.py'])
        for relative, data in {'scripts/sid_coll_ops.py':raw,'scripts/sid_coll_job_entry.py':b'# synthetic entry\n'}.items():
            dest=self.repo/relative; dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
        self.configs={}
        for label in ('C1p','C2'):
            for seed in (42,2024,2025):
                relative=f'configs/{label}-{seed}.json';data=json.dumps({'variant':label,'seed':seed}).encode()
                dest=self.repo/relative;dest.parent.mkdir(exist_ok=True);dest.write_bytes(data);self.configs[label,seed]=(relative,sha(data))
        subprocess.run(['git','init','-q',str(self.repo)],check=True)
        subprocess.run(['git','-C',str(self.repo),'add','.'],check=True)
        subprocess.run(['git','-C',str(self.repo),'-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-qm','fixture'],check=True)
        self.commit=subprocess.check_output(['git','-C',str(self.repo),'rev-parse','HEAD'],text=True).strip()
        self.pins={str(f.relative_to(self.repo)):sha(f.read_bytes()) for f in self.repo.rglob('*') if f.is_file() and '.git' not in f.parts}
        self.ops=self.p/'ops';self.ops.mkdir();self.old=self.ops/'controller';self.old.mkdir();(self.old/'driver.lock').touch()
        self.root=self.p/'queue-root'
        for kind in ('pending','claimed','rejected'):(self.root/'queue'/kind).mkdir(parents=True)
        stages=('input_preflight','input','input_verify','sid_fit','sid_verify','preflight','cuda','dry','train42','verify42','train2024','verify2024','train2025','verify2025')
        self.plan={'schema':'vkgr-sid-coll-plan-v1','root':str(self.root),'control_dir':str(self.old),'created_epoch':time.time()-10,'deadline_epoch':time.time()-10+172800,'common_refs':[],'branches':{}}
        self.plan['deadline_epoch']=self.plan['created_epoch']+172800
        self.ops_lib=self.p/'lib';(self.ops_lib/'vkgr_ops').mkdir(parents=True)
        (self.ops_lib/'vkgr_ops/__init__.py').write_text('# fixture\n')
        self.ops_identity={'commit':'a'*40,'sha256':self.m.digest({'vkgr_ops/__init__.py':sha((self.ops_lib/'vkgr_ops/__init__.py').read_bytes())})}
        for label in ('C1p','C2'):
            branch=self.plan['branches'][label]={'stages':{}}
            for name in stages:
                generic='train' if name.startswith('train') else 'verify' if name.startswith('verify') else name
                seed=int(name[len(generic):]) if generic in ('train','verify') else 42
                rid=label.lower()+'-'+name
                refs={f'config{s}':{'path':str(self.repo/self.configs[label,s][0]),'sha256':self.configs[label,s][1]} for s in (42,2024,2025)}
                spec={'commit':self.commit,'source_files':self.pins,'variant':label,'refs':refs,'stages':{generic:{'run_id':rid,'parameters':{'seed':seed}}}}
                spec_ref=write(self.ops/(rid+'.json'),spec)
                argv=[sys.executable,'-B',str(self.repo/'scripts/sid_coll_job_entry.py'),'--stage',generic,'--spec',spec_ref['path'],'--spec-sha',spec_ref['sha256'],'--output',str(self.root/'runs'/rid/'out')]
                job={'schema':'vkgr-queue-job-v1','run_id':rid,'kind':'ops','cwd':str(self.repo),'argv':argv,'env':['PYTHONPATH='+str(self.repo/'src'),'OMP_NUM_THREADS=1'],'config':spec_ref['path'],'seed':seed,'inputs':[spec_ref['path']],'pinned_manifest':None,'pinned_files':[],'cpu_cores':1,'ram_mib':8192,'vram_mib':0,'disk_gib':4,'wall_hours':24.0,'stop_mode':'signal','no_admission':False}
                expected={'schema':'coll-stage-receipt-v1','status':'complete' if generic=='train' else 'verified','stage':generic,'variant':label,'run_id':rid,'binding.commit':self.commit,'binding.spec_sha256':spec_ref['sha256'],'binding.config_sha256':None,'binding.data':{k:v['sha256'] for k,v in refs.items()}}
                branch['stages'][name]={'job':job,'proof':{'path':str(self.root/'runs'/rid/'out/receipt.json'),'expected':expected}}
        dead=subprocess.Popen([sys.executable,'-c','pass']);dead.wait()
        self.owner=write(self.old/'driver-pid-v3.json',{'pid':dead.pid,'ppid':1,'poll_seconds':900,'deadline_epoch':self.plan['deadline_epoch']})
        write(self.old/'alert-C1p.json',{'reason':'stage train42 failed'});write(self.old/'alert-C2.json',{'reason':'stage train42 failed'})
        write(self.old/'summary-final.json',{'status':'ALERT'})
        self.original=write(self.ops/'old-plan.json',self.plan)
        self.approval=write(self.ops/'approval.json',{'source':'Eddy direct reply in this conversation','question_item_id':'synthetic-direct','answer':'批准按提案恢复','scope':['synthetic metadata-only recovery'],'not_yet_executed':True})
        self.req={'schema':'vkgr-sid-coll-configsha-handoff-v1','old_plan':self.original,'old_control_dir':str(self.old),'new_plan':str(self.ops/'new-plan.json'),'new_control_dir':str(self.ops/'recovery'),'old_owner':self.owner,'old_pid':dead.pid,'created_epoch':self.plan['created_epoch'],'deadline_epoch':self.plan['deadline_epoch'],'scientific_commit':self.commit,'git_repo':str(self.repo),'model_root':str(self.repo),'source_files':self.pins,'python':sys.executable,'ops_library':str(self.ops_lib),'ops_files':{'vkgr_ops/__init__.py':sha((self.ops_lib/'vkgr_ops/__init__.py').read_bytes())},'run_ops_source':self.ops_identity,'approval':self.approval}
        for label in self.plan['branches']:
            for name in stages[:9]:self.complete(label,name)
        self.helper=self.p/'helper'; (self.helper/'analysis').mkdir(parents=True)
        (self.helper/'analysis/sid_coll_configsha_handoff.py').write_bytes(TOOL.read_bytes())
        subprocess.run(['git','init','-q',str(self.helper)],check=True)
        subprocess.run(['git','-C',str(self.helper),'add','.'],check=True)
        subprocess.run(['git','-C',str(self.helper),'-c','user.name=Fixture','-c','user.email=fixture@example.invalid','commit','-qm','helper'],check=True)
        self.req.update(helper_root=str(self.helper),helper_commit=subprocess.check_output(['git','-C',str(self.helper),'rev-parse','HEAD'],text=True).strip(),helper_source_sha256=sha(TOOL.read_bytes()))

    def complete(self,label,name):
        stage=self.plan['branches'][label]['stages'][name];job=copy.deepcopy(stage['job']);rid=job['run_id']
        write(self.root/'queue/claimed'/('1-'+rid+'.json'),job)
        rd=self.root/'runs'/rid
        write(rd/'terminal.json',{'schema':'vkgr-terminal-v1','run_id':rid,'status':'succeeded','exit_code':0})
        proof={};expected=stage['proof']['expected']
        for key,val in expected.items():
            dest=proof;parts=key.split('.')
            for part in parts[:-1]:dest=dest.setdefault(part,{})
            dest[parts[-1]]=copy.deepcopy(val)
        if name.startswith(('train','verify')):
            seed=job['seed'];proof['binding']['config_sha256']=self.configs[label,seed][1]
        proof['raw_proof']={'forbidden_unverified_score':'DO_NOT_INSPECT'}
        write(Path(stage['proof']['path']),proof)
        inputs=[{'path':p,'sha256':sha(Path(p).read_bytes()),'bytes':Path(p).stat().st_size} for p in job['inputs']]
        manifest={'schema':'vkgr-run-v1','run_id':rid,'kind':job['kind'],'cwd':job['cwd'],'argv':job['argv'],'env':dict(x.split('=',1) for x in job['env']),'git':{'commit':self.commit,'dirty':False,'dirty_files':[]},'overlay':None,'pinned_files':{},'config':{'path':job['config'],'sha256':sha(Path(job['config']).read_bytes())},'seed':job['seed'],'inputs':inputs,'resources':{k:job[k] for k in ('cpu_cores','ram_mib','vram_mib','disk_gib')}|{'wall_seconds':job['wall_hours']*3600},'stop_mode':job['stop_mode'],'ops_source':self.ops_identity}
        write(rd/'run.json',manifest)
        write(self.old/('intent-'+rid+'.json'),{'run_id':rid,'job_sha256':self.m.digest(job),'created_utc':'synthetic'})

    def rewrite_plan(self):self.req['old_plan']=write(Path(self.original['path']),self.plan)

    def test_exact_patch_twelve_and_no_other_changes(self):
        result=self.m.inventory(self.req)
        fixed=copy.deepcopy(self.plan);fixed['control_dir']=self.req['new_control_dir']
        for label in fixed['branches']:
            for generic in ('train','verify'):
                for seed in (42,2024,2025):fixed['branches'][label]['stages'][generic+str(seed)]['proof']['expected']['binding.config_sha256']=self.configs[label,seed][1]
        self.assertEqual(result['new_plan'],fixed);self.assertEqual(len(result['patches']),12)
        self.assertNotIn('DO_NOT_INSPECT',json.dumps(result))
        self.assertFalse(Path(self.req['new_control_dir']).exists())

    def test_prepare_exclusive_preserves_guards_and_old_stop(self):
        before={p: p.read_bytes() for p in self.old.iterdir()}
        result=self.m.prepare(self.req)
        new=Path(self.req['new_control_dir'])
        self.assertEqual(json.loads(Path(self.req['new_plan']).read_text()),result['new_plan'])
        for path,data in before.items():
            self.assertEqual(path.read_bytes(),data)
            if path.name.startswith('intent-'):self.assertEqual((new/path.name).read_bytes(),data)
        self.assertFalse((new/'alert-C1p.json').exists())
        with self.assertRaises((ValueError,FileExistsError)):self.m.prepare(self.req)

    def test_verify_pending_adopted_not_submitted(self):
        stage=self.plan['branches']['C1p']['stages']['verify42'];job=stage['job'];rid=job['run_id']
        write(self.root/'queue/pending'/('2-'+rid+'.json'),job)
        write(self.old/('intent-'+rid+'.json'),{'run_id':rid,'job_sha256':self.m.digest(job),'created_utc':'synthetic'})
        result=self.m.inventory(self.req)
        self.assertEqual(result['runs'][rid]['status'],'queued')

    def test_uncertain_verify_intent_stops(self):
        job=self.plan['branches']['C1p']['stages']['verify42']['job'];rid=job['run_id']
        write(self.old/('intent-'+rid+'.json'),{'run_id':rid,'job_sha256':self.m.digest(job),'created_utc':'synthetic'})
        with self.assertRaisesRegex(ValueError,'uncertain'):self.m.inventory(self.req)

    def test_rejected_verify_and_reason_stops(self):
        job=self.plan['branches']['C1p']['stages']['verify42']['job'];rid=job['run_id']
        write(self.root/'queue/rejected'/('1-'+rid+'.json'),job)
        (self.root/'queue/rejected'/('1-'+rid+'.reason')).write_text('failure')
        with self.assertRaisesRegex(ValueError,'rejected'):self.m.inventory(self.req)

    def test_failed_train_never_repaired(self):
        rid=self.plan['branches']['C1p']['stages']['train42']['job']['run_id']
        write(self.root/'runs'/rid/'terminal.json',{'run_id':rid,'status':'failed','exit_code':1})
        with self.assertRaisesRegex(ValueError,'failed'):self.m.inventory(self.req)

    def test_manifest_and_queue_identity_mutations_stop(self):
        stage=self.plan['branches']['C1p']['stages']['train42'];rid=stage['job']['run_id'];path=self.root/'runs'/rid/'run.json'
        original=json.loads(path.read_text());changed=copy.deepcopy(original);changed['seed']=2025;write(path,changed)
        with self.assertRaisesRegex(ValueError,'manifest'):self.m.inventory(self.req)
        write(path,original);p=self.root/'queue/claimed'/('1-'+rid+'.json');job=json.loads(p.read_text());job['ram_mib']+=1;write(p,job)
        with self.assertRaisesRegex(ValueError,'queue'):self.m.inventory(self.req)

    def test_second_train_identity_difference_stops(self):
        stage=self.plan['branches']['C1p']['stages']['train42'];p=Path(stage['proof']['path']);value=json.loads(p.read_text());value['binding']['data']['config42']='e'*64;write(p,value)
        with self.assertRaisesRegex(ValueError,'receipt'):self.m.inventory(self.req)

    def test_spec_seed_and_git_config_tamper_stop(self):
        stage=self.plan['branches']['C1p']['stages']['verify2024'];p=Path(stage['job']['config']);value=json.loads(p.read_text());value['stages']['verify']['parameters']['seed']=2025;ref=write(p,value);stage['proof']['expected']['binding.spec_sha256']=ref['sha256'];stage['job']['argv'][stage['job']['argv'].index('--spec-sha')+1]=ref['sha256'];self.rewrite_plan()
        with self.assertRaisesRegex(ValueError,'seed'):self.m.inventory(self.req)

    def test_old_owner_alive_and_held_lock_stop(self):
        req=copy.deepcopy(self.req);req['old_pid']=os.getpid();req['old_owner']=write(self.old/'live.json',{'pid':os.getpid(),'ppid':1,'poll_seconds':900,'deadline_epoch':req['deadline_epoch']})
        with self.assertRaisesRegex(ValueError,'alive'):self.m.inventory(req)
        import fcntl
        with (self.old/'driver.lock').open('rb') as fd:
            fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with self.assertRaisesRegex(ValueError,'lock'):self.m.inventory(self.req)

    def test_request_python_deadline_and_extra_env_rejected(self):
        for key,value in [('python','/usr/bin/false'),('deadline_epoch',self.req['deadline_epoch']+1),('env',{'PYTHONPATH':'bad'})]:
            req=copy.deepcopy(self.req);req[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):self.m.inventory(req)

    def test_guard_tamper_after_prepare_stops_start_before_spawn(self):
        self.m.prepare(self.req)
        guard=next(self.old.glob('intent-*.json'));guard.write_text('{}')
        with patch.object(self.m.subprocess,'Popen',side_effect=AssertionError('must not spawn')):
            with self.assertRaises(ValueError):self.m.start(self.req)

    def test_stdin_transport_has_no_file_global_dependency(self):
        import io
        payload={'action':'inventory','request':self.req}
        with patch.object(self.m.sys,'stdin',io.StringIO(json.dumps(payload))),patch.object(self.m.sys,'stdout',io.StringIO()) as out:
            self.assertEqual(self.m.main([]),0)
        self.assertEqual(json.loads(out.getvalue())['status'],'inventory_ok')

    def test_helper_source_tamper_and_wrong_head_stop(self):
        req=copy.deepcopy(self.req);req['helper_commit']='e'*40
        with self.assertRaises((ValueError,subprocess.CalledProcessError)):self.m.inventory(req)
        (self.helper/'analysis/sid_coll_configsha_handoff.py').write_text('# changed')
        with self.assertRaisesRegex(ValueError,'helper'):self.m.inventory(self.req)

    def test_config_does_not_need_to_be_a_python_source_pin(self):
        for key in list(self.pins):
            if key.startswith('configs/'):del self.pins[key]
        self.req['source_files']=self.pins
        for branch in self.plan['branches'].values():
            for stage in branch['stages'].values():
                p=Path(stage['job']['config']);value=json.loads(p.read_text());value['source_files']=self.pins
                newref=write(p,value);stage['proof']['expected']['binding.spec_sha256']=newref['sha256'];argv=stage['job']['argv'];argv[argv.index('--spec-sha')+1]=newref['sha256']
        self.rewrite_plan()
        # Recreate accepted receipts/manifests with the updated synthetic spec pins.
        for label in self.plan['branches']:
            for name in ('input_preflight','input','input_verify','sid_fit','sid_verify','preflight','cuda','dry','train42'):self.complete(label,name)
        self.assertEqual(len(self.m.inventory(self.req)['patches']),12)

    def test_owner_lock_parser_requires_same_actual_pid(self):
        info=os.stat(self.old/'driver.lock')
        dev=f'{os.major(info.st_dev):02x}:{os.minor(info.st_dev):02x}:{info.st_ino}'
        self.assertTrue(self.m.lock_owned_by('1: FLOCK ADVISORY WRITE 123 '+dev+' 0 EOF',info,123))
        self.assertFalse(self.m.lock_owned_by('1: FLOCK ADVISORY WRITE 124 '+dev+' 0 EOF',info,123))
        self.assertFalse(self.m.lock_owned_by('1: POSIX ADVISORY WRITE 123 '+dev+' 0 EOF',info,123))

    def test_owner_waits_for_launchers_global_lock_release(self):
        with self.m.global_lock(self.req):
            with self.assertRaises(ValueError):self.m.global_lock(self.req,wait=.02)
        with self.m.global_lock(self.req,wait=.02):pass

    def test_rejected_or_uncertain_future_stage_never_hidden(self):
        stage=self.plan['branches']['C2']['stages']['verify2025'];rid=stage['job']['run_id']
        write(self.old/('intent-'+rid+'.json'),{'run_id':rid,'job_sha256':self.m.digest(stage['job']),'created_utc':'synthetic'})
        with self.assertRaisesRegex(ValueError,'uncertain|unresolved'):self.m.inventory(self.req)

    def test_deadline_expiry_source_tamper_and_partial_newdir_stop(self):
        with patch.object(self.m.time,'time',return_value=self.req['deadline_epoch']):
            with self.assertRaisesRegex(ValueError,'expired'):self.m.inventory(self.req)
        Path(self.req['new_control_dir']).mkdir()
        with self.assertRaises(FileExistsError):self.m.prepare(self.req)
        (self.repo/'scripts/sid_coll_job_entry.py').write_text('# tamper')
        with self.assertRaisesRegex(ValueError,'source'):self.m.inventory(self.req)

    def test_pinned_file_cli_rejects_wrong_sha_and_duplicate_json(self):
        path=self.ops/'request.json';pin=write(path,self.req)['sha256']
        with self.assertRaisesRegex(ValueError,'SHA'):self.m.main(['inventory','--request',str(path),'--request-sha','0'*64])
        path.write_text('{"x":1,"x":2}')
        with self.assertRaisesRegex(ValueError,'duplicate'):self.m.main(['inventory','--request',str(path),'--request-sha',sha(path.read_bytes())])

    def test_real_original_tick_adopts_train42_and_submits_only_missing_verify_once(self):
        result=self.m.prepare(self.req);plan=result['new_plan'];stage=plan['branches']['C2']['stages']['verify42'];job=stage['job']
        write(self.root/'queue/pending'/('2-'+job['run_id']+'.json'),job)
        spec=importlib.util.spec_from_file_location('frozen_original',self.repo/'scripts/sid_coll_ops.py')
        op=importlib.util.module_from_spec(spec);spec.loader.exec_module(op)
        calls=[]
        def publish(root,job):
            calls.append(job['run_id']);return write(Path(root)/'queue/pending'/('3-'+job['run_id']+'.json'),job)['path']
        one=op.tick(plan,submitter=publish);two=op.tick(plan,submitter=publish)
        self.assertEqual(calls,['c1p-verify42'])
        self.assertEqual(one['branches']['C1p']['stages']['train42']['status'],'accepted')
        self.assertEqual(two['new_submissions'],[])
        waiter=op.EventWaiter(plan)
        try:self.assertEqual(waiter.metadata()['fallback_seconds'],900)
        finally:waiter.close()

    def test_detached_process_no_publication_before_release_then_survives_controller(self):
        # Real immutable controller + real locks/process; synthetic publisher only.
        (self.ops_lib/'vkgr_ops/launcher.py').write_text('def prepare(job): return {"synthetic": True}\n')
        (self.ops_lib/'vkgr_ops/scheduler.py').write_text('import json\nfrom pathlib import Path\ndef submit(root,job):\n p=Path(root)/"queue/pending"/("9-"+job["run_id"]+".json")\n with p.open("x") as f: json.dump(job,f)\n return p\n')
        self.req['ops_files']={str(p.relative_to(self.ops_lib)):sha(p.read_bytes()) for p in (self.ops_lib/'vkgr_ops').glob('*.py')}
        self.ops_identity['sha256']=self.m.digest(self.req['ops_files'])
        for path in (self.root/'runs').glob('*/run.json'):
            value=json.loads(path.read_text());value['ops_source']=self.ops_identity;write(path,value)
        self.m.prepare(self.req)
        self.m.once(Path(self.req['new_control_dir'])/'launch-intent.json',self.m.encoded({'request_sha256':self.m.digest(self.req),'at_utc':'synthetic','deadline_epoch':self.req['deadline_epoch']}))
        request_file=write(self.ops/'request.json',self.req)
        controller=subprocess.Popen([sys.executable,'-B',str(self.helper/'analysis/sid_coll_configsha_handoff.py'),'_detach','--request',request_file['path'],'--request-sha',request_file['sha256']],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,env=self.m.owner_env(self.req),start_new_session=True)
        controller.wait(timeout=10);self.assertEqual(controller.returncode,0)
        ctl=Path(self.req['new_control_dir']);ready=ctl/'owner-ready.json';end=time.monotonic()+15
        while not ready.exists() and time.monotonic()<end:time.sleep(.05)
        self.assertTrue(ready.exists(),'detached owner did not reach locked readiness')
        value=json.loads(ready.read_text());pid=value['pid'];self.addCleanup(self.stop_fixture_owner,pid)
        self.assertEqual(value['ppid'],1);self.assertTrue(self.m.alive(pid))
        time.sleep(.2);self.assertEqual(list((self.root/'queue/pending').iterdir()),[])
        for lock in (self.old/'driver.lock',self.m.global_path(self.req),ctl/'driver.lock'):self.assertTrue(self.m.held(lock))
        # The controller has exited. Only an explicit pinned release permits ticks.
        self.m.release(self.req,value)
        end=time.monotonic()+10
        while len(list((self.root/'queue/pending').glob('*.json')))<2 and time.monotonic()<end:time.sleep(.05)
        self.assertEqual(sorted(p.name for p in (self.root/'queue/pending').glob('*.json')),['9-c1p-verify42.json','9-c2-verify42.json'])
        self.assertTrue(self.m.alive(pid))
        with self.assertRaises(FileExistsError):self.m.start(self.req,request_file['path'],request_file['sha256'])

    def stop_fixture_owner(self,pid):
        import signal
        try:os.kill(pid,signal.SIGTERM)
        except ProcessLookupError:pass
        end=time.monotonic()+3
        while self.m.alive(pid) and time.monotonic()<end:time.sleep(.05)

    def test_prepare_keeps_old_lock_until_all_guards_are_sealed(self):
        original=self.m.once;observed=[]
        def checked(path,raw):
            observed.append(self.m.held(self.old/'driver.lock'))
            return original(path,raw)
        with patch.object(self.m,'once',side_effect=checked):self.m.prepare(self.req)
        self.assertTrue(observed);self.assertTrue(all(observed))

    def test_approval_answer_and_ops_source_aggregate_must_match(self):
        original=copy.deepcopy(self.req)
        self.req['approval']=write(self.ops/'not-approved.json',{'source':'Eddy direct reply in this conversation','question_item_id':'synthetic','answer':'not approved','scope':['x'],'not_yet_executed':True})
        with self.assertRaisesRegex(ValueError,'approval'):self.m.inventory(self.req)
        self.req=original;self.req['run_ops_source']['sha256']='e'*64
        with self.assertRaisesRegex(ValueError,'ops source'):self.m.inventory(self.req)

    def test_detach_requires_controller_launch_intent_before_fork(self):
        self.m.prepare(self.req);request_file=write(self.ops/'request.json',self.req)
        with patch.object(self.m.os,'fork',side_effect=AssertionError('fork must not run')):
            with self.assertRaises((ValueError,FileNotFoundError)):self.m.main(['_detach','--request',request_file['path'],'--request-sha',request_file['sha256']])

    def test_ack_requires_live_detached_pid_and_lock(self):
        self.m.prepare(self.req)
        with self.assertRaises((ValueError,FileNotFoundError)):self.m.ack(self.req)

class RelativeSourcePaths(unittest.TestCase):
    def test_only_exact_root_uv_lock_is_allowed_as_lockfile(self):
        spec=importlib.util.spec_from_file_location('handoff_relative',TOOL)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.assertEqual(module.relative('uv.lock'),Path('uv.lock'))
        for value in ('other.lock','uv.lock.pt','checkpoint.pt','checkpoint.ckpt',
                      'vectors.npy','nested/uv.lock','../uv.lock','/tmp/uv.lock'):
            with self.subTest(path=value),self.assertRaises(ValueError):module.relative(value)

if __name__=='__main__':unittest.main()
