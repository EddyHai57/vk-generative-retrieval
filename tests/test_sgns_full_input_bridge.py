"""Transport-only source bridge tests; no real PF/training/acceptance fixtures."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace


class FullInputBridge(unittest.TestCase):
    def test_closed_source_changes_and_additions(self):
        from analysis import sgns_full_input_bridge as b
        pins={'old_source':{'commit':'a'*40,'files':{'science.py':'1'*64,'adapter.py':'2'*64}},
              'allowed_changed_paths':['adapter.py'],'added_paths':['bridge.py']}
        current={'commit':'b'*40,'files':{'science.py':'1'*64,'adapter.py':'3'*64,'bridge.py':'4'*64}}
        self.assertEqual(b.source_delta(current,copy.deepcopy(current),pins)['changed']['adapter.py'],{'before':'2'*64,'after':'3'*64})
        for member in ('science.py','unknown.py'):
            bad=copy.deepcopy(current);bad['files'][member]='5'*64
            with self.subTest(member=member),self.assertRaises(ValueError):b.source_delta(bad,bad,pins)
        expected=copy.deepcopy(current);expected['files']['adapter.py']='6'*64
        with self.assertRaises(ValueError):b.source_delta(current,expected,pins)

    def test_override_rejects_nonfixed_bytes_before_parse(self):
        from analysis import sgns_full_input_bridge as b
        from analysis.sgns_subset_acceptance import sha
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp).resolve()/'fake';p.write_bytes(b'{}')
            record={'path':str(p),'sha256':sha(b'{}')}
            with self.assertRaises(ValueError):b.resource_override({k:record for k in ('plan','before','after')})

    def test_atomic_replace_keeps_backup_and_refuses_state_changes(self):
        from analysis import sgns_full_input_bridge as b
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();ops=root/'OPS';q=ops/'queue/pending/job.json';q.parent.mkdir(parents=True)
            q.write_bytes(b'before');rid='transport-fixture'
            result=b.replace_pending(q,rid,b'before',b'after',b'plan',root/'backup')
            self.assertEqual(q.read_bytes(),b'after');self.assertEqual((root/'backup/before.json').read_bytes(),b'before')
            self.assertEqual(result['status'],'pending_resource_replaced')
            with self.assertRaises((ValueError,FileExistsError)):b.replace_pending(q,rid,b'before',b'after',b'plan',root/'other')

    def test_each_atomic_state_refused_independently(self):
        from analysis import sgns_full_input_bridge as b
        import fcntl
        for state in ('run','claimed','rejected','backup','temp','wrong-before','locked'):
            with self.subTest(state=state),tempfile.TemporaryDirectory() as temp:
                root=Path(temp).resolve();ops=root/'OPS';rid='transport-fixture'
                q=ops/'queue/pending'/('000-'+rid+'.json');q.parent.mkdir(parents=True);q.write_bytes(b'before')
                backup=root/'backup'
                if state in ('run','claimed','rejected'):
                    target=ops/'runs'/rid if state=='run' else ops/'queue'/state/q.name
                    target.parent.mkdir(parents=True,exist_ok=True)
                    if state=='run':target.mkdir()
                    else:target.write_bytes(b'occupied')
                if state=='backup':backup.mkdir()
                if state=='temp':q.with_name(q.name+'.resource-'+b.sha(b'plan')[:16]+'.tmp').write_bytes(b'occupied')
                lock=None
                if state=='locked':
                    lock=(ops/'scheduler.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                try:
                    with self.assertRaises((ValueError,FileExistsError,BlockingIOError)):
                        b.replace_pending(q,rid,b'wrong' if state=='wrong-before' else b'before',b'after',b'plan',backup)
                finally:
                    if lock:lock.close()
                self.assertEqual(q.read_bytes(),b'before')

    def test_atomic_recheck_before_replace(self):
        from analysis import sgns_full_input_bridge as b
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();ops=root/'OPS';rid='transport-fixture'
            q=ops/'queue/pending'/('000-'+rid+'.json');q.parent.mkdir(parents=True);q.write_bytes(b'before')
            original=b.os.fsync
            def race(fd):
                original(fd)
                (ops/'runs'/rid).mkdir(parents=True,exist_ok=True)
            with patch.object(b.os,'fsync',side_effect=race),self.assertRaises(ValueError):
                b.replace_pending(q,rid,b'before',b'after',b'plan',root/'backup')
            self.assertEqual(q.read_bytes(),b'before')

    def test_transport_only_override_semantics(self):
        from analysis import sgns_full_input_bridge as b
        pins=b.load_pins()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve()
            def records(change):
                before={'schema':'vkgr-queue-job-v1','run_id':pins['run_id'],'ram_mib':pins['old_ram_mib'],'seed':42,'cwd':str(root/'CODE')}
                after={**before,'ram_mib':pins['new_ram_mib'],**change}
                raw={'before':b.encoded(before),'after':b.encoded(after)}
                plan={'schema':'vkgr-full-input-pf-resource-override-v1','authority':'3c4cd42','run_id':pins['run_id'],
                    'source_commit':pins['old_source']['commit'],'before_sha256':b.sha(raw['before']),'after_sha256':b.sha(raw['after']),
                    'before_bytes':len(raw['before']),'after_bytes':len(raw['after']),
                    'allowed_change':{'ram_mib':{'before':pins['old_ram_mib'],'after':pins['new_ram_mib']}},
                    'code_root':before['cwd'],'queue_path':str(root/'OPS/queue/pending/job.json')}
                raw['plan']=b.encoded(plan);result={};local=copy.deepcopy(pins)
                for name,data in raw.items():
                    path=root/name;path.write_bytes(data);local[name+'_sha256']=b.sha(data)
                    result[name]={'path':str(path),'sha256':b.sha(data)}
                return result,local
            result,local=records({})
            with patch.object(b,'load_pins',return_value=local):self.assertEqual(b.resource_override(result)['after']['ram_mib'],196608)
            wrong=copy.deepcopy(local);wrong['plan_sha256']='f'*64
            with patch.object(b,'load_pins',return_value=wrong),self.assertRaises(ValueError):b.resource_override(result)
            for change in ({'seed':43},{'ram_mib':1},{'cwd':str(root/'foreign')}):
                result,local=records(change)
                with self.subTest(change=change),patch.object(b,'load_pins',return_value=local),self.assertRaises(ValueError):b.resource_override(result)

    def test_prepare_bridge_hook_precedes_scientific_inputs(self):
        from analysis import sgns_gpu_full as s,sgns_full_input_bridge as b
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();args=SimpleNamespace(output=root/'OUT',deployment_sha='a'*40,lock=root/'LOCK',source_bridge_request=root/'REQUEST',source_bridge_sha256='b'*64)
            with patch.object(s,'source_identity',return_value={'commit':'a'*40}),patch.object(s,'check_source'),patch.object(s,'runtime'),patch.object(b,'preflight',side_effect=ValueError('bridge blocked')),patch.object(s,'stage2_binding',side_effect=AssertionError('science read')) as science:
                with self.assertRaisesRegex(ValueError,'bridge blocked'):s.prepare(args,readonly=False)
                science.assert_not_called()
            manifest=root/'manifest.json';manifest.write_bytes(b'{}')
            value={'schema':'gpu-sgns-full-input-v1','status':'prepared','engineering':False,'training':False,'test_read':False,
                'source':{},'runtime':{},'records':s.fixed_records(),'preflight_job':{'opaque':'transport'},'source_bridge':{}}
            with patch.object(s,'pinned_json',return_value=value),patch.object(s,'fixed_records',return_value=value['records']),patch.object(b,'verify_manifest',side_effect=ValueError('bridge bundle blocked')):
                with self.assertRaisesRegex(ValueError,'bridge bundle blocked'):s.read_bundle(root,s.digest(manifest),{},{})

    def test_prepare_argv_and_pair_rejection(self):
        from analysis import sgns_gpu_full as s
        args=SimpleNamespace(**{n:'/transport/'+n for n in ('data','embedding_root','stage2','stage2_run','lock','output','deployment_sha')},source_bridge_request=Path('/transport/request'),source_bridge_sha256='a'*64)
        argv=s.input_argv('prepare',args)
        self.assertEqual(argv[19:],['--source-bridge-request','/transport/request','--source-bridge-sha256','a'*64])
        raw=s.input_argv('preflight',args)
        self.assertEqual(len(raw),19)
        run=subprocess.run(argv,capture_output=True,text=True)
        self.assertNotEqual(run.returncode,0)

    def test_bridged_launcher_contract_exact_inputs_and_resources(self):
        from analysis import sgns_gpu_full as s,sgns_full_input_bridge as b
        args=SimpleNamespace(**{n:'/transport/'+n for n in ('data','embedding_root','stage2','stage2_run','lock','output')},deployment_sha='a'*40,
            source_bridge_request=Path('/transport/request'),source_bridge_sha256='b'*64)
        source={'commit':'a'*40};argv=s.input_argv('prepare',args)+['--preflight-run','/transport/PF']
        fake=lambda path,digest:{'path':str(path),'sha256':digest,'bytes':1}
        records=[{'path':'/transport/'+n,'sha256':'c'*64} for n in ('request','plan','before','after')]
        dag=s.fixed_dag()
        with patch.object(s,'fixed_dag',return_value=dag),patch.object(s,'file_record',side_effect=fake),patch.object(s,'digest',return_value='d'*64),patch.object(b,'bridge_inputs',return_value=records):
            value=s.full_job_contract(Path('/transport/PREPARE'),argv,source)
            self.assertEqual(value['resources']['ram_mib'],196608)
            self.assertEqual(value['resources']['wall_seconds'],86400.0)
            self.assertEqual(value['argv'],argv)
            self.assertEqual([x['path'] for x in value['inputs']][-4:],[x['path'] for x in records])
            self.assertEqual(s.full_job_contract(Path('/transport/PF'),s.input_argv('preflight',args),source)['resources']['ram_mib'],262144)
            for position in (0,3,4,5,18,19,21,23):
                bad=list(argv);bad[position]='foreign'
                with self.subTest(position=position),self.assertRaises(ValueError):s.full_job_contract(Path('/transport/PREPARE'),bad,source)

    def test_old_contract_exact_transport_metadata(self):
        from analysis import sgns_gpu_full as s,sgns_full_input_bridge as b
        pins=b.load_pins();code=Path('/transport/OLD_CODE');run=Path('/transport')/pins['run_id']
        args=SimpleNamespace(**{n:'/transport/'+n for n in ('data','embedding_root','stage2','stage2_run','lock','output')},deployment_sha=pins['old_source']['commit'])
        argv=s.input_argv('preflight',args)
        env={'PYTHONPATH':str(code/'src'),'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8'}
        spec={'argv':argv,'cwd':str(code),'config':str(code/'configs/full-gpu-sgns/fixed-dag.json'),
            'inputs':['/transport/lock','/transport/stage2/UR001-manifest.json'],'env':[k+'='+v for k,v in env.items()],
            'kind':'ops','seed':42,'cpu_cores':1,'ram_mib':196608,'vram_mib':0,'disk_gib':16,'wall_hours':24,
            'stop_mode':'signal','no_admission':False,'pinned_manifest':None,'pinned_files':[]}
        value={'old_code_root':str(code),'preflight_run':str(run)}
        with patch.object(s,'file_record',side_effect=lambda p,h:{'path':str(p),'sha256':h,'bytes':1}):
            result=b.old_contract(value,{'after':spec},args)
            self.assertEqual(result['cwd'],str(code));self.assertEqual(result['resources']['ram_mib'],196608)
            for key,bad in (('env',[]),('ram_mib',262144),('cwd','/foreign'),('inputs',[]),('seed',False),('pinned_files',{})):
                with self.subTest(key=key),self.assertRaises(ValueError):b.old_contract(value,{'after':{**spec,key:bad}},args)
            changed=copy.deepcopy(args);changed.data='/foreign'
            with self.assertRaises(ValueError):b.old_contract(value,{'after':spec},changed)

    def test_claimed_override_mismatch_blocks_before_pf_read(self):
        from analysis import sgns_full_input_bridge as b
        from analysis import sgns_gpu_full as s
        for state in ('wrong-claimed','pending','rejected'):
            with self.subTest(state=state),tempfile.TemporaryDirectory() as temp:
                root=Path(temp).resolve();q=root/'OPS/queue/pending/job.json';claimed=q.parent.parent/'claimed'/q.name
                claimed.parent.mkdir(parents=True);claimed.write_bytes(b'wrong' if state=='wrong-claimed' else b'after')
                if state!='wrong-claimed':
                    path=q if state=='pending' else q.parent.parent/'rejected'/q.name
                    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'after')
                args=SimpleNamespace(source_bridge_request=root/'request',source_bridge_sha256='a'*64,preflight_run=root/'PF')
                value={'preflight_run':str(root/'PF')};change={'plan':{'queue_path':str(q)},'raw':{'after':b'after'}}
                with patch.object(b,'read_request',return_value=(value,change,{})),patch.object(s,'last_receipt',side_effect=AssertionError('PF read')) as read:
                    with self.assertRaisesRegex(ValueError,'claimed queue'):b.preflight(args,{})
                    read.assert_not_called()

    def test_manifest_bridge_revalidates_and_rejects_false_proof(self):
        from analysis import sgns_gpu_full as s,sgns_full_input_bridge as b
        args=SimpleNamespace(**{n:'/transport/'+n for n in ('data','embedding_root','stage2','stage2_run','lock','output')},deployment_sha='a'*40,
            source_bridge_request=Path('/transport/request'),source_bridge_sha256='b'*64)
        argv=s.input_argv('prepare',args)+['--preflight-run','/transport/PF']
        proof={'preflight_job':{'transport-only':'hash'}}
        value={'prepare_argv':argv,'source_bridge':proof,'preflight_job':proof['preflight_job']}
        with patch.object(b,'preflight',return_value={}) as before,patch.object(b,'finish',return_value=proof) as after:
            self.assertEqual(b.verify_manifest(value,{'commit':'a'*40}),proof)
            before.assert_called_once();after.assert_called_once()
            for bad in ({**value,'source_bridge':{}},{**value,'preflight_job':{}}):
                with self.assertRaises(ValueError):b.verify_manifest(bad,{'commit':'a'*40})
            for pos in (0,3,4,5,18,19,21,23):
                bad=copy.deepcopy(value);bad['prepare_argv'][pos]='foreign'
                with self.subTest(pos=pos),self.assertRaises(ValueError):b.verify_manifest(bad,{'commit':'a'*40})

    def test_subprocess_dry_and_invalid(self):
        run=subprocess.run([sys.executable,'-B','-m','analysis.sgns_full_input_bridge','dry'],capture_output=True,text=True)
        self.assertEqual(run.returncode,0,run.stderr);self.assertFalse(json.loads(run.stdout)['queue_modified'])
        bad=subprocess.run([sys.executable,'-B','-m','analysis.sgns_full_input_bridge','invalid'],capture_output=True)
        self.assertNotEqual(bad.returncode,0)


if __name__=='__main__':unittest.main()
