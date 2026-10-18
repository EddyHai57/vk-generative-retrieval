"""Readonly CLI and original job/manifest binding; no formal data is faked."""
import copy,contextlib,hashlib,importlib,io,json,os,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from vkgr import sid_learn_input as inp
from vkgr.data_audit import canonical_bytes,file_sha256
import test_sid_learn_input as input_tests
ROOT=Path(__file__).resolve().parents[1]
class LearnRunTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((ROOT/'src/vkgr/sid_learn_run.py').is_file(),'missing job entry');self.mod=importlib.import_module('vkgr.sid_learn_run')
    def args(self,r,command='preflight'):
        helper=input_tests.LearnInputTests();helper.mod=inp;refs,histories,physical=helper.build(r)
        spec=Path(r).resolve()/'inputs.json';spec.write_bytes(canonical_bytes({'schema':'sid-learn-input-spec-v1','refs':refs}))
        output=Path(r).resolve()/'out';cfg=ROOT/'configs/sid_learn/fit-s42.json'
        args=[command,'--config',str(cfg),'--config-sha',file_sha256(cfg),'--inputs',str(spec),'--inputs-sha',file_sha256(spec),'--output',str(output)]
        return args,histories,physical,output
    def test_engineering_preflight_runs_real_helpers_readonly_and_is_marked(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            args,histories,physical,output=self.args(d);stream=io.StringIO()
            with patch.object(inp.legacy,'load_histories',return_value=(histories,physical)),contextlib.redirect_stdout(stream):code=self.mod.main(args+['--engineering'])
            result=json.loads(stream.getvalue());self.assertEqual(code,0);self.assertEqual(result['status'],'preflight_ok');self.assertIs(result['engineering'],True);self.assertFalse(result['written']);self.assertFalse(output.exists());self.assertIn('producer',result)
    def test_formal_cli_without_owned_job_stops_before_inputs(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            args,_,_,_=self.args(d)
            with patch.object(self.mod,'source_identity',return_value={'commit':'a'*40,'dirty':False,'producer':{},'producer_runtime':{}}),patch.dict(os.environ,{},clear=True),patch.object(inp,'load_inputs',side_effect=AssertionError('formal input opened')) as read,contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(self.mod.main(args),1)
            read.assert_not_called()
    def metadata(self,r):
        rid='sid-learn-fit-verify-v1';run=Path(r).resolve()/rid;run.mkdir();out=run/'out';cfg=ROOT/'configs/sid_learn/fit-s42.json';spec=Path(r).resolve()/'input.json';spec.write_text('{}');tail=['verify','--output',str(out)]
        meta={'run_id':rid,'git':{'commit':'a'*40,'dirty':False},'cwd':str(ROOT),'config':{'path':str(cfg),'sha256':file_sha256(cfg)},'inputs':[{'path':str(spec),'sha256':file_sha256(spec)}],'argv':[sys.executable,'-B','-m','vkgr.sid_learn_run',*tail]}
        (run/'run.json').write_bytes(canonical_bytes(meta));return run,out,cfg,spec,tail,meta
    def test_verified_binding_uses_actual_own_manifest(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            run,out,cfg,spec,tail,meta=self.metadata(d)
            with patch.dict(os.environ,VKGR_RUN_ID=run.name):
                result=self.mod.bind_job('verify',out,file_sha256(cfg),spec,file_sha256(spec),tail,{'commit':'a'*40},False)
            self.assertEqual(result['run_id'],run.name);self.assertEqual(result['run_manifest_sha256'],file_sha256(run/'run.json'))
    def test_original_fit_terminal_claim_and_stdout_are_bound_before_replay(self):
        from types import SimpleNamespace
        for mode in ('valid','dirty','failed','stdout','duplicate'):
            with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
                root=Path(d).resolve();runs=root/'runs';fitrun=runs/'sid-learn-fit-v1';fitrun.mkdir(parents=True);(fitrun/'logs').mkdir()
                for state in ('pending','claimed','rejected'):(root/'queue'/state).mkdir(parents=True)
                output=runs/'sid-learn-fit-verify-v1/out';cfg=ROOT/'configs/sid_learn/fit-s42.json'
                meta={'run_id':fitrun.name,'git':{'commit':'a'*40,'dirty':mode=='dirty'},'config':{'sha256':file_sha256(cfg)},'cwd':str(ROOT),'argv':[sys.executable,'-B','-m','vkgr.sid_learn_run','fit']}
                (fitrun/'run.json').write_bytes(canonical_bytes(meta))
                term={'run_id':fitrun.name,'status':'failed' if mode=='failed' else 'succeeded','exit_code':1 if mode=='failed' else 0,'wall_cap_exceeded':False};(fitrun/'terminal.json').write_bytes(canonical_bytes(term))
                claimed={k:meta[k] for k in ('run_id','argv','cwd')};(root/'queue/claimed/001-sid-learn-fit-v1.json').write_bytes(canonical_bytes(claimed))
                if mode=='duplicate':(root/'queue/claimed/002-sid-learn-fit-v1.json').write_bytes(canonical_bytes(claimed))
                args=SimpleNamespace(producer_run=str(fitrun/'run.json'),producer_run_sha=file_sha256(fitrun/'run.json'),bundle=str(fitrun/'out/payload/run_manifest.json'),sha='b'*64,inputs_sha='c'*64)
                raw={'status':'complete','engineering':False,'run_id':fitrun.name,'run_manifest_sha256':args.producer_run_sha,'manifest_sha256':'0'*64 if mode=='stdout' else args.sha,'fit_config_file_sha256':file_sha256(cfg),'input_spec_file_sha256':args.inputs_sha,'source_commit':'a'*40}
                (fitrun/'logs/stdout.log').write_text(json.dumps(raw)+'\n');loaded=SimpleNamespace(manifest={'source_commit':'a'*40})
                with self.subTest(mode=mode):
                    if mode=='valid':
                        proof=self.mod._producer_binding(args,output,loaded,file_sha256(cfg),False)
                        self.assertEqual(proof.get('producer_terminal_sha256'),file_sha256(fitrun/'terminal.json'))
                    else:
                        with self.assertRaises(ValueError):self.mod._producer_binding(args,output,loaded,file_sha256(cfg),False)

    def test_full_replay_obeys_own_resource_guard_before_updates(self):
        from test_sid_learn_bundle import fixture
        from types import SimpleNamespace
        from vkgr import sid_learn_bundle as bundle
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            root=Path(d).resolve();data=fixture();bundle.write_bundle(root/'bundle',**data)
            def stop():raise RuntimeError('own resource headroom')
            with self.assertRaisesRegex(RuntimeError,'own resource headroom'):
                self.mod.verify_replay(None,root/'bundle/run_manifest.json',{'content':data['content'],'target':data['content'][:,:32].copy(),'fallback':data['fallback']},data['config'],device='cpu',engineering=True,guard=stop)

    def test_wrong_job_source_config_argv_or_cwd_are_rejected(self):
        for field in ('run_id','git','config','argv','cwd','inputs'):
            with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
                run,out,cfg,spec,tail,meta=self.metadata(d)
                if field=='run_id':meta[field]='other'
                if field=='git':meta[field]['commit']='b'*40
                if field=='config':meta[field]['sha256']='0'*64
                if field=='argv':meta[field]=[sys.executable,'-c','pass',*tail]
                if field=='cwd':meta[field]=str(Path(d).resolve())
                if field=='inputs':meta[field][0]['sha256']='0'*64
                (run/'run.json').write_bytes(canonical_bytes(meta))
                with patch.dict(os.environ,VKGR_RUN_ID=run.name),self.subTest(field=field),self.assertRaises(ValueError):self.mod.bind_job('verify',out,file_sha256(cfg),spec,file_sha256(spec),tail,{'commit':'a'*40},False)
if __name__=='__main__':unittest.main()
