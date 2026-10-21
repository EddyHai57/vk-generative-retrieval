import unittest
from pathlib import Path
from analysis import sgns_subset_ops as subject

class FixedJobs(unittest.TestCase):
    def test_fixed_nontraining_argv(self):
        argv=subject.worker_argv('input-preflight', Path('/workspace/raw'),Path('/workspace/embed'),
                                 Path('/workspace/lock'),Path('/workspace/bundle'),Path('/workspace/output'))
        self.assertEqual(argv[3:5],['analysis.sgns_gpu_subset','preflight'])
        self.assertNotIn('fit',argv)
        self.assertEqual(argv[argv.index('--output')+1],'/workspace/bundle')

    def test_only_device_differs_for_fit(self):
        kwargs=dict(data=Path('/workspace/raw'),embedding=Path('/workspace/embed'),
                    lock=Path('/workspace/lock'),bundle=Path('/workspace/bundle'),
                    output=Path('/workspace/output'),bundle_sha='1'*64)
        cpu=subject.worker_argv('cpu-fit',**kwargs)
        cuda=subject.worker_argv('cuda-fit',**kwargs)
        index=cpu.index('--device')+1
        self.assertEqual(cpu[index],'cpu'); self.assertEqual(cuda[index],'cuda')
        cuda[index]='cpu'; self.assertEqual(cpu,cuda)
        self.assertNotIn('--seed',cpu); self.assertNotIn('--config',cpu)

    def test_reject_open_stage(self):
        with self.assertRaises(ValueError):
            subject.worker_argv('arbitrary',*[Path('/workspace/x')]*5)


class ProducerBinding(unittest.TestCase):
    def test_original_stdout_terminal_and_bundle_are_required(self):
        import json
        import tempfile
        from types import SimpleNamespace
        from unittest.mock import patch
        import hashlib
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();run=root/'sgns-gpu-subset-input-preflight-v1'
            (run/'logs').mkdir(parents=True)
            args=SimpleNamespace(stage='input-prepare',predecessor_run=run,data=root/'raw',
                embedding_root=root/'emb',lock=root/'lock',bundle=root/'bundle',output=root/'out',bundle_sha=None)
            source={'commit':'f'*40}
            receipt={'status':'verified','records':subject.fixed_records(),'source':source,
                     'engineering':False,'training':False,'payload_written':False}
            job={'run_id':run.name,'kind':'ops','git':source,
                 'argv':subject.worker_argv('input-preflight',args.data,args.embedding_root,args.lock,args.bundle,args.output)}
            terminal={'run_id':run.name,'status':'succeeded','exit_code':0}
            (run/'run.json').write_text(json.dumps(job));(run/'terminal.json').write_text(json.dumps(terminal))
            (run/'logs/stdout.log').write_text(json.dumps(receipt)+'\n')
            self.assertEqual(len(subject.predecessor(args,source)),3)
            for key,bad in [('status','running'),('run_id','other'),('exit_code',1)]:
                changed={**terminal,key:bad};(run/'terminal.json').write_text(json.dumps(changed))
                with self.assertRaises(ValueError):subject.predecessor(args,source)
                (run/'terminal.json').write_text(json.dumps(terminal))
            (run/'logs/stdout.log').write_text(json.dumps({**receipt,'source':{'commit':'0'*40}}))
            with self.assertRaises(ValueError):subject.predecessor(args,source)

class PreparedProducerBinding(unittest.TestCase):
    def test_argv_and_actual_manifest_equal_original_stdout(self):
        import json
        import tempfile
        from types import SimpleNamespace
        import hashlib
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();run=root/'sgns-gpu-subset-input-prepare-v1'
            (run/'logs').mkdir(parents=True);bundle=root/'bundle';bundle.mkdir()
            args=SimpleNamespace(stage='cpu-fit',predecessor_run=run,data=root/'raw',
                embedding_root=root/'emb',lock=root/'lock',bundle=bundle,output=root/'out',bundle_sha=None)
            source={'commit':'f'*40}
            receipt={'status':'prepared','records':subject.fixed_records(),'source':source,
                     'engineering':False,'training':False,'payload_written':True}
            manifest=bundle/'manifest.json';manifest.write_text(json.dumps(receipt))
            args.bundle_sha=hashlib.sha256(manifest.read_bytes()).hexdigest()
            job={'run_id':run.name,'kind':'ops','git':source,
                 'argv':subject.worker_argv('input-prepare',args.data,args.embedding_root,args.lock,bundle,args.output)}
            terminal={'run_id':run.name,'status':'succeeded','exit_code':0}
            (run/'run.json').write_text(json.dumps(job));(run/'terminal.json').write_text(json.dumps(terminal))
            (run/'logs/stdout.log').write_text(json.dumps(receipt)+'\n')
            self.assertEqual(len(subject.predecessor(args,source)),3)
            (run/'run.json').write_text(json.dumps({**job,'argv':job['argv']+['--unexpected']}))
            with self.assertRaises(ValueError):subject.predecessor(args,source)
            (run/'run.json').write_text(json.dumps(job))
            manifest.write_text(json.dumps({**receipt,'events':999}))
            args.bundle_sha=hashlib.sha256(manifest.read_bytes()).hexdigest()
            with self.assertRaises(ValueError):subject.predecessor(args,source)

if __name__=='__main__':unittest.main()
