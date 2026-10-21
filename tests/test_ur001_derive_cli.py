import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'src'))


def module():
    from analysis import ur001_derive_cli
    return ur001_derive_cli


class FormalDerivationTests(unittest.TestCase):
    def fixture(self, root):
        import pyarrow as pa
        import pyarrow.parquet as pq
        data=root/'data';records=[]
        for week in range(26):
            rel=f"{'train' if week<25 else 'validation'}/week_{week:02d}.parquet"
            p=data/rel;p.parent.mkdir(parents=True,exist_ok=True)
            rows=[{'user_id':1,'item_id':week+1,'like':True},
                  {'user_id':1,'item_id':9,'like':False}]
            if week==25:rows += [{'user_id':2,'item_id':90,'like':True}]
            pq.write_table(pa.Table.from_pylist(rows,schema=pa.schema([('user_id',pa.uint32()),('item_id',pa.uint32()),('like',pa.bool_())])),p,row_group_size=1)
            records.append({'path':rel,'size':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
        return data,records

    def test_formal_subprocess_dry_nonempty_and_invalid_nonzero(self):
        for argv,want in [(['dry'],0),(['illegal'],2)]:
            r=subprocess.run([sys.executable,'-B','-m','analysis.ur001_derive_cli',*argv],cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(r.returncode,want,r.stderr)
            if want==0:self.assertFalse(json.loads(r.stdout)['science_training'])

    def test_raw_gate_exact_weeks_hash_schema_and_no_test(self):
        m=module()
        with tempfile.TemporaryDirectory() as t:
            data,records=self.fixture(Path(t).resolve())
            got=m.inspect_inputs(data,records,workers=2)
            self.assertEqual(got['train_rows'],50)
            self.assertEqual(got['validation_rows'],3)
            bad=[dict(r) for r in records];bad[0]['sha256']='0'*64
            with self.assertRaises(ValueError):m.inspect_inputs(data,bad,workers=1)
            (data/'train/week_00-extra.parquet').write_bytes((data/'train/week_00.parquet').read_bytes())
            with self.assertRaises(ValueError):m.inspect_inputs(data,records,workers=1)

    def test_artifacts_dag_oracle_and_new_only(self):
        m=module()
        with tempfile.TemporaryDirectory() as t:
            root=Path(t).resolve();data,records=self.fixture(root);out=root/'UR001-output'
            inp=m.inspect_inputs(data,records,workers=1)
            result=m.write_derivation(data,out,inp,workers=2,guard=lambda:None,source={'commit':'fixture','files':{}},runtime={'fixture':True},oracle=True)
            self.assertEqual(result['counts']['train_examples'],24)
            self.assertEqual(result['counts']['eligible'],1)
            self.assertTrue(result['oracle']['bitwise_equal'])
            dag=json.loads((out/'UR001-manifest.json').read_text())
            self.assertEqual(dag['upstream']['UR001-input-manifest.json'],hashlib.sha256((out/'UR001-input-manifest.json').read_bytes()).hexdigest())
            for rel,entry in dag['artifacts'].items():
                self.assertTrue(rel.startswith('UR001-'))
                self.assertEqual(entry['sha256'],hashlib.sha256((out/rel).read_bytes()).hexdigest())
            with self.assertRaises(FileExistsError):m.write_derivation(data,out,inp,workers=1,guard=lambda:None,source={'commit':'fixture'},runtime={'fixture':True},oracle=False)

    def test_resource_guard_and_progress_called(self):
        m=module()
        calls=[]
        rows=list(m.observed_rows([{'x':i} for i in range(9)],lambda:calls.append(1),interval=3))
        self.assertEqual(len(rows),9);self.assertGreaterEqual(len(calls),3)
        def stop():raise MemoryError('guard')
        with self.assertRaises(MemoryError):list(m.observed_rows([{}],stop,interval=1))

    def test_dag_verifier_rejects_changed_upstream_and_artifact(self):
        m=module()
        with tempfile.TemporaryDirectory() as t:
            root=Path(t).resolve();data,records=self.fixture(root);out=root/'UR001-output'
            m.write_derivation(data,out,m.inspect_inputs(data,records,1),workers=1,guard=lambda:None,source={'commit':'fixture'},runtime={'fixture':True},oracle=True)
            self.assertTrue(m.verify_dag(out)['verified'])
            p=out/'UR001-validation.jsonl';p.write_text('{}\n')
            with self.assertRaises(ValueError):m.verify_dag(out)

    def test_formal_preflight_original_producer_binding(self):
        from types import SimpleNamespace
        m=module()
        with tempfile.TemporaryDirectory() as t:
            root=Path(t).resolve();run=root/'pf-run';(run/'logs').mkdir(parents=True)
            args=SimpleNamespace(data=root/'data',output=root/'UR001-output',profile='subset',aggregate=root/'aggregate',lock=root/'lock',workers=2)
            inputs={'source':{'commit':'frozen'},'files':[]}
            argv=[sys.executable,'-B','-m','analysis.ur001_derive_cli','preflight','--data',str(args.data),'--output',str(args.output),'--profile','subset','--aggregate',str(args.aggregate),'--lock',str(args.lock),'--workers','2','--execute-on-new-instance']
            r={'run_id':'pf-run','kind':'ops','git':{'commit':'frozen'},'argv':argv}
            term={'run_id':'pf-run','status':'succeeded','exit_code':0}
            receipt={'verified':True,'status':'read_only_preflight','engineering':False,'payload_written':False,'inputs':inputs}
            def save():
                (run/'run.json').write_text(json.dumps(r));(run/'terminal.json').write_text(json.dumps(term));(run/'logs/stdout.log').write_text(json.dumps(receipt)+'\n')
            save();self.assertEqual(len(m.check_preflight(run,inputs,args)),3)
            term['status']='wall_cap_exceeded';save()
            with self.assertRaises(ValueError):m.check_preflight(run,inputs,args)
            term['status']='succeeded';r['argv'][-2]='1';save()
            with self.assertRaises(ValueError):m.check_preflight(run,inputs,args)

    def test_dag_semantics_negative_cases(self):
        m=module()
        with tempfile.TemporaryDirectory() as t:
            root=Path(t).resolve();data,records=self.fixture(root);out=root/'UR001-output'
            m.write_derivation(data,out,m.inspect_inputs(data,records,1),workers=1,guard=lambda:None,source={'commit':'fixture'},runtime={'fixture':True},oracle=True)
            p=out/'UR001-manifest.json';original=json.loads(p.read_text())
            for field,value in [('counts',{'train_examples':-123}),('revision','wrong'),('history_schema','wrong'),('science_training',True),('source',{}),('runtime',{}),('upstream',{})]:
                changed=dict(original);changed[field]=value;p.write_text(json.dumps(changed))
                with self.subTest(field=field),self.assertRaises(ValueError):m.verify_dag(out)
            changed=dict(original);changed['oracle']=dict(changed['oracle'],train_sha256='0'*64);p.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):m.verify_dag(out)
            p.write_text(json.dumps(original))
            with self.assertRaises(ValueError):m.verify_dag(out,require_profile='subset',source={'commit':'fixture'},runtime={'fixture':True})

    def test_final_manifest_absent_when_postread_check_fails(self):
        m=module()
        with tempfile.TemporaryDirectory() as t:
            root=Path(t).resolve();data,records=self.fixture(root);out=root/'UR001-output'
            def fail(counts):raise ValueError('postread raw drift')
            with self.assertRaises(ValueError):
                m.write_derivation(data,out,m.inspect_inputs(data,records,1),workers=1,guard=lambda:None,source={'commit':'fixture'},runtime={'fixture':True},oracle=True,final_check=fail)
            self.assertFalse((out/'UR001-manifest.json').exists())
            self.assertTrue((out/'UR001-train.jsonl').exists())


if __name__=='__main__':unittest.main()
