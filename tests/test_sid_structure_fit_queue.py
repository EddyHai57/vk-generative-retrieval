"""Metadata-only publication of the three fixed structural fit stages."""
import importlib.util,json,tempfile,unittest,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def module():
    spec=importlib.util.spec_from_file_location('structure_fit_queue',ROOT/'scripts/sid_structure_fit_queue.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
class FitQueueTests(unittest.TestCase):
    def request(self,r):
        for state in ('pending','claimed','rejected'):(r/'queue'/state).mkdir(parents=True)
        (r/'ops').mkdir();(r/'runs').mkdir()
        return {'root':str(r),'ops':str(r/'ops'),'job':{'run_id':'sid-structure-fit-preflight-v1','argv':[sys.executable,'-B','-m','vkgr.sid_structure_run','preflight'],'cpu_cores':1,'vram_mib':256,'no_admission':False}}
    def test_unknown_run_stage_engineering_and_duplicate_are_rejected(self):
        m=module()
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);req=self.request(r);m.validate(req)
            for key,value in [('run_id','other'),('argv',['python','-m','vkgr.sid_structure_run','fit','--engineering']),('cpu_cores',2)]:
                bad={**req,'job':{**req['job'],key:value}}
                with self.subTest(key=key),self.assertRaises(ValueError):m.validate(bad)
            (r/'queue/rejected/001-sid-structure-fit-preflight-v1.json').write_text('{}')
            with self.assertRaises(FileExistsError):m.check_fresh(req)
    def test_python_c_cannot_hide_a_formal_stage_in_trailing_arguments(self):
        m=module()
        with tempfile.TemporaryDirectory() as d:
            req=self.request(Path(d))
            for prefix in ([sys.executable,'-c',"print('wrong')"],[sys.executable,'-I','-c',"print('wrong')"],['/bin/echo','-B']):
                req['job']['argv']=[*prefix,'-m','vkgr.sid_structure_run','preflight']
                with self.subTest(prefix=prefix),self.assertRaises(ValueError):m.validate(req)

    def test_intent_prevents_retry_even_if_publication_crashes(self):
        m=module()
        with tempfile.TemporaryDirectory() as d:
            req=self.request(Path(d))
            def failed(*args):raise OSError('transport interrupted')
            with self.assertRaises(OSError):m.submit(req,prepare=lambda req,job:{},publisher=failed)
            self.assertTrue((Path(req['ops'])/'intent-sid-structure-fit-preflight-v1.json').is_file())
            with self.assertRaises(FileExistsError):m.submit(req,prepare=lambda req,job:{},publisher=failed)
if __name__=='__main__':unittest.main()
