"""New-family input semantics; original subset oracle is never relabelled."""
import copy
import unittest
from vkgr.data_audit import canonical_bytes

class NativePayloadTests(unittest.TestCase):
    def fixture(self):
        rows=[{'user_id':1,'item_id':item,'like':like} for item,like in [(3,True),(4,False),(5,True),(3,False),(4,True)]]
        parts=[(week,copy.deepcopy(rows) if week==0 else []) for week in range(25)]
        validation=(25,[{'user_id':1,'item_id':5,'like':True},{'user_id':1,'item_id':3,'like':True}])
        return parts,validation
    def test_original_payload_oracle_exact_types_order_and_statistics(self):
        from vkgr import sgns_subset_a1_inputs as inputs
        from vkgr.retrieval_data import prefix_examples
        from vkgr.data_audit import approved_policy,build_queries
        from vkgr.history_v2 import build_observed,overlay_examples
        from analysis.ur001_derivation import derive
        parts,val=self.fixture(); emitted=[]
        ur001=derive(parts,val,emitted.append)
        actual=inputs.native_payloads(parts,val,ur001_train=emitted,ur001_validation=ur001['validation'],ur001_slices=ur001['slices'])
        original=list(prefix_examples(parts)); queries=build_queries(parts,val,approved_policy(),set(ur001['catalog']))['queries']
        observed=build_observed(parts,original,queries,ur001['catalog'],max_len=50)
        self.assertEqual(canonical_bytes(actual['like_examples']),canonical_bytes(original))
        self.assertEqual(canonical_bytes(actual['like_queries']),canonical_bytes(queries))
        self.assertEqual(canonical_bytes(actual['observed_examples']),canonical_bytes(overlay_examples(original,observed['train'],50,set(ur001['catalog']))))
        self.assertEqual(actual['observed_examples'][0]['history'],[3,4])
        self.assertEqual(actual['like_examples'][0]['history'],[3])
        self.assertEqual(actual['slices']['item_like_counts'],{'3':1,'4':1,'5':1})
        self.assertIs(type(actual['slices']['item_like_counts']['3']),int)
        self.assertEqual(actual['observed_queries'][0]['target_position'],[25,0])
        self.assertTrue(all(position[0]<25 for position in actual['observed_queries'][0]['history_positions']))
        self.assertEqual(actual['catalog'],[3,4,5])
    def test_oracle_corruption_and_non_train_partitions_fail(self):
        from vkgr import sgns_subset_a1_inputs as inputs
        from analysis.ur001_derivation import derive
        parts,val=self.fixture();train=[];result=derive(parts,val,train.append)
        for kind in ('train','validation','slices'):
            bad=copy.deepcopy({'train':train,'validation':result['validation'],'slices':result['slices']})
            if kind=='train':bad[kind][0]['history'].reverse()
            elif kind=='validation':bad[kind][0]['target_position']=[25,1]
            else:bad[kind]['item_like_counts']['3']=True
            with self.assertRaisesRegex(ValueError,'scientific payload differs'):
                inputs.native_payloads(parts,val,ur001_train=bad['train'],ur001_validation=bad['validation'],ur001_slices=bad['slices'])
        for badparts,badval in ((parts[:-1],val),(parts,(26,val[1])),(parts+[(25,[])],val)):
            with self.assertRaisesRegex(ValueError,'25 train weeks|validation week25'):
                inputs.native_payloads(badparts,badval,ur001_train=train,ur001_validation=result['validation'],ur001_slices=result['slices'])


class FamilyBindingTests(unittest.TestCase):
    def test_new_manifest_cannot_claim_legacy_or_test_or_different_inputs(self):
        from vkgr import sgns_subset_a1_inputs as inputs
        value={'schema':inputs.SCHEMA,'authority':'ad1c428','subset_only':True,'engineering':False,'synthetic':False,
               'test_read':False,'base_codec_policy':'same_verified_gpu_sid_as_final_codec',
               'gpu_sid_manifest_sha256':'1'*64,'content':{'path':'/tmp/content.npy','sha256':'2'*64},
               'history':{'path':'/tmp/history/manifest.json','sha256':'3'*64},'ur001':{'path':'/tmp/UR001-manifest.json','sha256':inputs.SUBSET_MANIFEST_SHA},
               'raw_data':'/tmp/raw','raw_records':[],'source':{},'runtime':{},'artifacts':{},'semantic':{},'statistics_binding':{}}
        self.assertEqual(inputs.family_manifest(value,value['history'],value['content'],'1'*64),value)
        for key,bad in [('synthetic',True),('test_read',True),('subset_only',False),('schema','G3'),('base_codec_policy','old_accepted_G1')]:
            copyv=copy.deepcopy(value);copyv[key]=bad
            with self.assertRaisesRegex(ValueError,'new-family'):inputs.family_manifest(copyv,value['history'],value['content'],'1'*64)
        for history,content,sid in (({**value['history'],'sha256':'4'*64},value['content'],'1'*64),(value['history'],{**value['content'],'sha256':'4'*64},'1'*64),(value['history'],value['content'],'4'*64)):
            with self.assertRaisesRegex(ValueError,'new-family'):inputs.family_manifest(value,history,content,sid)
    def test_injected_base_loader_preserves_native_code(self):
        from vkgr import sgns_subset_a1 as a1
        from vkgr import sid_structure_a1_run as original
        sentinel=lambda *a:None
        engine=a1.native_functions(sentinel,sentinel,sentinel,base_loader=sentinel)
        self.assertIs(engine['_contexts'].__code__,original._contexts.__code__)
        self.assertIs(engine['_contexts'].__globals__['gr'].load_context,sentinel)
        self.assertIs(engine['run'].__globals__['_contexts'],engine['_contexts'])

class FamilyRoundTripTests(unittest.TestCase):
    def test_native_statistics_context_roundtrip_and_tampering(self):
        import tempfile,json,hashlib
        from pathlib import Path
        from unittest.mock import patch
        import numpy as np
        from vkgr import sgns_subset_a1_inputs as inputs,generative_data as gd
        from vkgr.sid import SIDCatalog
        from analysis.ur001_derivation import derive
        parts,val=NativePayloadTests().fixture();train=[];ur=derive(parts,val,train.append)
        payload=inputs.native_payloads(parts,val,ur001_train=train,ur001_validation=ur['validation'],ur001_slices=ur['slices'])
        raw=[{'path':f'train/week_{i:02d}.parquet','sha256':'1'*64,'size':1} for i in range(25)]
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve();content=root/'content.npy';np.save(content,np.zeros((4,64),dtype=np.float32))
            crecord={'path':str(content),'sha256':hashlib.sha256(content.read_bytes()).hexdigest()}
            urecord={'path':str(root/'UR001-manifest.json'),'sha256':inputs.SUBSET_MANIFEST_SHA}
            output=inputs.write_family(root/'family',payload,parts,raw_data=str(root/'raw'),raw_records=raw,ur001=urecord,content=crecord,
                                      sid_sha='2'*64,source={'fixture':'transport_only_not_real'},runtime={'fixture':'synthetic'})
            codec=gd._ValidatedCodec(SIDCatalog(np.array([3,4,5],dtype=np.uint32),np.array([[0,0,0,0],[0,1,0,0],[1,0,0,0]],dtype=np.uint32),2,3,[]))
            with patch.object(inputs,'_raw_payloads',return_value=(payload,parts,{'files':raw},{})):
                context=inputs.load_context(root,output['path'],output['sha256'],root/'temporary',codec=codec,sid_sha='2'*64,source={'fixture':'transport_only_not_real'})
                self.assertEqual(dict(context.slice_definition.item_counts),{3:1,4:1,5:1})
                self.assertEqual(context.slice_definition.thresholds,(1,1))
                self.assertIs(type(context.slice_definition.thresholds[0]),int)
                self.assertEqual(context.examples.rows([0]),payload['like_examples'][:1])
                self.assertEqual(context.queries,payload['like_queries'])
                self.assertFalse(context.binding['synthetic']);self.assertFalse(context.binding['engineering'])
                self.assertEqual(context.binding['base_codec_policy'],inputs.POLICY)
                with self.assertRaisesRegex(ValueError,'new-family source differs'):
                    inputs.load_context(root,output['path'],output['sha256'],root/'wrong-source',codec=codec,sid_sha='2'*64,source={'fixture':'wrong'})
                # Mutate a consumed examples byte without changing its pinned manifest.
                p=Path(output['path']).parent/'like-examples.jsonl';p.write_bytes(p.read_bytes()+b' ')
                with self.assertRaisesRegex(ValueError,'consumed artifact bytes differ'):
                    inputs.load_context(root,output['path'],output['sha256'],root/'temp2',codec=codec,sid_sha='2'*64,source={'fixture':'transport_only_not_real'})
    def test_input_module_cli_dry_and_invalid(self):
        import subprocess,sys,os,json
        from pathlib import Path
        root=Path(__file__).resolve().parents[1]
        env={**os.environ,'PYTHONPATH':str(root/'src')}
        for command in ('dry','invalid'):
            p=subprocess.run([sys.executable,'-B','-m','analysis.sgns_subset_a1_inputs',command],cwd=root,env=env,capture_output=True,text=True)
            if command=='dry':self.assertEqual(p.returncode,0,p.stderr);self.assertTrue(json.loads(p.stdout)['subset_only'])
            else:self.assertNotEqual(p.returncode,0)


class InputProducerTests(unittest.TestCase):
    def test_original_input_job_terminal_stdout_and_output_binding(self):
        import tempfile,json,hashlib
        from pathlib import Path
        from vkgr import sgns_subset_a1_inputs as inputs
        with tempfile.TemporaryDirectory() as td:
            base=Path(td).resolve();run_id='sgns-gpu-subset-a1-inputs-v1';out=base/'family';out.mkdir();ops=base/run_id;ops.mkdir()
            def write(path,value):
                raw=canonical_bytes(value);path.write_bytes(raw);return {'path':str(path),'sha256':hashlib.sha256(raw).hexdigest()}
            prepared=write(out/'manifest.json',{'x':1});history={'path':str(out/'history/manifest.json'),'sha256':'1'*64}
            request=write(base/'request.json',{'x':2});source={'commit':'a'*40}
            run={'run_id':run_id,'kind':'ops','seed':42,'git':{'commit':'a'*40,'dirty':False,'dirty_files':[]},
                 'argv':['python','-B','-m','analysis.sgns_subset_a1_inputs','derive','--request',request['path'],'--request-sha256',request['sha256']],
                 'config':request,'inputs':[request]}
            record={'run':write(ops/'run.json',run),'terminal':write(ops/'terminal.json',{'run_id':run_id,'status':'succeeded','exit_code':0,'signal':None})}
            receipt={'schema':'gpu-sgns-subset-a1-input-stage-v1','authority':'ad1c428','status':'verified','verified':True,'run_id':run_id,
                     'run_manifest_sha256':record['run']['sha256'],'request_sha256':request['sha256'],'source':source,'output':{**prepared,'history':history,'subset_four_gate_acceptance':False},
                     'science_training':False,'test_read':False,'subset_four_gate_acceptance':False}
            record['receipt']=write(out/'stage-receipt.json',receipt);record['stdout']=write(ops/'stdout.log',receipt)
            self.assertEqual(inputs.input_producer(record,prepared,history,source),receipt)
            record['terminal']=write(ops/'terminal.json',{'run_id':run_id,'status':'succeeded','exit_code':0,'signal':9})
            with self.assertRaisesRegex(ValueError,'original input producer'):
                inputs.input_producer(record,prepared,history,source)

if __name__=='__main__':unittest.main()
