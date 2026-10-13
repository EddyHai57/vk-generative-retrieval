"""Independent G2 CE, sample coverage, accumulation and resume oracles."""
import copy, math, tempfile, unittest
from pathlib import Path
import numpy as np
import torch
from vkgr.sid import SIDCatalog
from vkgr.tiger import SIDTokens
try:
    from vkgr import generative_training as gt
except ImportError:
    gt=None

def codec():
    return SIDTokens(SIDCatalog(np.array([10,20,30,40],dtype=np.uint32),np.array([[0,0,0,0],[0,0,0,1],[1,0,0,0],[1,1,0,0]],dtype=np.uint32),2,3,[]))
def config():
    return dict(d_model=64,heads=4,encoder_layers=1,decoder_layers=1,ff=256,dropout=0.,precision='fp32',learning_rate=.001,weight_decay=.0001,gradient_clip_norm=1.,effective_batch=512,seed=42)
class LossTests(unittest.TestCase):
    def setUp(self):self.assertIsNotNone(gt,'G2 training module missing')
    def test_ce_hand_oracle_full_denominator_includes_pad_bos(self):
        logits=torch.tensor([[[4.,0.,1.,2.,3.]]*4,[[0.,1.,2.,3.,4.]]*4],dtype=torch.float64,requires_grad=True)
        labels=torch.tensor([[2,3,4,2],[4,3,2,4]])
        expected=sum(math.log(sum(math.exp(float(x)) for x in row))-float(row[label]) for rows,ys in zip(logits.detach(),labels) for row,label in zip(rows,ys))/8
        loss=gt.full_vocab_ce(logits,labels);self.assertAlmostEqual(float(loss.detach()),expected,places=12)
        loss.backward();self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad[0,0,0]),0) # PAD is in denominator, never masked.
    def test_ce_shape_label_and_nonfinite_reject(self):
        for x,y in [(torch.zeros(1,3,5),torch.zeros(1,3,dtype=torch.long)),(torch.zeros(1,4,5),torch.zeros(1,4)),(torch.zeros(1,4,5),torch.full((1,4),5)),(torch.full((1,4,5),float('nan')),torch.zeros(1,4,dtype=torch.long))]:
            with self.assertRaises(ValueError):gt.full_vocab_ce(x,y)
    def test_real_tiger_adamw_updates_and_fixed_core_config(self):
        torch.manual_seed(42);c=codec();m=gt.make_model(c,config(),device='cpu');o=gt.make_optimizer(m,config())
        before=[p.detach().clone() for p in m.parameters()]
        result=gt.train_effective_batch(m,o,[{'history':[10,20],'target':30},{'history':[30],'target':40}],c,microbatch=1,device='cpu')
        self.assertEqual((result.examples,result.microbatch_sizes,result.optimizer_steps),(2,(1,1),1));self.assertTrue(math.isfinite(result.loss_mean))
        self.assertTrue(any(not torch.equal(a,b) for a,b in zip(before,m.parameters())))
        for field,value in [('heads',2),('dropout',.1),('weight_decay',.01),('precision','fp16')]:
            bad=config();bad[field]=value
            with self.assertRaises(ValueError):gt.make_model(c,bad,device='cpu')
class CursorStoreTests(unittest.TestCase):
    def test_epoch_no_replacement_tail_and_explicit_boundary(self):
        self.assertTrue(hasattr(gt,'EpochCursor'),'G2 EpochCursor missing')
        c=gt.EpochCursor(42,513);first=c.next_batch(512);tail=c.next_batch(512)
        self.assertEqual(len(first),512);self.assertEqual(len(tail),1);self.assertEqual(sorted(first+tail),list(range(513)));self.assertEqual(c.next_batch(512),[])
        self.assertEqual(c.phase,'evaluate')
        with self.assertRaises(ValueError):c.advance_epoch()
        c.complete_evaluation();c.advance_epoch();self.assertEqual(c.epoch,2);self.assertEqual(c.phase,'train')
        second=c.next_batch(513);self.assertEqual(sorted(second),list(range(513)));self.assertNotEqual(second,first+tail)
    def test_cursor_state_replays_exact_remaining_and_rejects_mutation(self):
        self.assertTrue(hasattr(gt,'EpochCursor'),'G2 EpochCursor missing')
        a=gt.EpochCursor(42,17);a.next_batch(5);state=a.state_dict();b=gt.EpochCursor(42,17);b.load_state_dict(state)
        self.assertEqual(a.next_batch(12),b.next_batch(12))
        for field,value in [('next_offset',True),('epoch',0),('phase','unknown'),('permutation',[0]*17),('seed',2024)]:
            bad=copy.deepcopy(state);bad[field]=value
            with self.assertRaises(ValueError):b.load_state_dict(bad)
    def test_example_store_external_pin_count_index_and_file_change(self):
        self.assertTrue(hasattr(gt,'ExampleStore'),'G2 ExampleStore missing')
        import json,hashlib
        with tempfile.TemporaryDirectory(dir="/private/tmp") as d:
            path=Path(d)/'examples.jsonl';path.write_text(''.join(json.dumps({'example_id':i,'history':[10],'target':20})+'\n' for i in range(5)))
            pin=hashlib.sha256(path.read_bytes()).hexdigest();store=gt.ExampleStore(path,pin,5)
            self.assertEqual(len(store),5);self.assertEqual([x['example_id'] for x in store.rows([4,0,2])],[4,0,2])
            with self.assertRaises(ValueError):gt.ExampleStore(path,'0'*64,5)
            with self.assertRaises(ValueError):gt.ExampleStore(path,pin,6)
            with self.assertRaises(ValueError):store.rows([-1])
            path.write_text(path.read_text()+'{}\n')
            with self.assertRaises(ValueError):store.rows([0])
class AccumulationTests(unittest.TestCase):
    def test_whole_512_and_micro_256_128_and_tail_actual_weight(self):
        self.assertIsNotNone(gt)
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__();self.emb=torch.nn.Embedding(10,8);self.output=torch.nn.Linear(8,10)
            def forward(self,source,decoder):return self.output(self.emb(decoder))
        torch.manual_seed(42);base=Toy();c=codec()
        for n,micros in [(512,[512,256,128]),(17,[17,8,2])]:
            rows=[{'history':[10,20] if i%2 else [30],'target':[10,20,30,40][i%4]} for i in range(n)];states=[]
            for micro in micros:
                m=copy.deepcopy(base);o=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001)
                result=gt.train_effective_batch(m,o,rows,c,microbatch=micro,device='cpu');self.assertEqual(result.examples,n)
                states.append((m.state_dict(),o.state_dict(),result.loss_mean))
            for state,optimizer,loss in states[1:]:
                for key,tensor in states[0][0].items():self.assertTrue(torch.allclose(tensor,state[key],atol=2e-6,rtol=2e-6),key)
                for key,values in states[0][1]['state'].items():
                    for field,tensor in values.items():self.assertTrue(torch.allclose(tensor,optimizer['state'][key][field],atol=2e-6,rtol=2e-6))
                self.assertAlmostEqual(states[0][2],loss,places=5)
def budget(steps=1):
    import datetime
    return dict(reserved_steps=steps,completed_steps=steps,consumed_wall_by_phase={'train':0.,'evaluate':0.,'verify':0.},resume_attempts=0,max_optimizer_steps=10,max_resume_attempts=1,deadline_utc=(datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(seconds=60)).isoformat(),parent_ledger_head=None)
def selection():return dict(epoch=0,best_epoch=None,best_metric=None,bad_epochs=0)
def seed_all(seed):
    import random
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
def tensor_tree_hash(value):
    import hashlib,json
    digest=hashlib.sha256()
    def visit(v):
        if isinstance(v,torch.Tensor):digest.update(str((str(v.dtype),list(v.shape))).encode());digest.update(v.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(v,np.ndarray):digest.update(v.dtype.str.encode());digest.update(v.tobytes())
        elif isinstance(v,dict):
            for k in sorted(v,key=str):digest.update(str(k).encode());visit(v[k])
        elif isinstance(v,(list,tuple)):
            for x in v:visit(x)
        else:digest.update(json.dumps(v,sort_keys=True,allow_nan=False).encode())
    visit(value);return digest.hexdigest()
class CheckpointTests(unittest.TestCase):
    def test_checkpoint_new_process_resume_all_state_matches_continuous(self):
        self.assertTrue(hasattr(gt,'save_checkpoint'),'G2 save_checkpoint missing')
        import json,subprocess,sys,os,random
        seed_all(42);c=codec();m=gt.make_model(c,config(),device='cpu');o=gt.make_optimizer(m,config());cursor=gt.EpochCursor(42,4)
        rows=[{'history':[10,20],'target':t} for t in [10,20,30,40]]
        gt.train_effective_batch(m,o,[rows[i] for i in cursor.next_batch(2)],c,microbatch=1,device='cpu')
        binding={'effective_config':config(),'fixture_sha256':'a'*64,'device':'cpu'}
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            path=Path(d)/'checkpoint.pt';sha=gt.save_checkpoint(path,m,o,cursor,binding=binding,selection=selection(),budget=budget())
            gt.train_effective_batch(m,o,[rows[i] for i in cursor.next_batch(2)],c,microbatch=1,device='cpu')
            expected={'state':tensor_tree_hash({'model':m.state_dict(),'optimizer':o.state_dict(),'cursor':cursor.state_dict()}),'draw':[random.random(),float(np.random.random()),float(torch.rand(()))]}
            output=Path(d)/'result.json'
            code="import sys,json,random;sys.path.insert(0,'tests');from test_generative_training import *;seed_all(999);c=codec();m=gt.make_model(c,config(),device='cpu');o=gt.make_optimizer(m,config());cur=gt.EpochCursor(42,4);state=gt.load_checkpoint(Path("+repr(str(path))+"),"+repr(sha)+",m,o,cur,expected_binding="+repr(binding)+");rows="+repr(rows)+";gt.train_effective_batch(m,o,[rows[i] for i in cur.next_batch(2)],c,microbatch=1,device='cpu');result={'state':tensor_tree_hash({'model':m.state_dict(),'optimizer':o.state_dict(),'cursor':cur.state_dict()}),'draw':[random.random(),float(np.random.random()),float(torch.rand(()))]};Path("+repr(str(output))+").write_text(json.dumps(result))"
            result=subprocess.run([sys.executable,'-B','-c',code],capture_output=True,text=True,env=dict(os.environ,OMP_NUM_THREADS='1',PYTHONPATH='src'))
            self.assertEqual(result.returncode,0,result.stderr);self.assertEqual(json.loads(output.read_text()),expected)
            self.assertFalse(torch.cuda.is_initialized())
    def test_checkpoint_external_hash_binding_rng_optimizer_and_budget_reject(self):
        self.assertTrue(hasattr(gt,'save_checkpoint'),'G2 save_checkpoint missing')
        import hashlib
        seed_all(42);m=gt.make_model(codec(),config(),device='cpu');o=gt.make_optimizer(m,config());cursor=gt.EpochCursor(42,4);binding={'effective_config':config(),'fixture_sha256':'a'*64,'device':'cpu'}
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            path=Path(d)/'state.pt';pin=gt.save_checkpoint(path,m,o,cursor,binding=binding,selection=selection(),budget=budget(0))
            with self.assertRaises(ValueError):gt.load_checkpoint(path,'0'*64,m,o,cursor,expected_binding=binding)
            bad=copy.deepcopy(binding);bad['fixture_sha256']='b'*64
            with self.assertRaises(ValueError):gt.load_checkpoint(path,pin,m,o,cursor,expected_binding=bad)
            original=torch.load(path,weights_only=True)
            for mode in ('rng','optimizer','budget','cursor','model'):
                value=copy.deepcopy(original)
                if mode=='rng':value['rng']['torch_cpu']=torch.zeros(3,dtype=torch.uint8)
                if mode=='optimizer':value['optimizer']['param_groups'][0]['lr']=.5
                if mode=='budget':value['budget']['completed_steps']=11
                if mode=='cursor':value['cursor']['permutation']=[0]*4
                if mode=='model':value['model'][next(iter(value['model']))].fill_(float('nan'))
                altered=Path(d)/(mode+'.pt');torch.save(value,altered);sha=hashlib.sha256(altered.read_bytes()).hexdigest();before=tensor_tree_hash(m.state_dict())
                with self.assertRaises(ValueError):gt.load_checkpoint(altered,sha,m,o,cursor,expected_binding=binding)
                self.assertEqual(tensor_tree_hash(m.state_dict()),before)
            with self.assertRaises(FileExistsError):gt.save_checkpoint(path,m,o,cursor,binding=binding,selection=selection(),budget=budget(0))
class StrictOptimizerTests(unittest.TestCase):
    def test_missing_moments_wrong_step_and_negative_variance_reject(self):
        import hashlib
        seed_all(42);c=codec();cfg=config();m=gt.make_model(c,cfg,device='cpu');o=gt.make_optimizer(m,cfg);cursor=gt.EpochCursor(42,2)
        gt.train_effective_batch(m,o,[{'history':[10],'target':20},{'history':[20],'target':30}],c,microbatch=2,device='cpu');cursor.next_batch(2)
        b=budget();b['reserved_steps']=b['completed_steps']=1;binding={'effective_config':cfg,'fixture_sha256':'a'*64,'device':'cpu'}
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            path=Path(d)/'good.pt';gt.save_checkpoint(path,m,o,cursor,binding=binding,selection=selection(),budget=b)
            value=torch.load(path,weights_only=True)
            for mode in ('missing','step','variance'):
                with self.subTest(mode=mode):
                    bad=copy.deepcopy(value)
                    if mode=='missing':bad['optimizer']['state']={}
                    elif mode=='step':next(iter(bad['optimizer']['state'].values()))['step'].zero_()
                    else:next(iter(bad['optimizer']['state'].values()))['exp_avg_sq'].fill_(-1)
                    candidate=Path(d)/(mode+'.pt');torch.save(bad,candidate);pin=hashlib.sha256(candidate.read_bytes()).hexdigest()
                    with self.assertRaises(ValueError):gt.load_checkpoint(candidate,pin,m,o,cursor,expected_binding=binding)

if __name__=='__main__':unittest.main()
