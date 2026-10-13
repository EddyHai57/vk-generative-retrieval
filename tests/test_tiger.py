"""Synthetic model and exhaustive constrained beam oracles."""
import unittest
import numpy as np
import torch
from vkgr.sid import SIDCatalog
from vkgr import tiger


def catalog():
    return SIDCatalog(np.array([10,20,30,40],dtype=np.uint32),np.array([[0,0,0,0],[0,0,0,1],[1,0,0,0],[1,1,0,0]],dtype=np.uint32),2,3,[])


class TigerTests(unittest.TestCase):
    def test_namespace_history_teacher_forcing_and_oov(self):
        codec=tiger.SIDTokens(catalog())
        self.assertEqual(codec.offsets,[2,4,6,8])
        self.assertEqual(codec.item(20),[2,4,6,9])
        source,decoder,labels=codec.batch([[10,20],[30]],[20,40])
        self.assertEqual(source.tolist(),[[2,4,6,8,2,4,6,9],[3,4,6,8,0,0,0,0]])
        self.assertEqual(decoder.tolist(),[[1,2,4,6],[1,3,5,6]])
        self.assertEqual(labels.tolist(),[[2,4,6,9],[3,5,6,8]])
        with self.assertRaisesRegex(ValueError,'OOV'): codec.batch([[999]],[10])
        with self.assertRaises(ValueError): codec.batch([[]],[10])
        self.assertEqual(codec.history([10]*51),codec.history([10]*50))

    def test_masks_no_future_leak_and_padding_invariance(self):
        torch.manual_seed(7)
        model=tiger.TigerModel(10,d_model=16,nhead=4,encoder_layers=1,decoder_layers=1,ff=32).eval()
        source=torch.tensor([[2,4,6,8,0,0,0,0]])
        a=torch.tensor([[1,2,4,6]]);b=torch.tensor([[1,3,5,7]])
        with torch.no_grad():
            x=model(source,a);y=model(source,b)
            self.assertTrue(torch.equal(x[:,0],y[:,0]))
            self.assertFalse(torch.equal(x[:,2],y[:,2]))
            z=model(source[:,:4],a)
            self.assertTrue(torch.allclose(x,z,atol=2e-6,rtol=2e-6))
        self.assertEqual(tiger.causal_mask(3).tolist(),[[False,True,True],[False,False,True],[False,False,False]])

    def test_beam_exhaustive_global_width_and_item_oracle(self):
        c=catalog();codec=tiger.SIDTokens(c)
        def score(prefixes):
            # Full vocabulary normalization BEFORE legal masking is frozen.
            rows=[]
            for p in prefixes:
                row=torch.arange(codec.vocab_size,dtype=torch.float64)*.07
                if p: row=row+torch.tensor([((sum(p)+i)%3)*.1 for i in range(codec.vocab_size)])
                rows.append(row)
            return torch.stack(rows)
        predictions,audit=tiger.constrained_beam(c,codec,score,beam=20)
        oracle=[]
        for item in c.ids:
            p=();total=0.
            for digit in c.encode(int(item)):
                total+=float(torch.log_softmax(score([p])[0],dim=0)[codec.offsets[len(p)]+digit]);p=p+(digit,)
            oracle.append((int(item),total))
        oracle.sort(key=lambda x:(-x[1],x[0]))
        self.assertEqual(predictions,[x[0] for x in oracle])
        self.assertEqual(audit['invalid'],0);self.assertEqual(audit['deadend'],0);self.assertEqual(audit['collision'],0)
        one,_=tiger.constrained_beam(c,codec,score,beam=1)
        self.assertEqual(len(one),1)
        with self.assertRaises(ValueError): tiger.constrained_beam(c,codec,lambda ps:torch.full((len(ps),codec.vocab_size),float('nan')),20)

    def test_each_decoder_position_future_invariant(self):
        torch.manual_seed(8);m=tiger.TigerModel(10,16,4,1,1,32).eval();src=torch.tensor([[2,4,6,8]])
        a=torch.tensor([[1,2,4,6]])
        with torch.no_grad():
            expected=m(src,a)
            for position in range(3):
                b=a.clone();b[:,position+1:]=9
                self.assertTrue(torch.allclose(expected[:,:position+1],m(src,b)[:,:position+1],atol=1e-7,rtol=1e-7))

    def test_pruned_global_beam_matches_independent_layer_oracle(self):
        import itertools,math
        paths=list(itertools.product(range(3),repeat=4));ids=np.arange(100,181,dtype=np.uint32)
        c=SIDCatalog(ids,np.array(paths,dtype=np.uint32),3,3,[]);codec=tiger.SIDTokens(c);width=5
        def score(prefixes):
            return torch.tensor([[math.sin(i+sum((j+1)*v for j,v in enumerate(p)))*3 for i in range(codec.vocab_size)] for p in prefixes],dtype=torch.float64)
        actual,audit=tiger.constrained_beam(c,codec,score,width)
        active=[((),0.)]
        for depth in range(4):
            options=[]
            for prefix,total in active:
                values=score([prefix])[0].tolist();normal=math.log(sum(math.exp(x) for x in values))
                for digit in range(3):options.append((prefix+(digit,),total+values[codec.offsets[depth]+digit]-normal))
            active=sorted(options,key=lambda x:(-x[1],x[0]))[:width]
        expected=[c.decode(p) for p,_ in sorted(active,key=lambda x:(-x[1],c.decode(x[0])))]
        self.assertEqual(actual,expected);self.assertEqual(audit['max_active'],width)
