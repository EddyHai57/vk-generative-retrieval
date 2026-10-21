"""Original small SID encoder-decoder and catalog constrained beam search.

PAD=0, BOS=1; four disjoint SID namespaces. No EOS for fixed four digits.
Beam accumulates full-vocabulary log probabilities before legal masking.
"""
import math
import torch
from torch import nn


class SIDTokens:
    def __init__(self,catalog):
        if len(catalog.bases)!=4: raise ValueError('four SID digits required')
        self.catalog=catalog;self.offsets=[];total=2
        for base in catalog.bases:
            self.offsets.append(total);total+=base
        self.vocab_size=total

    def item(self,item):
        return [offset+digit for offset,digit in zip(self.offsets,self.catalog.encode(item))]

    def history(self,history):
        if not isinstance(history,list) or not history: raise ValueError('nonempty history required')
        return [token for item in history[-50:] for token in self.item(item)]

    def batch(self,histories,targets,device='cpu'):
        if len(histories)!=len(targets) or not targets: raise ValueError('invalid batch')
        rows=[self.history(h) for h in histories];width=max(map(len,rows))
        source=torch.tensor([r+[0]*(width-len(r)) for r in rows],dtype=torch.long,device=device)
        labels=torch.tensor([self.item(t) for t in targets],dtype=torch.long,device=device)
        decoder=torch.cat([torch.ones((len(rows),1),dtype=torch.long,device=device),labels[:,:-1]],dim=1)
        return source,decoder,labels


def causal_mask(length,device='cpu'):
    return torch.ones((length,length),dtype=torch.bool,device=device).triu(1)


class TigerModel(nn.Module):
    def __init__(self,vocab_size,d_model=128,nhead=4,encoder_layers=2,decoder_layers=2,ff=512):
        super().__init__();self.d_model=d_model
        self.embedding=nn.Embedding(vocab_size,d_model,padding_idx=0)
        self.position=nn.Embedding(200,d_model)
        self.transformer=nn.Transformer(d_model=d_model,nhead=nhead,num_encoder_layers=encoder_layers,num_decoder_layers=decoder_layers,dim_feedforward=ff,dropout=0.,batch_first=True)
        # Avoid nested tensor conversion changing the padding invariant/backend.
        self.transformer.encoder.enable_nested_tensor=False
        self.transformer.encoder.use_nested_tensor=False
        self.output=nn.Linear(d_model,vocab_size)

    def embed(self,tokens):
        if tokens.ndim!=2 or not 1<=tokens.shape[1]<=200: raise ValueError('invalid token sequence')
        positions=torch.arange(tokens.shape[1],device=tokens.device)
        return self.embedding(tokens)*math.sqrt(self.d_model)+self.position(positions)[None]

    def encode(self,source):
        padding=source.eq(0)
        if padding.all(dim=1).any(): raise ValueError('empty source')
        return self.transformer.encoder(self.embed(source),src_key_padding_mask=padding),padding

    def decode(self,memory,padding,decoder):
        hidden=self.transformer.decoder(self.embed(decoder),memory,tgt_mask=causal_mask(decoder.shape[1],decoder.device),memory_key_padding_mask=padding)
        return self.output(hidden)

    def forward(self,source,decoder):
        memory,padding=self.encode(source)
        return self.decode(memory,padding,decoder)


def constrained_beam(catalog,codec,score,beam=20):
    if type(beam) is not int or beam<1: raise ValueError('invalid beam')
    active=[((),0.)];audit={'invalid':0,'deadend':0,'collision':0,'max_active':1}
    for position in range(4):
        logits=score([path for path,_ in active])
        if logits.shape!=(len(active),codec.vocab_size) or not torch.isfinite(logits).all(): raise ValueError('nonfinite/invalid beam logits')
        logprob=torch.log_softmax(logits,dim=-1).detach().cpu();expanded=[]
        for row,(path,total) in enumerate(active):
            allowed=catalog.allowed_next(path)
            if not allowed: audit['deadend']+=1
            for digit in allowed:
                expanded.append((path+(digit,),total+float(logprob[row,codec.offsets[position]+digit])))
        if not expanded: raise ValueError('beam deadend')
        expanded.sort(key=lambda x:(-x[1],x[0]));active=expanded[:beam]
        audit['max_active']=max(audit['max_active'],len(active))
    ranked=[]
    for path,total in active:
        item=catalog.decode(path);ranked.append((item,total))
    ranked.sort(key=lambda x:(-x[1],x[0]))
    audit['collision']=len(ranked)-len(set(x[0] for x in ranked))
    if audit['collision']: raise ValueError('beam item collision')
    return [x[0] for x in ranked],audit


def decode_history(model,codec,history,beam=20):
    device=next(model.parameters()).device
    source=torch.tensor([codec.history(history)],dtype=torch.long,device=device)
    model.eval()
    with torch.no_grad():
        memory,padding=model.encode(source)
        def score(prefixes):
            rows=[[1]+[codec.offsets[j]+digit for j,digit in enumerate(path)] for path in prefixes]
            decoder=torch.tensor(rows,dtype=torch.long,device=device)
            return model.decode(memory.expand(len(rows),-1,-1),padding.expand(len(rows),-1),decoder)[:,-1]
        return constrained_beam(codec.catalog,codec,score,beam)
