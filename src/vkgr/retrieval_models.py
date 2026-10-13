"""Original fixed core two-towers; PAD=0, train-defined ID support."""
import torch
MODEL_ORDER=("id_mean","id_content_mean","ordinal_content_mean","content_transformer")

class TwoTower(torch.nn.Module):
 def __init__(self,architecture,content,supported_indices):
  super().__init__()
  if architecture not in MODEL_ORDER:raise ValueError('unsupported architecture')
  if content.dtype!=torch.float32 or content.ndim!=2 or content.shape[1]!=64 or len(content)<2 or not torch.isfinite(content).all() or torch.any(content[0]):raise ValueError('invalid content/PAD')
  self.architecture=architecture
  if architecture!="id_mean":
   self.register_buffer("content",content.detach().clone())
   self.content_projection=torch.nn.Linear(64,64,bias=False)
  if architecture=="content_transformer":
   self.positions=torch.nn.Embedding(50,64)
   self.transformer=torch.nn.TransformerEncoderLayer(64,4,dim_feedforward=256,dropout=.1,batch_first=True)
   self.user_projection=torch.nn.Linear(64,64,bias=False)
  self.embedding=torch.nn.Embedding(len(content),64,padding_idx=0)
  supported=torch.zeros(len(content),dtype=torch.bool)
  if any(type(i)is not int or not 1<=i<len(content) for i in supported_indices):raise ValueError('invalid supported ID')
  supported[supported_indices]=True;self.register_buffer('supported',supported)
 def items(self,indices):
  if indices.dtype!=torch.long or torch.any(indices<0) or torch.any(indices>=len(self.supported)):raise ValueError('invalid item indices')
  encoded=self.embedding(indices)*self.supported[indices].unsqueeze(-1)
  if self.architecture!="id_mean":encoded=encoded+self.content_projection(self.content[indices])
  return encoded
 def users(self,histories):
  if histories.ndim!=2 or not 1<=histories.shape[1]<=50 or torch.any((histories!=0).sum(1)==0):raise ValueError('invalid/empty history')
  valid=histories!=0;encoded=self.items(histories)
  if self.architecture=="content_transformer":
   ordinal=(valid.long().cumsum(1)-1).clamp_min(0)
   encoded=encoded+self.positions(ordinal)*valid.unsqueeze(-1)
   encoded=self.transformer(encoded,src_key_padding_mask=~valid)
   return self.user_projection((encoded*valid.unsqueeze(-1)).sum(1)/valid.sum(1,keepdim=True))
  if self.architecture=="ordinal_content_mean":
   positions=valid.long().cumsum(1)-1
   distances=valid.sum(1,keepdim=True)-1-positions
   weights=torch.pow(encoded.new_tensor(.8),distances)*valid
   return (encoded*weights.unsqueeze(-1)).sum(1)/weights.sum(1,keepdim=True)
  return encoded.sum(1)/valid.sum(1,keepdim=True)
 def load_state_dict(self,state_dict,strict=True,assign=False):
  for name in ('supported','content'):
   if hasattr(self,name):
    expected=getattr(self,name);actual=state_dict.get(name)
    if not isinstance(actual,torch.Tensor) or actual.dtype!=expected.dtype or not torch.equal(actual.cpu(),expected.cpu()):raise ValueError('checkpoint immutable support/content mismatch')
  return super().load_state_dict(state_dict,strict=strict,assign=assign)
