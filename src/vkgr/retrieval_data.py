"""Traditional protocol consumer; legacy producer validation remains separate."""
from pathlib import Path
from .data_audit import file_sha256
from .manifest import read_json
FROZEN='36fc3a451f1e8bd2dc691b15c382015b6762fec8342d318ce9771bad5adfc972'
def checked(path,digest):
 path=Path(path)
 if path.is_symlink() or file_sha256(path)!=digest:raise ValueError('input hash mismatch')
 return path

def validate_contract(root):
 root=Path(root);path=checked(root/'artifacts/p3d-retrieval-contract-manifest.json',FROZEN);value=read_json(path)
 for rec in value['bound_files']:checked(root/rec['path'],rec['sha256'])
 return value

def catalog_map(items):
 if not items or any(type(i)is not int or not 0<=i<2**32 for i in items) or items!=sorted(set(items)):raise ValueError('invalid sorted uint32 catalog')
 return {item:index+1 for index,item in enumerate(items)}

def prefix_examples(partitions):
 from collections import deque
 histories={};past={};previous=-1;example_id=0
 for week,rows in partitions:
  if type(week)is not int or not previous<week<=24:raise ValueError('train ordering conflict')
  previous=week
  for ordinal,row in enumerate(rows):
   if any(type(row.get(k))is not int or not 0<=row[k]<2**32 for k in ('user_id','item_id')) or type(row.get('like'))is not bool:raise ValueError('invalid interaction')
   if not row['like']:continue
   user,item=row['user_id'],row['item_id'];h=histories.setdefault(user,deque(maxlen=50));seen=past.setdefault(user,set());position=[week,ordinal]
   if h:
    yield {'example_id':example_id,'user_id':user,'history':[x[0] for x in h],'history_positions':[x[1] for x in h], 'target':item,'target_position':position,'past_likes':sorted(seen)}
    example_id+=1
   h.append((item,position));seen.add(item)

def read_split(directory,split,digest,catalog):
 if split!='validation':raise ValueError('test gate closed; unsupported split')
 rows=read_json(checked(Path(directory)/f'{split}-queries.json',digest));members=set(catalog);users=set()
 for q in rows:
  if set(q)!={'user_id','history','history_positions','target','target_position','oov'} or type(q['user_id'])is not int or not 0<=q['user_id']<2**32 or q['user_id'] in users or type(q['target'])is not int or not 0<=q['target']<2**32 or type(q['oov'])is not bool or q['oov']!=(q['target']not in members):raise ValueError('invalid query schema')
  users.add(q['user_id'])
  if not 1<=len(q['history'])<=50 or any(type(i)is not int or i not in members for i in q['history']) or len(q['history_positions'])!=len(q['history']):raise ValueError('query history mismatch')
  positions=q['history_positions'];target=q['target_position']
  if any(type(p)is not list or len(p)!=2 or any(type(x)is not int or x<0 for x in p) or p[0]>24 for p in positions) or any(a>=b for a,b in zip(positions,positions[1:])) or type(target)is not list or len(target)!=2 or target[0]!=25 or type(target[1])is not int or target[1]<0:raise ValueError('query ordering conflict')
 return rows

def content_join(catalog,ids,embedding):
 import numpy as np
 catalog_map(catalog)
 if ids.dtype!=np.uint32 or ids.ndim!=1 or embedding.dtype!=np.float16 or embedding.shape!=(len(ids),64) or len(set(ids.tolist()))!=len(ids) or not np.isfinite(embedding).all():raise ValueError('official content schema mismatch')
 lookup={int(i):j for j,i in enumerate(ids)}
 if any(i not in lookup for i in catalog):raise ValueError('missing content')
 joined=embedding[[lookup[i] for i in catalog]].astype(np.float32);norm=np.linalg.norm(joined,axis=1,keepdims=True);joined=np.divide(joined,norm,out=np.zeros_like(joined),where=norm!=0)
 if not np.isfinite(joined).all():raise ValueError('nonfinite content')
 return np.concatenate([np.zeros((1,64),dtype=np.float32),joined])

def validate_example(row,idmap,index):
 if set(row)!={'example_id','user_id','history','history_positions','target','target_position','past_likes'} or row['example_id']!=index or type(row['user_id'])is not int or not 0<=row['user_id']<2**32 or type(row['target'])is not int or row['target']not in idmap:raise ValueError('invalid example identity')
 h=row['history'];positions=row['history_positions'];target=row['target_position'];past=row['past_likes']
 if not 1<=len(h)<=50 or len(positions)!=len(h) or any(type(i)is not int or i not in idmap for i in h+past) or past!=sorted(set(past)) or not set(h)<=set(past):raise ValueError('invalid prefix evidence')
 if any(type(p)is not list or len(p)!=2 or any(type(i)is not int or i<0 for i in p) or p[0]>24 for p in positions+[target]) or any(a>=b for a,b in zip(positions,positions[1:])) or positions[-1]>=target:raise ValueError('prefix ordering conflict')

def write_prepared(output,catalog,features,examples,binding,supported_items):
 import numpy as np
 from .data_audit import canonical_bytes
 output=Path(output);output.mkdir(parents=True,exist_ok=False);idmap=catalog_map(catalog)
 if features.dtype!=np.float32 or features.shape!=(len(catalog)+1,64) or not np.isfinite(features).all() or np.any(features[0]):raise ValueError('invalid prepared feature shape')
 with (output/'catalog.json').open('xb') as stream:stream.write(canonical_bytes(catalog))
 with (output/'content.npy').open('xb') as stream:np.save(stream,features,allow_pickle=False)
 count=0;supported=set(supported_items)
 if sorted(supported)!=supported_items or any(type(i)is not int or i not in idmap for i in supported):raise ValueError("invalid train positive support")
 with (output/'examples.jsonl').open('xb') as stream:
  for row in examples:
   validate_example(row,idmap,count)
   if not set(row['past_likes']+[row['target']])<=supported:raise ValueError('example positive missing support')
   stream.write(canonical_bytes(row));count+=1
 if count<1:raise ValueError('empty training examples')
 with (output/'support.json').open('xb') as stream:stream.write(canonical_bytes(supported_items))
 with (output/'idmap.json').open('xb') as stream:stream.write(canonical_bytes({'items':catalog,'pad':0,'start':1,'supported_indices':sorted(idmap[i] for i in supported)}))
 manifest={'schema_version':1,'task_id':'P4B','kind':'prepare','status':'success','binding':binding,'examples':count,'catalog_items':len(catalog),'artifacts':{name:file_sha256(output/name) for name in ('catalog.json','content.npy','examples.jsonl','idmap.json','support.json')}}
 with (output/'prepare-manifest.json').open('xb') as stream:stream.write(canonical_bytes(manifest))
 return manifest

def verify_prepared(output):
 import numpy as np,json
 output=Path(output);value=read_json(output/'prepare-manifest.json')
 if value.get('schema_version')!=1 or value.get('task_id')!='P4B' or value.get('kind')!='prepare' or value.get('status')!='success' or set(value['artifacts'])!={'catalog.json','content.npy','examples.jsonl','idmap.json','support.json'}:raise ValueError('invalid prepare manifest')
 for name,digest in value['artifacts'].items():checked(output/name,digest)
 catalog=read_json(output/'catalog.json');idmap=catalog_map(catalog);features=np.load(output/'content.npy',allow_pickle=False)
 if features.dtype!=np.float32 or features.shape!=(len(catalog)+1,64) or not np.isfinite(features).all() or np.any(features[0]):raise ValueError('prepared content mismatch')
 count=0;support_items=read_json(output/'support.json');supported=set(support_items)
 if support_items!=sorted(supported) or any(type(i)is not int or i not in idmap for i in supported):raise ValueError("invalid train positive support")
 with (output/'examples.jsonl').open() as stream:
  for line in stream:
   row=json.loads(line);validate_example(row,idmap,count)
   if not set(row['past_likes']+[row['target']])<=supported:raise ValueError('example support mismatch')
   count+=1
 if count!=value['examples'] or len(catalog)!=value['catalog_items'] or read_json(output/'idmap.json')!={'items':catalog,'pad':0,'start':1,'supported_indices':sorted(idmap[i] for i in supported)}:raise ValueError('prepared identity mismatch')
 return value
