"""One-off production-input leakage receipt. No checkpoint, fitting or optimizer."""
import argparse,copy,datetime,hashlib,json,os,subprocess,sys
from pathlib import Path

def flip_target(row,catalog):
 out=copy.deepcopy(row);out['target']=next(x for x in catalog if x!=row['target']);out['target_position']=[26,999];out['oov']=not row.get('oov',False);out['past_likes']=[]
 return out

def audit_runtime(data,run,model_module,train_count=512):
 if data.get('synthetic') is not False:raise ValueError('engineering=False production input required')
 import numpy as np
 import torch
 rows,queries=run.fr._view(data,50);rows=rows[:train_count];all_rows=rows+queries
 if not rows or not queries:raise ValueError('empty input population')
 variants={'A':model_module.A_FEATURE,'C':'f4_item_popularity','D':'f4_item_quality','E':model_module.E_FEATURE};result={}
 supported=sorted(data['support'])
 if len(supported)<2:raise ValueError('reverse oracle needs two supported items')
 for label,variant in variants.items():
  buckets=data['stat_buckets'] if label in ('C','D') else None
  model=model_module.make_feat04b_model(variant,data['content'],data['supported_indices'],50,42,stat_buckets=buckets).eval()
  if hasattr(model,'recency_logit_pos'):
   with torch.no_grad():
    model.recency_logit_pos.copy_(torch.arange(len(model.recency_logit_pos),dtype=torch.float32)*.01+.01)
    model.recency_logit_gap.copy_(torch.arange(len(model.recency_logit_gap),dtype=torch.float32)*.02+.02)
  if hasattr(model,'behavior_beta'):
   with torch.no_grad():model.behavior_beta.copy_(torch.arange(len(model.behavior_beta),dtype=torch.float32)*.03+.03)
  checks={};counts={};shifted_count=0
  def tensor(batch):return run._tensors4b(batch,data['idmap'],50,variant)
  with torch.inference_mode():
   for start in range(0,len(all_rows),128):
    batch=all_rows[start:start+128];a=tensor(batch);u=model.users(*a)
    changed=[flip_target(r,data['catalog']) for r in batch];b=tensor(changed)
    if not (torch.equal(a[0],b[0]) and torch.equal(a[1],b[1]) and torch.equal(u,model.users(*b))):raise AssertionError(label+' L1 target entered user path')
    reverse=copy.deepcopy(batch)
    for row in reverse:row['history'][0]=supported[0] if row['history'][0]!=supported[0] else supported[1]
    moved=model.users(*tensor(reverse))
    if not bool((moved!=u).any(dim=1).all()):raise AssertionError(label+' L3 historical item had no effect')
    shiftable=[];moved_rows=[]
    for row in batch:
     weeks=[v[0] for v in row['history_positions']];delta=1 if max(weeks)<=23 else -1 if min(weeks)>=1 else None
     if delta is None:continue
     shiftable.append(row);other=copy.deepcopy(row);other['history_positions']=[[w+delta,o] for w,o in row['history_positions']];moved_rows.append(other)
    if shiftable:
     if not torch.equal(model.users(*tensor(shiftable)),model.users(*tensor(moved_rows))):raise AssertionError(label+' L4 absolute week leak')
     shifted_count+=len(shiftable)
    for i,row in enumerate(batch):
     expected=[data['idmap'][x] for x in row['history']]
     if a[0][i,:len(expected)].tolist()!=expected:raise AssertionError(label+' L5 item identity mismatch')
    bad=copy.deepcopy(batch[:1]);bad[0]['history_positions'][-1][0]=25
    try:model.users(*tensor(bad))
    except (ValueError,RuntimeError):pass
    else:raise AssertionError(label+' L8 week25 was accepted')
   if shifted_count==0:raise AssertionError(label+' L4 no eligible shift oracle')
   feedback_count=0
   if label=='E':
    chosen=[r for r in all_rows if len(r['history'])>1 and len(set(r['history']))>1 and any(i in data['support'] for i in r['history'])][:128]
    if not chosen:raise AssertionError('E no eligible feedback oracle')
    flipped=copy.deepcopy(chosen)
    for row in flipped:row['side'][0][0] ^= 1
    if not bool((model.users(*tensor(chosen))!=model.users(*tensor(flipped))).any(dim=1).all()):raise AssertionError('E own-history feedback did not affect nonzero beta')
    feedback_count=len(chosen)
   qflip=[flip_target(q,data['catalog']) for q in queries]
   left=run.export_vectors_side4b(model,data['catalog'],queries,data['idmap'],128,50,variant)
   right=run.export_vectors_side4b(model,data['catalog'],qflip,data['idmap'],128,50,variant)
   if any(not np.array_equal(a,b) for a,b in zip(left,right)):raise AssertionError(label+' L2 export target leakage')
   catalog_ids=torch.tensor([data['idmap'][x] for x in data['catalog']],dtype=torch.long)
   raw_candidates=model.candidate_items(catalog_ids)
   if not torch.equal(model.candidate_items(catalog_ids.flip(0)),raw_candidates.flip(0)):raise AssertionError(label+' L5 candidate depends on item order')
   if label in ('C','D'):
    ids=catalog_ids
    raw=model.candidate_items(ids).cpu().numpy();norm=np.linalg.norm(raw.astype(np.float64),axis=1)
    from vkgr.retrieval_experiment import normalize_export
    normalized=normalize_export(raw)
    if not np.array_equal(left[1],normalized) or not np.array_equal(left[2],norm):raise AssertionError(label+' L9 training/export candidate mismatch')
    import random
    for offset in range(0,len(rows),128):
     config={**run.default_config(),'feature':variant}
     columns=run.b1._columns(rows[offset:offset+128],data,config,random.Random(0),None)[0].flatten()
     actual=model.candidate_items(columns).cpu().numpy()
     positions={data['idmap'][item]:j for j,item in enumerate(data['catalog'])}
     if not np.array_equal(actual,np.stack([raw[positions[int(i)]] for i in columns])):raise AssertionError(label+' L5/L9 training columns differ from export candidates')
  for name in ('L1','L2','L3','L4','L5','L8','L9'):
   applicable=name!='L9' or label in ('C','D')
   count=len(queries) if name=='L2' else shifted_count if name=='L4' else len(data['catalog']) if name=='L9' and applicable else len(all_rows) if name!='L8' else (len(all_rows)+127)//128
   checks[name]={'status':'pass' if applicable else 'NOT_APPLICABLE','samples':count if applicable else 0}
  if label=='E':checks['L3_own_feedback']={'status':'pass','samples':feedback_count}
  for name in ('L6','L7','L10'):checks[name]={'status':'STATIC_REVIEWED','evidence':'fixed-commit independent review; not runtime execution'}
  result[label]={'variant':variant,'checks':checks,'nonzero_recency':hasattr(model,'recency_logit_pos'),'nonzero_beta':hasattr(model,'behavior_beta'),'engineering':False}
 return {'variants':result,'train_samples':len(rows),'validation_samples':len(queries),'status':'pass'}

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--parameters',required=True);parser.add_argument('--parameters-sha',required=True);parser.add_argument('--output',required=True);a=parser.parse_args()
 payload=Path(a.parameters).read_bytes()
 if hashlib.sha256(payload).hexdigest()!=a.parameters_sha:raise ValueError('private parameters SHA mismatch')
 req=json.loads(payload);root=Path(req['model_root'])
 if subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()!=req['model_commit']:raise ValueError('frozen model commit mismatch')
 for name,pin in req['source_files'].items():
  if hashlib.sha256((root/name).read_bytes()).hexdigest()!=pin:raise ValueError('frozen model source mismatch')
 for n in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[n]='1'
 sys.path.insert(0,str(root/'src'))
 import torch
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 from vkgr import retrieval_feat04combo_run as combo
 from vkgr import retrieval_feat04b_run as run
 from vkgr import retrieval_feat04b as model
 if hashlib.sha256(Path(req['inputs']).read_bytes()).hexdigest()!=req['inputs_sha256']:raise ValueError('input manifest hash mismatch')
 spec=combo.validate_inputs4b(json.loads(Path(req['inputs']).read_text()),combo.m4.E_FEATURE)
 data,_,binding=combo.b1.real_inputs(spec);data,extra=combo.apply_feat04b(data,spec,50,combo.m4.E_FEATURE)
 if data.get('synthetic') is not False or len(data['rows'])!=86785:raise ValueError('production loader did not return frozen real input')
 try:
  record=audit_runtime(data,run,model)
 except (AssertionError,ValueError,RuntimeError) as exc:
  record={'status':'FAIL','error':type(exc).__name__+': '+str(exc),'variants':{v:{'checks':{k:{'status':'NOT_ACCEPTED','samples':0} for k in ('L1','L2','L3','L4','L5','L6','L7','L8','L9','L10')}} for v in ('A','C','D','E')}}
 record.update(tool_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),read_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),execution_commit=subprocess.check_output(['git','-C',str(Path(__file__).resolve().parent.parent),'rev-parse','HEAD'],text=True).strip(),model_commit=req['model_commit'],model_source_files=req['source_files'],loader='retrieval_feat04combo_run.validate_inputs4b/b1.real_inputs/apply_feat04b(engineering=False)',inputs_sha256=req['inputs_sha256'],binding={**binding,**extra},real_data_digest={k:combo.b1._digest(data[k]) for k in ('rows','queries','catalog')},static_review=req['review'],number_map={'design_L3':'guard_l4_reverse_controls','design_L4':'guard_l5_week_shift_invariance','design_L8':'test_l6_week_25_rejected'},test_read=False,checkpoint_read=False,training=False)
 with Path(a.output).open('x') as f:json.dump(record,f,sort_keys=True,allow_nan=False)
 print(json.dumps(record,sort_keys=True,allow_nan=False))
 if record['status']!='pass':raise SystemExit(2)
if __name__=='__main__':main()
