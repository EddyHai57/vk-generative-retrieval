"""Read-only stack production-path leakage matrix. Never optimizer or checkpoint."""
import copy
import hashlib
import inspect
import random

RUNTIME=('L1','L2','L3','L4','L5','L8','L9')
STATIC_REVIEW_SHA256='a64c25676d9385e5c5fb37890e7af9b159e195ea12ce760bd32d6fe9a8ffee4e'
REVIEWED_MODEL_COMMIT='4e03caa65cde93807593872812c74af380a5dee4'
STATIC={'L6':'train-only stats builder cutoff (0..24); constructor review',
        'L7':'fixed model assembly source reviewed; runtime auditor is not model input',
        'L10':'stats loader/constructor has no validation query binding'}

def flip_target(row,catalog):
    other=copy.deepcopy(row);other['target']=next(i for i in catalog if i!=row['target'])
    other.update(target_position=[30,999],past_likes=[],oov=not row.get('oov',False));return other

def audit_runtime(data,run,config,train_count=512):
    if data.get('synthetic') is not False:raise ValueError('production engineering=False input required')
    import numpy as np
    import torch
    from . import item_stats
    from . import retrieval_seq_combo_stack as model_module
    from .retrieval_experiment import normalize_export
    rows,queries=run.fr._view(data,50);rows=rows[:train_count];all_rows=rows+queries
    checks={k:{'status':'NOT_RUN','samples':0} for k in RUNTIME}
    references={'L6':item_stats.build_item_stats,'L7':model_module.SeqComboStack.anchor_vector,'L10':item_stats.load_item_stats}
    for key,meaning in STATIC.items():
        checks[key]={'status':'STATIC_REVIEWED','evidence':meaning,'source_sha256':hashlib.sha256(inspect.getsource(references[key]).encode()).hexdigest(),
                     'review':'2026-10-17-seq-combo-stack-code-review; unchanged model/builder boundary only (overall CHANGES_REQUESTED)',
                     'review_sha256':STATIC_REVIEW_SHA256,'reviewed_model_commit':REVIEWED_MODEL_COMMIT}
    receipt={'schema':'seq-combo-stack-leakage-v1','status':'NOT_ACCEPTED','model':'SeqComboStack','engineering':False,
             'train_samples':len(rows),'validation_samples':len(queries),'checks':checks,
             'number_map':{'design_L3':'historical_reverse_controls (old guard_l4_reverse_controls)',
                           'design_L4':'week_shift (old guard_l5_week_shift_invariance)',
                           'design_L8':'reject_week25 (old test_l6_week_25_rejected)'},
             'test_read':False,'checkpoint_read':False,'training':False}
    active='L1'
    try:
        if not rows or not queries:raise ValueError('nonempty train and validation required')
        model=run._model(data,config,'cpu').eval()
        with torch.no_grad():
            model.gate.fill_(.2)
            model.recency_logit_pos.copy_(torch.arange(7)*.01+.01)
            model.recency_logit_gap.copy_(torch.arange(6)*.02+.02)
            model.behavior_beta.copy_(torch.arange(6)*.03+.03)
        receipt['nonzero']={'gate':float(model.gate.detach()),'recency':True,'beta':True}
        def tensor(batch):return run.fr._tensors2(batch,data['idmap'],50,config['feature'])
        def require(condition,message):
            if not condition:raise AssertionError(message)
        with torch.inference_mode():
            for active in ('L1','L3','L4','L8'):
                count=0
                for offset in range(0,len(all_rows),128):
                    batch=all_rows[offset:offset+128];a=tensor(batch);base=model.users(*a)
                    if active=='L1':
                        b=tensor([flip_target(row,data['catalog']) for row in batch])
                        require(torch.equal(a[0],b[0]) and torch.equal(a[1],b[1]) and torch.equal(base,model.users(*b)),'target entered user inputs');count+=len(batch)
                    elif active=='L3':
                        changed=copy.deepcopy(batch)
                        supported=sorted(data['support'])
                        if len(supported)<2:raise ValueError('two supported items required')
                        for row in changed:row['history'][0]=supported[0] if row['history'][0]!=supported[0] else supported[1]
                        require(bool((model.users(*tensor(changed))!=base).any(dim=1).all()),'historical item reverse oracle had no effect')
                        require(torch.equal(model.users(*a),base),'restoring history did not restore users');count+=len(batch)
                    elif active=='L4':
                        originals=[];shifted=[]
                        for row in batch:
                            weeks=[p[0] for p in row['history_positions']];delta=1 if max(weeks)<=23 else -1 if min(weeks)>=1 else None
                            if delta is None:continue
                            originals.append(row);other=copy.deepcopy(row);other['history_positions']=[[w+delta,n] for w,n in row['history_positions']];shifted.append(other)
                        if originals:
                            require(torch.equal(model.users(*tensor(originals)),model.users(*tensor(shifted))),'absolute week leaked');count+=len(originals)
                    else:
                        # Every sampled example is checked, rather than only one per batch.
                        bad=a[1].clone();lengths=(a[0]!=0).sum(1);bad[torch.arange(len(batch)),lengths-1,3]=25
                        try:model.users(a[0],bad)
                        except ValueError:pass
                        else:raise AssertionError('week25 history accepted')
                        count+=len(batch)
                require(count>0,'no applicable runtime samples');checks[active]={'status':'pass','samples':count}
            active='L3'
            eligible=[row for row in all_rows if len(row['history'])>1 and len(set(row['history']))>1 and any(i in data['support'] for i in row['history'])][:128]
            require(bool(eligible),'no eligible feedback reverse oracle')
            moved=copy.deepcopy(eligible)
            for row in moved:row['side'][0][0]^=1
            require(bool((model.users(*tensor(eligible))!=model.users(*tensor(moved))).any(dim=1).all()),'own history feedback had no effect')
            # Feedback is confined to m. The Transformer branch itself must stay equal.
            require(torch.equal(model.parts(*tensor(eligible)).r,model.parts(*tensor(moved)).r),'feedback entered sequence branch')
            checks['L3_own_feedback']={'status':'pass','samples':len(eligible)}
            active='L2'
            exports=[]
            for batch in (rows,queries):
                left=run.combo.export_vectors_side4b(model,data['catalog'],batch,data['idmap'],128,50,run.cmodel.E_FEATURE)
                right=run.combo.export_vectors_side4b(model,data['catalog'],[flip_target(q,data['catalog']) for q in batch],data['idmap'],128,50,run.cmodel.E_FEATURE)
                require(all(np.array_equal(x,y) for x,y in zip(left,right)),'export depends on target fields');exports.append(left)
            checks['L2']={'status':'pass','samples':len(all_rows),'train_samples':len(rows),'validation_samples':len(queries)}
            active='L5';ids=torch.tensor([data['idmap'][i] for i in data['catalog']]);raw=model.candidate_items(ids).cpu().numpy()
            require(np.array_equal(model.candidate_items(ids.flip(0)).cpu().numpy(),raw[::-1]),'candidate depends on order or companions')
            checks['L5']={'status':'pass','samples':len(ids)}
            active='L9';normalized=normalize_export(raw);norm=np.linalg.norm(raw.astype(np.float64),axis=1)
            require(all(np.array_equal(x[1],normalized) and np.array_equal(x[2],norm) for x in exports),'raw versus export candidates differ')
            index={data['idmap'][item]:j for j,item in enumerate(data['catalog'])}
            for offset in range(0,len(rows),128):
                columns=run.b1._columns(rows[offset:offset+128],data,config,random.Random(0),None)[0].flatten()
                require(np.array_equal(model.candidate_items(columns).cpu().numpy(),np.stack([raw[index[int(i)]] for i in columns])),'training columns and catalog candidates differ')
            checks['L9']={'status':'pass','samples':len(ids),'train_samples':len(rows)}
        receipt['status']='pass'
    except (AssertionError,ValueError,RuntimeError,KeyError) as exc:
        receipt['status']='FAIL';checks[active]={'status':'FAIL','samples':checks[active].get('samples',0),'error':type(exc).__name__+': '+str(exc)}
    return receipt
