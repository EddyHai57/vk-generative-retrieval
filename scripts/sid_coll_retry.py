"""Eddy-approved one-time preflight repair; reuse immutable accepted inputs/SIDs.
Original plan/failed directories are never modified; deadline is inherited.
"""
import copy,hashlib,json
from pathlib import Path
REUSED=('input_preflight','input','input_verify','sid_fit','sid_verify')
DEP='/workspace/vkgr-sid-coll-g-v2';OPS='/workspace/vkgr-sid-coll-ops-v2'
OLDDEP='/workspace/vkgr-sid-coll-g-v1';OLDOPS='/workspace/vkgr-sid-coll-ops-v1'
R='/workspace/vkgr-infra01-root'
def wire(v):return json.dumps(v,sort_keys=True,allow_nan=False).encode()
def sha(v):return hashlib.sha256(wire(v)).hexdigest()
def build(original,snapshot,deployment,deployment_ref):
 old=original['pipeline-plan.json'];plan=copy.deepcopy(old);docs={};sources=deployment['files'];commit=deployment['commit']
 plan['control_dir']=OPS+'/controller';plan['common_refs']=[deployment_ref]
 plan['authorization']='Eddy 2026-10-17: one repaired preflight per branch; reuse verified vectors/SID; original deadline'
 plan['retained_failed_runs']=['sid-coll-c1-preflight-v1','sid-coll-c2-preflight-v1']
 for label,branch in plan['branches'].items():
  originals=old['branches'][label]['stages'];byrid={s['job']['run_id']:n for n,s in originals.items()};newpins={};newids={}
  for name,stage in branch['stages'].items():
   rid=stage['job']['run_id']
   if name in REUSED:
    accepted=snapshot['runs'][rid]
    if accepted['state']!='ACCEPTED':raise ValueError('reused stage not accepted: '+rid)
    plan['common_refs'].append(copy.deepcopy(accepted['receipt_ref']));continue
   spec=copy.deepcopy(original[Path(stage['job']['config']).name]);params=spec['stages'];generic=next(iter(params))
   newrid=rid[:-3]+'-v2' if name=='preflight' else rid;newids[name]=newrid
   spec['commit']=commit;spec['source_files']=copy.deepcopy(sources);params[generic]['run_id']=newrid
   for role,ref in list(spec['refs'].items()):
    if ref.get('sha256')!='DYNAMIC':
     if ref['path'].startswith(OLDDEP+'/'):
      relative=ref['path'][len(OLDDEP)+1:]
      if sources.get(relative)!=ref['sha256']:raise ValueError('frozen config bytes changed')
      ref['path']=DEP+'/'+relative
     continue
    origin=ref['from'];parent=byrid[origin['run_id']]
    if parent in REUSED:
     accepted=snapshot['runs'][origin['run_id']]
     proof=accepted['receipt_ref'] if origin['proof']=='__receipt__' else accepted['artifacts'][origin['proof']]
     if proof['path']!=ref['path']:raise ValueError('reused artifact path differs')
     spec['refs'][role]={k:proof[k] for k in ('path','sha256')}
    else:
     origin['spec_sha256']=newpins[parent];origin['run_id']=newids[parent];origin['receipt_path']=R+'/runs/'+newids[parent]+'/out/receipt.json'
     ref['path']=ref['path'].replace('/runs/'+originals[parent]['job']['run_id']+'/', '/runs/'+newids[parent]+'/')
   pin=sha(spec);newpins[name]=pin;basename=newrid+'.json';docs[basename]=spec
   job=stage['job'];job['run_id']=newrid;job['cwd']=DEP;job['config']=OPS+'/'+basename
   job['inputs']=[job['config'],OPS+'/deployment.json'];job['env']=[v.replace(OLDDEP,DEP) for v in job['env']]
   argv=[v.replace(OLDDEP+'/',DEP+'/') for v in job['argv']];argv=argv[:argv.index('--ref')] if '--ref' in argv else argv
   argv[argv.index('--spec')+1]=job['config'];argv[argv.index('--spec-sha')+1]=pin;argv[argv.index('--output')+1]=R+'/runs/'+newrid+'/out'
   for role,ref in spec['refs'].items():
    if ref.get('sha256')=='DYNAMIC':
     parent=next(n for n,r in newids.items() if r==ref['from']['run_id']);proof=ref['from']['proof']
     token='@'+parent+('.receipt_sha256' if proof=='__receipt__' else '.proofs.'+proof+'.sha256');argv+=['--ref',role,token]
   job['argv']=argv;stage['proof']['path']=R+'/runs/'+newrid+'/out/receipt.json'
   data={}
   for role,ref in spec['refs'].items():
    if ref['sha256']!='DYNAMIC':data[role]=ref['sha256'];continue
    parent=next(n for n,r in newids.items() if r==ref['from']['run_id']);proof=ref['from']['proof']
    data[role]='@'+parent+('.receipt_sha256' if proof=='__receipt__' else '.proofs.'+proof+'.sha256')
   stage['proof']['expected']={'schema':'coll-stage-receipt-v1','status':'complete' if generic=='train' else 'verified','stage':generic,'variant':spec['variant'],'run_id':newrid,'binding.commit':commit,'binding.spec_sha256':pin,'binding.config_sha256':spec['refs'].get('config'+str(params[generic].get('seed')),{}).get('sha256'),'binding.data':data}
 docs['pipeline-plan.json']=plan
 for n,v in docs.items():
  if len(wire(v))>=(2*1024*1024 if n=='pipeline-plan.json' else 48*1024):raise ValueError('oversized recovery metadata')
 return docs
