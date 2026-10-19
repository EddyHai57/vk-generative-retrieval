"""P stats freshdeploy helper, copied without algorithm changes.

Provenance: project-local helper scripts/sid_structure_deploy.py at commit
135a853d8c570e20553e113b337b432bc1b9cfda (blob f9483e4a0d973fb4d272a405ad043a7fe3c13fb4).
Only this provenance docstring identifies the authorized source copy.
"""
"""Fresh pinned Git worktree deployment only; paths supplied privately at runtime."""
import hashlib,json,os,re,subprocess,sys,datetime
from pathlib import Path

P3C = {'path':'docs/p3c_validation.md','git_sha256':'5449378c3e2c00d5af4102da73e93a28839898107c139df110f29d50c321c466',
       'contract_sha256':'c89b6466c509628f6482b7d7dbc9f51c202ab665075caf580beed64bb3274b5f',
       'source_path':'/workspace/vkgr-seqm1-p-v1/docs/p3c_validation.md'}

def _p3c(req):
 channel=req.get('p3c_data_channel')
 if type(channel) is not dict or set(channel)!=set(P3C)|{'ops_saved_path'} or any(channel.get(k)!=v for k,v in P3C.items()):
  raise ValueError('closed fixed p3c data-channel mapping required')
 if type(channel['ops_saved_path']) is not str:raise ValueError('fixed private OPS p3c-original save path required')
 saved=Path(channel['ops_saved_path'])
 if not saved.is_absolute() or '..' in saved.parts or saved.is_symlink() or saved.name!='p3c-validation-git-sanitized-v1.md':raise ValueError('fixed private OPS p3c-original save path required')
 if saved.parent!=Path(req['ops']):raise ValueError('p3c original bytes must be saved directly in this OPS directory')
 return channel

def _document_bytes(value):
 try:return json.dumps(value,sort_keys=True,allow_nan=False).encode('utf-8')
 except (TypeError,ValueError) as exc:raise ValueError('new OPS document must be finite JSON') from exc

def _new_documents(req):
 if 'new_ops_documents' not in req:return {}
 documents=req['new_ops_documents']
 if type(documents) is not dict:raise ValueError('new_ops_documents must be a filename-to-JSON dictionary')
 result={}
 for name,value in documents.items():
  if type(name) is not str or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,99}\.json',name) or name=='deployment.json':raise ValueError('safe new JSON basename required; deployment.json is reserved')
  if name=='parameters.json':
   if type(value) is not dict or not value:raise ValueError('parameters.json must be a nonempty absolute-path dictionary')
   for key,path in value.items():
    if type(key) is not str or not key or type(path) is not str or not Path(path).is_absolute() or '..' in Path(path).parts:raise ValueError('parameters.json requires string keys and private absolute paths')
  result[name]=_document_bytes(value)
 return result

def _unique_pairs(pairs):
 result={}
 for key,value in pairs:
  if key in result:raise ValueError('duplicate input JSON field')
  result[key]=value
 return result

def _reject_constant(value):raise ValueError('nonfinite input JSON value')

def _input_pin(req):
 if 'inputs_file_pin' not in req:return None
 pin=req['inputs_file_pin']
 if type(pin) is not dict or set(pin)!={'path','sha256'} or type(pin['path']) is not str or type(pin['sha256']) is not str or not re.fullmatch('[a-f0-9]{64}',pin['sha256']):raise ValueError('inputs_file_pin requires path and sha256')
 path=Path(pin['path'])
 if not path.is_absolute() or '..' in path.parts or path.is_symlink() or not path.is_file():raise ValueError('pinned input must be an existing private absolute nonsymlink JSON file')
 data=path.read_bytes();actual=hashlib.sha256(data).hexdigest()
 if actual!=pin['sha256']:raise ValueError('pinned input JSON sha256 mismatch')
 try:_document_bytes(json.loads(data.decode('utf-8'),object_pairs_hook=_unique_pairs,parse_constant=_reject_constant))
 except (ValueError,UnicodeError) as exc:raise ValueError('pinned input must be valid finite UTF-8 JSON with unique fields') from exc
 return actual

def validate(req):
 expected={'root','source','destination','ops','branch','commit','files','run_ids'}
 if not expected<=set(req) or set(req)-expected-{'new_ops_documents','inputs_file_pin','p3c_data_channel'}:raise ValueError('deployment fields')
 for key in ('root','source','destination','ops'):
  p=Path(req[key])
  if not p.is_absolute() or '..' in p.parts or p.is_symlink():raise ValueError('private absolute nonsymlink paths')
 if not re.fullmatch('codex/[a-z0-9-]+',req['branch']) or not re.fullmatch('[a-f0-9]{40}',req['commit']):raise ValueError('work branch and fixed commit required')
 if not req['files'] or not req['run_ids'] or len(set(req['run_ids']))!=len(req['run_ids']):raise ValueError('empty or duplicate scope')
 for name,pin in req['files'].items():
  p=Path(name)
  if p.is_absolute() or '..' in p.parts or not re.fullmatch('[a-f0-9]{64}',pin):raise ValueError('unsafe file pin')
 for rid in req['run_ids']:
  if not re.fullmatch('[a-z0-9][a-z0-9-]{0,119}',rid):raise ValueError('invalid run identity')
 _new_documents(req);_input_pin(req)
 _p3c(req)

def inventory(req):
 validate(req);root=Path(req['root']);jobs={}
 for rid in req['run_ids']:
  states={s:sorted(p.name for p in (root/'queue'/s).glob('*-'+rid+'.json')) for s in ('pending','claimed','rejected')}
  states['run_exists']=(root/'runs'/rid).exists();jobs[rid]=states
 return {'clear':not any(any(v[k] for k in ('pending','claimed','rejected','run_exists')) for v in jobs.values()) and not Path(req['destination']).exists() and not Path(req['ops']).exists(),'jobs':jobs,'destination_exists':Path(req['destination']).exists(),'ops_exists':Path(req['ops']).exists(),'request_sha256':hashlib.sha256(json.dumps(req,sort_keys=True).encode()).hexdigest()}

def deploy(req,pin):
 before=inventory(req)
 if not before['clear'] or before['request_sha256']!=pin:raise ValueError('fresh read-only inventory required')
 src=Path(req['source']);dest=Path(req['destination']);ops=Path(req['ops'])
 subprocess.run(['git','-C',str(src),'fetch','-q','origin',req['branch']],check=True)
 remote=subprocess.check_output(['git','-C',str(src),'rev-parse','origin/'+req['branch']],text=True).strip()
 if remote!=req['commit']:raise ValueError('published commit mismatch')
 channel=_p3c(req);rel=channel['path'];source=Path(channel['source_path']);saved=Path(channel['ops_saved_path'])
 git_bytes=subprocess.check_output(['git','-C',str(src),'show',req['commit']+':'+rel])
 git_sha=hashlib.sha256(git_bytes).hexdigest()
 if git_sha!=channel['git_sha256']:raise ValueError('p3c Git-original SHA mismatch')
 if source.is_symlink() or not source.is_file():raise ValueError('fixed private p3c data-channel source required')
 source_bytes=source.read_bytes();source_before=hashlib.sha256(source_bytes).hexdigest()
 if source_before!=channel['contract_sha256']:raise ValueError('p3c contract source SHA mismatch')
 subprocess.run(['git','-C',str(src),'worktree','add','-q','--detach',str(dest),req['commit']],check=True)
 for name,pin in req['files'].items():
  p=dest/name
  if p.is_symlink() or hashlib.sha256(p.read_bytes()).hexdigest()!=pin:raise ValueError('deployment file mismatch '+name)
 dirty=subprocess.check_output(['git','-C',str(dest),'status','--porcelain'],text=True).strip()
 if dirty:raise ValueError('deployment is dirty before the fixed p3c overlay')
 ops.mkdir()
 with saved.open('xb') as f:f.write(git_bytes);f.flush();os.fsync(f.fileno())
 saved_sha=hashlib.sha256(saved.read_bytes()).hexdigest()
 if saved_sha!=channel['git_sha256']:raise ValueError('saved p3c Git-original SHA mismatch')
 target=dest/rel
 if target.is_symlink() or not target.is_file():raise ValueError('tracked p3c deployment document required')
 target.write_bytes(source_bytes)
 source_after=hashlib.sha256(source.read_bytes()).hexdigest();target_sha=hashlib.sha256(target.read_bytes()).hexdigest()
 if source_after!=channel['contract_sha256'] or target_sha!=channel['contract_sha256']:raise ValueError('p3c source/overlay changed during copy')
 dirty=subprocess.check_output(['git','-C',str(dest),'status','--porcelain'],text=True).splitlines()
 if dirty!=[' M '+rel]:raise ValueError('only the fixed p3c path may be dirty after data-channel overlay')
 documents=_new_documents(req);inputs_sha=_input_pin(req)
 for name,data in documents.items():
  with (ops/name).open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 record={'commit':req['commit'],'files':req['files'],'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'data_overlay':True,'p3c_data_channel':{**channel,'copy_sha256':{'git_bytes_before_save':git_sha,'ops_saved_after_copy':saved_sha,'source_before_copy':source_before,'source_after_copy':source_after,'deployment_after_copy':target_sha}}}
 if inputs_sha is not None:record['inputs_sha256']=inputs_sha
 if 'parameters.json' in documents:record['parameters_sha256']=hashlib.sha256((ops/'parameters.json').read_bytes()).hexdigest()
 with (ops/'deployment.json').open('x') as f:json.dump(record,f,sort_keys=True);f.flush();os.fsync(f.fileno())
 return {'status':'deployed','commit':req['commit'],'files_checked':len(req['files']),'data_overlay':True,'p3c_data_channel':record['p3c_data_channel']}

def main():
 payload=json.loads(sys.stdin.read(2*1024*1024+1));req=payload['request']
 if payload['action']=='inventory':answer=inventory(req)
 elif payload['action']=='deploy':answer=deploy(req,payload['inventory_pin'])
 else:raise ValueError('unknown action')
 answer['read_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();print(json.dumps(answer,sort_keys=True))
if __name__=='__main__':main()
