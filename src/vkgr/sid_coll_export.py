"""C2 original M1-mean s42 best-only export. Metadata preflight never loads a checkpoint."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from . import generative_data as gd, sid_tok_audit as au, sid_v2
from .data_audit import canonical_bytes,file_sha256
from .sid_coll_v1 import _hex

TEACHER_COMMIT='0ca61bc35ad68bd79b86103b54e470445cbab07f'
LIMITATIONS=[
 'Only supported videos (approximately 18%) carry the collaborative block; the remaining approximately 82% are content-only. Actual fractions are reported.',
 'Supported and unsupported videos are systematically separated in 128-dimensional space; BHKM may first split by collaborative-block presence, approximately whether a video was liked.']


def compose_c2(p,content,supported):
    """Pure [unit(P),unit(content)] then unit-row, with original supported mask."""
    if not isinstance(content,np.ndarray) or content.dtype!=np.float32 or content.ndim!=2 or content.shape[1]!=64:
        raise ValueError('content must be float32 [N,64]')
    if not isinstance(p,np.ndarray) or p.dtype!=np.float32 or p.shape!=content.shape: raise ValueError('P must be float32 [N,64]')
    if not isinstance(supported,np.ndarray) or supported.dtype!=np.bool_ or supported.shape!=(len(p),): raise ValueError('bool supported mask required')
    if not np.isfinite(content).all() or not np.isfinite(p).all(): raise ValueError('nonfinite P/content STOP')
    cn=np.linalg.norm(content.astype('float64'),axis=1)
    if np.any(cn==0): raise ValueError('missing/zero content STOP')
    pn=np.linalg.norm(p.astype('float64'),axis=1)
    use=supported & (pn>0)
    punit=np.zeros_like(p);punit[use]=(p[use]/pn[use,None]).astype('float32')
    cunit=(content/cn[:,None]).astype('float32')
    x=np.concatenate((punit,cunit),axis=1)
    x=(x/np.linalg.norm(x.astype('float64'),axis=1)[:,None]).astype('float32')
    return x,{'supported_items':int(supported.sum()),'supported_fraction':float(supported.mean()),
      'unsupported_items':int((~supported).sum()),'supported_zero_items':int((supported & (pn==0)).sum()),
      'collaborative_block_items':int(use.sum()),'content_only_items':int((~use).sum()),'limitations':LIMITATIONS}


def _legacy_digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _git(root):
    return subprocess.run(['git','-C',str(root),'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()


def metadata_preflight(teacher_root,run_manifest_path,run_manifest_sha,lock,catalog_ids,content_with_pad,supported_mask,*,engineering=False):
    """Validate exact original source, pinned best selection, catalog and support, without checkpoint reads."""
    root=Path(teacher_root).absolute();gd._no_symlink(root)
    if type(lock) is not dict or lock.get('schema')!='coll-c2-source-lock-v1': raise ValueError('C2 source lock required')
    if lock.get('source_commit')!=TEACHER_COMMIT or _git(root)!=TEACHER_COMMIT: raise ValueError('wrong original teacher source commit')
    files=lock.get('source_files')
    # Pin the complete original package: dynamic imports cannot slip outside the source evidence.
    actual={str(p.relative_to(root)) for p in (root/'src/vkgr').glob('*.py')}
    if type(files) is not dict or set(files)!=actual: raise ValueError('all original teacher package source files must be pinned')
    for name,sha in files.items(): gd._checked_file(root/name,_hex(sha))
    run_path=gd._checked_file(run_manifest_path,run_manifest_sha);value=gd._json(run_path)
    cfg=value.get('config',{})
    if value.get('schema')!='p-seqm1-train-manifest-v1' or value.get('kind')!='p_seqm1_train' or value.get('status')!='complete' or cfg.get('arm')!='mean' or cfg.get('seed')!=42:
        raise ValueError('original complete M1-mean seed42 manifest required')
    best=value.get('best',{})
    if best.get('path')!='best.pt' or best.get('sha256')!=lock.get('checkpoint_sha256'): raise ValueError('not locked original best checkpoint')
    _hex(lock['checkpoint_sha256'])
    if type(best.get('epoch')) is not int or type(best.get('step')) is not int: raise ValueError('best selection evidence required')
    if not isinstance(catalog_ids,np.ndarray) or catalog_ids.dtype!=np.uint32 or catalog_ids.ndim!=1 or np.any(catalog_ids[1:]<=catalog_ids[:-1]): raise ValueError('ordered uint32 catalog required')
    n=len(catalog_ids)
    if not engineering and n!=120415: raise ValueError('formal C2 catalog must have 120415 items')
    if not isinstance(content_with_pad,np.ndarray) or content_with_pad.dtype!=np.float32 or content_with_pad.shape!=(n+1,64) or np.any(content_with_pad[0]): raise ValueError('CONTENT_V1 with PAD required')
    # Compose validates all finite/nonzero content, even before loading teacher.
    compose_c2(np.zeros((n,64),dtype='float32'),content_with_pad[1:],np.zeros(n,dtype=bool))
    content_digest=au.feature_digest(content_with_pad[1:])
    if not engineering and content_digest!=au.CONTENT_V1_FEATURE_SHA256: raise ValueError('CONTENT_V1 identity mismatch')
    if lock.get('catalog_sha256')!=sid_v2.catalog_digest(catalog_ids): raise ValueError('teacher catalog identity mismatch')
    if not isinstance(supported_mask,np.ndarray) or supported_mask.dtype!=np.bool_ or supported_mask.shape!=(n+1,) or supported_mask[0]: raise ValueError('original support bool[N+1] PAD false required')
    if lock.get('supported_sha256')!=hashlib.sha256(supported_mask.tobytes()).hexdigest(): raise ValueError('teacher supported identity mismatch')
    _hex(lock.get('train_sequence_sha256'))
    binding=value.get('binding')
    if type(binding) is not dict: raise ValueError('original teacher binding required')
    if binding.get('catalog_sha256')!=_legacy_digest([int(i) for i in catalog_ids]): raise ValueError('original manifest catalog binding mismatch')
    support_ids=[int(i) for i in catalog_ids[supported_mask[1:]]]
    if binding.get('support_sha256')!=_legacy_digest(support_ids): raise ValueError('original manifest support binding mismatch')
    if binding.get('content_sha256')!=hashlib.sha256(content_with_pad.tobytes()).hexdigest(): raise ValueError('original manifest content binding mismatch')
    recorded=binding.get('source',{}).get('files',{})
    if any(recorded.get(name)!=sha for name,sha in files.items()): raise ValueError('original manifest source ledger differs from teacher')
    return {'status':'metadata_verified','checkpoint_path':run_path.parent/'best.pt','checkpoint_sha256':best['sha256'],
      'binding_sha256':_legacy_digest(binding),'best_epoch':best['epoch'],'best_step':best['step'],
      'content_feature_sha256':content_digest,'catalog_sha256':sid_v2.catalog_digest(catalog_ids)}


_CHILD=r'''
import sys,json,hashlib,io
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import numpy as np
import torch
from vkgr.retrieval_seqm1 import make_seqm1_model
from vkgr.data_audit import canonical_bytes
root,cp,content_path,support_path,out,expect=sys.argv[1:]
expect=json.loads(expect)
content=np.load(content_path,allow_pickle=False);support=np.load(support_path,allow_pickle=False)
# Exactly one checkpoint load. No fallback, no unsafe unpickling, no other paths.
raw=Path(cp).read_bytes()
if hashlib.sha256(raw).hexdigest()!=expect['checkpoint_sha256']: raise ValueError('locked checkpoint bytes changed')
state=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
if state.get('schema')!='p-seqm1-checkpoint-v1' or state.get('binding_sha256')!=expect['binding_sha256'] or state.get('epoch')!=expect['best_epoch'] or state.get('step')!=expect['best_step']:
    raise ValueError('original best checkpoint identity mismatch')
model=make_seqm1_model(torch.from_numpy(content),np.flatnonzero(support).tolist(),50,'mean')
model.load_state_dict(state['model'],strict=True)
if not torch.equal(model.supported,torch.from_numpy(support)): raise ValueError('checkpoint original supported mask differs')
model.eval()
with torch.no_grad():
    p=model.items(torch.arange(1,len(support),dtype=torch.long)).cpu().numpy()
if p.dtype!=np.float32 or p.shape!=(len(support)-1,64): raise ValueError('original items() output shape')
np.save(Path(out)/'p-vectors.npy',p,allow_pickle=False)
'''


def export_c2(teacher_root,run_manifest_path,run_manifest_sha,lock_path,lock_sha,content_path,content_sha,catalog_path,catalog_sha,
              support_path,support_sha,output,*,commit,engineering=False):
    """Export once in an isolated process; retain masked 64D P for input verification without reloading CP."""
    lock=gd._json(gd._checked_file(lock_path,lock_sha))
    content_file=gd._checked_file(content_path,content_sha);support_file=gd._checked_file(support_path,support_sha)
    content=np.load(content_file,allow_pickle=False);support=np.load(support_file,allow_pickle=False)
    raw=gd._json(gd._checked_file(catalog_path,catalog_sha))
    if type(raw) is not list or any(type(i) is not int or not 0<=i<2**32 for i in raw): raise ValueError('catalog JSON list')
    ids=np.asarray(raw,dtype=np.uint32)
    meta=metadata_preflight(teacher_root,run_manifest_path,run_manifest_sha,lock,ids,content,support,engineering=engineering)
    cp=gd._checked_file(meta['checkpoint_path'],meta['checkpoint_sha256']) # hashing only, then one safe torch load in child
    out=Path(output).absolute();gd._no_symlink(out)
    if out.exists() or not out.parent.is_dir(): raise ValueError('new C2 output required')
    _hex(commit,40)
    own_git=gd._git(Path(__file__).resolve().parents[2])
    if own_git['git_base']!=commit or not engineering and own_git['git_dirty']: raise ValueError('export commit must equal actual clean public source')
    out.mkdir()
    env={k:v for k,v in os.environ.items() if k not in ('PYTHONPATH','PYTHONHOME')}
    subprocess.run([sys.executable,'-I','-c',_CHILD,str(Path(teacher_root).absolute()/'src'),str(cp),str(content_file),str(support_file),str(out),json.dumps({k:meta[k] for k in ('binding_sha256','best_epoch','best_step','checkpoint_sha256')})],check=True,env=env,cwd=out)
    p=np.load(out/'p-vectors.npy',allow_pickle=False)
    if np.any(p[~support[1:]]!=0): raise ValueError('original unsupported P must be exactly zero')
    x,report=compose_c2(p,content[1:],support[1:])
    np.save(out/'vectors.npy',x,allow_pickle=False);np.save(out/'supported.npy',support,allow_pickle=False)
    script=file_sha256(Path(__file__))
    manifest={'schema':'coll-input-produced-v1','status':'produced','variant':'C2','dimension':128,'engineering':engineering,
      'catalog_sha256':meta['catalog_sha256'],'vectors_sha256':au.feature_digest(x),'content_feature_sha256':meta['content_feature_sha256'],
      'train_sequence_sha256':lock['train_sequence_sha256'],'script_sha256':script,'commit':commit,'checkpoint_sha256':meta['checkpoint_sha256'],
      'source_files':{name:file_sha256(Path(__file__).resolve().parents[2]/name) for name in ('src/vkgr/sid_coll_export.py','src/vkgr/sid_coll_v1.py','src/vkgr/sid_v2.py','src/vkgr/sid_tok_audit.py','src/vkgr/data_audit.py','src/vkgr/generative_data.py')},'teacher_source_commit':TEACHER_COMMIT,'teacher_source_files':lock['source_files'],
      'original_manifest_sha256':run_manifest_sha,'source_lock_sha256':lock_sha,
      'artifacts':{n:file_sha256(out/n) for n in ('p-vectors.npy','vectors.npy','supported.npy')},'audit':report}
    (out/'producer-manifest.json').write_bytes(canonical_bytes(manifest))
    return manifest


def verify_c2_export(output,producer_sha,content_with_pad,catalog_ids,lock,*,engineering=False):
    """Independent algebra/hash replay of exported P; never opens any checkpoint."""
    from .sid_coll_v1 import validate_input
    out=Path(output).absolute();gd._no_symlink(out)
    m=gd._json(gd._checked_file(out/'producer-manifest.json',producer_sha))
    if m.get('schema')!='coll-input-produced-v1' or m.get('status')!='produced' or m.get('variant')!='C2' or m.get('engineering') is not engineering:
        raise ValueError('original C2 producer identity required')
    if m.get('checkpoint_sha256')!=lock.get('checkpoint_sha256') or m.get('teacher_source_commit')!=TEACHER_COMMIT or m.get('teacher_source_files')!=lock.get('source_files'):
        raise ValueError('C2 export source mismatch')
    if m.get('train_sequence_sha256')!=lock.get('train_sequence_sha256'): raise ValueError('C2 train sequence binding mismatch')
    if set(m.get('artifacts',{}))!={'p-vectors.npy','vectors.npy','supported.npy'}: raise ValueError('C2 artifact set mismatch')
    arrays={name:np.load(gd._checked_file(out/name,sha),allow_pickle=False) for name,sha in m['artifacts'].items()}
    p=arrays['p-vectors.npy'];support=arrays['supported.npy'];x=arrays['vectors.npy']
    n=len(catalog_ids)
    if support.dtype!=np.bool_ or support.shape!=(n+1,) or support[0] or hashlib.sha256(support.tobytes()).hexdigest()!=lock.get('supported_sha256'):
        raise ValueError('export original support mismatch')
    if content_with_pad.dtype!=np.float32 or content_with_pad.shape!=(n+1,64) or np.any(content_with_pad[0]): raise ValueError('content PAD identity')
    if au.feature_digest(content_with_pad[1:])!=m.get('content_feature_sha256'): raise ValueError('export content provenance mismatch')
    if np.any(p[~support[1:]]!=0): raise ValueError('unsupported P must remain exactly zero')
    replay,audit=compose_c2(p,content_with_pad[1:],support[1:])
    if not np.array_equal(x,replay) or m.get('audit')!=audit: raise ValueError('C2 algebra replay differs')
    accepted={**m,'schema':'coll-input-v1','status':'accepted','producer_manifest_sha256':producer_sha}
    validate_input(catalog_ids,x,accepted,engineering=engineering)
    return {'schema':'coll-input-verify-v1','status':'verified','producer_manifest_sha256':producer_sha,
       'checkpoint_sha256':m['checkpoint_sha256'],'vectors_sha256':m['vectors_sha256'],'checks':['original_support','artifact_hashes','CONTENT_V1','block_order','normalization_replay'],
       'input_manifest':accepted}


def main(argv=None):
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--teacher-root',required=True)
    for role in ('run-manifest','lock','content','catalog','support'):
        p.add_argument('--'+role,required=True);p.add_argument('--'+role+'-sha',required=True)
    p.add_argument('--output',required=True);p.add_argument('--commit',required=True);p.add_argument('--engineering',action='store_true')
    a=p.parse_args(argv)
    try:
        m=export_c2(a.teacher_root,a.run_manifest,a.run_manifest_sha,a.lock,a.lock_sha,a.content,a.content_sha,a.catalog,a.catalog_sha,
             a.support,a.support_sha,a.output,commit=a.commit,engineering=a.engineering)
        print(json.dumps({'status':'produced','vectors_sha256':m['vectors_sha256']}));return 0
    except (ValueError,KeyError,TypeError,OSError,RuntimeError,subprocess.CalledProcessError) as exc:
        print('C2 STOP: '+str(exc),file=sys.stderr);return 1

if __name__=='__main__':raise SystemExit(main())
