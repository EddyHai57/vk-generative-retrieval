"""Read-only B256D2 mapping audit. No model, checkpoint, or real-input fit."""
import collections
import datetime
import hashlib
import io
import json
import sys
from pathlib import Path

MAPPING=None
MAPPING_SHA='505d9c5ec258ea66afb653f7aa2f7be71700bfb186c021166c6a06b8e12d9453'
MANIFEST=None
MANIFEST_SHA='ce565f6874fde90437e2c7595ad3a040d2f7351b04924ae5b31b51eb3647d2e8'

def balance(rows,branching,depth):
    nodes=violations=max_spread=0
    for level in range(depth):
        parents={}
        for row in rows:
            parents.setdefault(tuple(row[:level]),collections.Counter())[row[level]]+=1
        for counts in parents.values():
            nodes+=1
            spread=max(counts.values())-min(counts.values())
            max_spread=max(max_spread,spread)
            violations+=int(len(counts)!=min(branching,sum(counts.values())) or spread>1)
    return dict(balanced=violations==0,nodes=nodes,violations=violations,max_spread=max_spread)

def audit_padding(rows,branching=256):
    if not rows or any(len(r)!=4 or any(type(x)!=int or x<0 for x in r) or r[2]!=0 for r in rows):
        raise ValueError('four columns with zero padding required')
    if len(set(map(tuple,rows)))!=len(rows):raise ValueError('duplicate full codes')
    raw=[[r[0],r[1],r[3]] for r in rows]
    return dict(raw3=balance(raw,branching,3),padded4_B1=balance(rows,branching,3),
                padded4_all_levels=balance(rows,branching,4))

def checked(path,pin):
    if path.is_symlink() or path.stat().st_size>16*1024*1024:raise ValueError('unsafe input')
    data=path.read_bytes()
    if hashlib.sha256(data).hexdigest()!=pin:raise ValueError('input SHA mismatch')
    return data


def configure_paths(paths):
    global MAPPING, MANIFEST
    if not isinstance(paths, dict) or set(paths) != {'mapping'}:
        raise ValueError("exact private path keys required")
    if any(not isinstance(v, str) or not Path(v).is_absolute() or ".." in Path(v).parts for v in paths.values()):
        raise ValueError("absolute private locations required")
    MAPPING = Path(paths['mapping'])
    MANIFEST = MAPPING.parent/'run_manifest.json'


def main():
    configure_paths(json.loads(sys.stdin.read(65537)))
    import numpy as np
    m=json.loads(checked(MANIFEST,MANIFEST_SHA));payload=checked(MAPPING,MAPPING_SHA)
    if m['artifacts']['mapping.npz']!=MAPPING_SHA:raise ValueError('manifest mapping binding mismatch')
    with np.load(io.BytesIO(payload),allow_pickle=False) as z:
        if set(z.files)!={'ids','sids'}:raise ValueError('unexpected mapping members')
        ids=z['ids'];sids=z['sids']
    if ids.shape!=(120415,) or sids.shape!=(120415,4) or not np.issubdtype(sids.dtype,np.integer) or not np.issubdtype(ids.dtype,np.integer) or len(np.unique(ids))!=120415:raise ValueError('frozen catalog/mapping shape mismatch')
    result=audit_padding(sids.tolist())
    print(json.dumps(dict(read_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        inputs={str(MANIFEST):MANIFEST_SHA,str(MAPPING):MAPPING_SHA},items=len(ids),positions_distinct=[len(np.unique(sids[:,k])) for k in range(4)],
        checks=result, gate_changed=False, test_read=False,checkpoint_read=False,new_submit=False,
        interpretation='B1 retains original first-three-column algorithm; padded4_all_levels is supplementary diagnostic only, not a new gate.'),sort_keys=True))

if __name__=='__main__':main()
