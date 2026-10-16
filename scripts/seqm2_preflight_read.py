"""Read-only formal real-input preflight for all M2 configurations."""
import json,sys
sys.dont_write_bytecode=True
sys.path.insert(0,'/workspace/vkgr-seqm2-p-v1/scripts')
import seqm2_ops
if __name__=='__main__':print(json.dumps(seqm2_ops.preflight(),sort_keys=True,allow_nan=False))
