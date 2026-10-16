"""Read-only real preflight; only a new scratch tree is created, never removed."""
import json
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,'/workspace/vkgr-gtok02-a1-v1/scripts')
import gtok02a1_ops
if __name__=='__main__':print(json.dumps(gtok02a1_ops.preflight(),sort_keys=True,allow_nan=False))
