"""Read-only B1024 A1 queue, raw receipts and finite-driver status."""
import json
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,'/workspace/vkgr-gtok02-a1-v1/scripts')
import gtok02a1_ops
if __name__=='__main__':print(json.dumps(gtok02a1_ops.read_status(),sort_keys=True,allow_nan=False))
