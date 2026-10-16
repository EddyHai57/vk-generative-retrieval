"""Read-only queue, raw verified receipts and finite driver observation."""
import json,sys
sys.dont_write_bytecode=True
sys.path.insert(0,'/workspace/vkgr-seqm2-p-v1/scripts')
import seqm2_ops
if __name__=='__main__':print(json.dumps(seqm2_ops.read_status(),sort_keys=True,allow_nan=False))
