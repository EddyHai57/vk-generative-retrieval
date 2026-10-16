"""Read-only accepted raw receipts and new driver status."""
import sys,json
sys.dont_write_bytecode=True
sys.path.insert(0,'/workspace/vkgr-sid-coll-g-v2/scripts')
import sid_coll_retry_control as c
if __name__=='__main__':print(json.dumps(c.status(),sort_keys=True,allow_nan=False))
