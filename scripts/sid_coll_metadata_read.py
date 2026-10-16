"""Deployed S1 metadata/source-only preflight. No checkpoint/data partition reads."""
import json,sys
sys.path.insert(0,'/workspace/vkgr-sid-coll-g-v1/scripts')
import sid_coll_control as c
if __name__=='__main__':print(json.dumps({'inventory':c.inventory(),'teacher_metadata':c.metadata()},sort_keys=True,allow_nan=False))
