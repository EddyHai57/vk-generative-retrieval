"""Read-only own queue/raw receipts; no writes, models or checkpoint access."""
import json,sys
sys.path.insert(0,'/workspace/vkgr-sid-coll-g-v1/scripts')
import sid_coll_control as c
import sid_coll_ops as o
if __name__=='__main__':
 m=c.identity();p=c.OPS/'pipeline-plan.json'
 plan=o.load_plan(p,c.sha(p))
 print(json.dumps({'inventory':c.inventory(),'summary':o.tick(plan,submit=False),'driver':c.load(c.OPS/'controller/driver-pid.json') if (c.OPS/'controller/driver-pid.json').exists() else None},sort_keys=True,allow_nan=False))
