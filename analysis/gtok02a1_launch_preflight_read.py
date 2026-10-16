"""Read-only real-input preflight through the fixed clean launcher."""
import json
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,'/workspace/vkgr-gtok02-a1-launch-v2/code/analysis')
import gtok02a1_launch_bridge
if __name__=='__main__':print(json.dumps(gtok02a1_launch_bridge.action({'action':'preflight'}),sort_keys=True,allow_nan=False))
