"""Read-only transport entry for fixed M1 queues and driver receipts."""
import hashlib
from pathlib import Path
import sys

p=Path('/workspace/vkgr-seqm1-night-20261016/code/analysis/seqm1_night.py')
if hashlib.sha256(p.read_bytes()).hexdigest()!='9e2b1a2e5114882e25a56233883ccacdebed4235be6f53e457c3eac3d8c7edda':raise ValueError('night module pin mismatch')
sys.path.insert(0,str(p.parent))
from seqm1_night import main
main()
