"""Readonly: summarize unittest failures in CFG work_dir/g-test-* logs: error headers, top error lines, and the
failure blocks NOT caused by /private/tmp. No writes. (DEPLOY-01 deploy01_gtest_errors_read.py + deploy01_gtest_errors2_read.py)"""
import collections,json,re
from pathlib import Path
out={}
for d in sorted(Path(CFG['server']['work_dir']).glob('g-test-*')):
    t=(d/'stderr.log').read_text(errors='replace') if (d/'stderr.log').exists() else ''
    heads=re.findall(r'^(ERROR|FAIL): (.+)$',t,re.M)
    last=collections.Counter(re.findall(r'^(\w+(?:Error|Exception)[^\n]{0,160})$',t,re.M))
    blocks=re.split(r'\n={70}\n',t)
    out[d.name]={'bytes':len(t),'n_heads':len(heads),'heads':heads[:12],'top_error_lines':last.most_common(12),
        'blocks':len(blocks)-1,'private_tmp_blocks':sum('/private/tmp' in b for b in blocks[1:]),'other':[b[:3000] for b in blocks[1:] if '/private/tmp' not in b]}
print(json.dumps(out,indent=1))
