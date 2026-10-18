"""Read-only diagnostic for the G5 catalog size/digest gate: prints header fields only (no rows, no candidates) of the P query metadata and the three
G assemblies under runs/g5-fusion-qmeta and runs/g5-fusion-asm-s<seed>. Verifies nothing, changes nothing, writes nothing. stdout only."""
import json
from pathlib import Path
ROOT = Path('/workspace/vkgr-infra01-root')
KEYS = ('schema', 'catalog_size', 'catalog_digest', 'query_count', 'history_len', 'manifests_checked', 'seed', 'run_id', 'thresholds', 'slice_statistics_manifest_sha256')
def header(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        return None
    value = json.loads(path.read_text())
    return {k: value[k] for k in KEYS if k in value}
def main():
    out = {'meta': header(ROOT / 'runs/g5-fusion-qmeta/out/p-query-meta.json')}
    for seed in (42, 2024, 2025):
        out[f'asm{seed}'] = header(ROOT / f'runs/g5-fusion-asm-s{seed}/out/g-assembly.json')
    print(json.dumps(out, sort_keys=True))
if __name__ == '__main__':
    main()
