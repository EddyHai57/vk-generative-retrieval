"""Pure arithmetic on the already-authorized aggregate A receipt; no data reads."""
import argparse
import hashlib
import json
import math
from pathlib import Path

REPORT_SHA = '64f7eec73ac04c92b4b1a283e6e6223a8a3b9a2f88e20328b0a0c67e19ced158'


def pairs(histogram):
    return sum(int(count) * 2 * sum(max(int(length)-d, 0) for d in range(1, 6))
               for length, count in histogram.items())


def utilization(n, branching):
    # Exact balanced child sizes, k=min(b,n), reporting allocated/full slots at
    # each semantic level. No choice of branching is made by this calculation.
    nodes = {n: 1}; result = []
    for depth in range(3):
        children = {}; occupied = 0
        for length, count in nodes.items():
            k = min(branching, length); occupied += k * count
            q, r = divmod(length, k)
            if k-r: children[q] = children.get(q, 0) + count * (k-r)
            if r: children[q+1] = children.get(q+1, 0) + count * r
        result.append(occupied / branching**(depth+1)); nodes = children
    return result


def calculate(path):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != REPORT_SHA: raise ValueError('A aggregate receipt identity mismatch')
    report = json.loads(raw); train = report['train']; hist = train['exposure_length']['histogram']
    assert sum(map(int, hist.values())) == train['users']
    assert sum(int(k)*int(v) for k,v in hist.items()) == train['rows']
    per_epoch = pairs(hist); full = 5 * per_epoch
    subset = 173851080  # 5 epochs, experiment_log S1 progress, C1 receipt
    ratio = full/subset
    input_wall = 2335.1391184329987
    verify_wall = 64.40261220932007
    simplified = 10*train['rows'] - 30*train['users']
    return {'schema': 'overnight-arithmetic-v1', 'a_report_sha256': REPORT_SHA,
        'train_rows': train['rows'], 'train_users': train['users'], 'train_catalog': train['items'],
        'short_lengths_1_to_5': {str(k): hist.get(str(k), 0) for k in range(1, 6)},
        'pairs_per_epoch': per_epoch, 'five_epoch_pairs': full,
        'simplified_all_L_ge_5': simplified, 'short_sequence_correction': per_epoch-simplified,
        'batch4096_steps_per_epoch': math.ceil(per_epoch/4096),
        'subset_five_epoch_pairs': subset, 'work_ratio': ratio,
        'subset_input_seconds': input_wall, 'subset_input_verify_seconds': verify_wall,
        'linear_input_hours': input_wall*ratio/3600,
        'linear_verify_hours': verify_wall*ratio/3600,
        'linear_combined_hours': (input_wall+verify_wall)*ratio/3600,
        'assumption': 'same full train exposure sequences, window5, 5 epochs; whole-job time linear in pairs; NOT measured ETA',
        'utilization': {str(k): utilization(train['items'], k) for k in (228,229,230,256)},
        'cube_root': train['items']**(1/3),
        'dense_root_lower_gib': {str(k): 32*train['items']*k/1024**3 for k in (228,229,230,256)}}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--aggregate', required=True); p.add_argument('--output', required=True)
    a=p.parse_args(argv); result=calculate(a.aggregate)
    with Path(a.output).open('x') as f: json.dump(result,f,indent=2)
    print(json.dumps(result)); return 0


if __name__ == '__main__':
    raise SystemExit(main())
