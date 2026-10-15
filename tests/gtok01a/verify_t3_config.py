#!/usr/bin/env python3
"""Verify T3 config matches A1 config except for sid/schema/task_id.

Assertion: g-gtok-a1-t3-b256d3-s42.json and g-g6-a1-bhkm-b16d4-s42.json
should be identical except for:
- sid: SID_BHKM_B256D3_S42 vs SID_BHKM_B16D4_S42
- task_id: G-TOK-01A vs G-HP-01-G6
- schema: (should be same g-g6-run-v1)

All other fields (model structure, hyperparams, gates, seed) must match exactly.
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
T3_CONFIG = REPO_ROOT / 'configs' / 'gtok01a' / 'g-gtok-a1-t3-b256d3-s42.json'
A1_CONFIG = REPO_ROOT / 'configs' / 'g6prep' / 'g-g6-a1-bhkm-b16d4-s42.json'

EXPECTED_DIFFS = {'sid', 'task_id'}  # schema should be same


def main():
    if not T3_CONFIG.exists():
        print(f"✗ T3 config not found: {T3_CONFIG}")
        return 1
    
    if not A1_CONFIG.exists():
        print(f"✗ A1 config not found: {A1_CONFIG}")
        return 1
    
    t3 = json.loads(T3_CONFIG.read_text())
    a1 = json.loads(A1_CONFIG.read_text())
    
    print(f"T3 config: {T3_CONFIG.name}")
    print(f"A1 config: {A1_CONFIG.name}")
    print()
    
    # Check keys are identical
    t3_keys = set(t3.keys())
    a1_keys = set(a1.keys())
    
    if t3_keys != a1_keys:
        print("✗ Key sets differ:")
        print(f"  T3 only: {t3_keys - a1_keys}")
        print(f"  A1 only: {a1_keys - t3_keys}")
        return 1
    
    print(f"✓ Both configs have {len(t3_keys)} fields")
    print()
    
    # Compare each field
    diffs = {}
    for key in sorted(t3_keys):
        if t3[key] != a1[key]:
            diffs[key] = (t3[key], a1[key])
    
    print("=== Differences ===")
    if not diffs:
        print("✗ No differences found (expected sid/task_id to differ)")
        return 1
    
    for key, (t3_val, a1_val) in sorted(diffs.items()):
        print(f"{key:20} | T3: {t3_val!r:30} | A1: {a1_val!r}")
    
    print()
    
    # Verify expected differences
    diff_keys = set(diffs.keys())
    
    if diff_keys != EXPECTED_DIFFS:
        print(f"✗ Unexpected differences:")
        print(f"  Expected: {EXPECTED_DIFFS}")
        print(f"  Actual:   {diff_keys}")
        print(f"  Extra:    {diff_keys - EXPECTED_DIFFS}")
        print(f"  Missing:  {EXPECTED_DIFFS - diff_keys}")
        return 1
    
    # Verify specific values
    errors = []
    
    if t3['sid'] != 'SID_BHKM_B256D3_S42':
        errors.append(f"T3 sid should be 'SID_BHKM_B256D3_S42', got {t3['sid']!r}")
    
    if t3['task_id'] != 'G-TOK-01A':
        errors.append(f"T3 task_id should be 'G-TOK-01A', got {t3['task_id']!r}")
    
    if t3['schema'] != 'g-g6-run-v1':
        errors.append(f"T3 schema should be 'g-g6-run-v1', got {t3['schema']!r}")
    
    if a1['sid'] != 'SID_BHKM_B16D4_S42':
        errors.append(f"A1 sid should be 'SID_BHKM_B16D4_S42', got {a1['sid']!r}")
    
    if a1['task_id'] != 'G-HP-01-G6':
        errors.append(f"A1 task_id should be 'G-HP-01-G6', got {a1['task_id']!r}")
    
    if errors:
        print("✗ Value checks failed:")
        for err in errors:
            print(f"  - {err}")
        return 1
    
    # Verify A1 model structure (论文规模)
    print("=== A1 structure verification ===")
    assert a1['encoder_layers'] == 4, f"A1 encoder should be 4, got {a1['encoder_layers']}"
    assert a1['decoder_layers'] == 4, f"A1 decoder should be 4, got {a1['decoder_layers']}"
    assert a1['d_model'] == 128, f"A1 d_model should be 128, got {a1['d_model']}"
    assert a1['heads'] == 6, f"A1 heads should be 6, got {a1['heads']}"
    assert a1['head_dim'] == 64, f"A1 head_dim should be 64, got {a1['head_dim']}"
    assert a1['ff'] == 1024, f"A1 ff should be 1024, got {a1['ff']}"
    assert a1['dropout'] == 0.1, f"A1 dropout should be 0.1, got {a1['dropout']}"
    assert a1['learning_rate'] == 0.0003, f"A1 LR should be 3e-4, got {a1['learning_rate']}"
    print("✓ A1 structure matches 论文规模 (4+4 layers, d128, 6×64 heads, FF1024, LR 3e-4)")
    
    # Verify T3 matches A1 structure
    print()
    print("=== T3 matches A1 structure ===")
    for key in ['encoder_layers', 'decoder_layers', 'd_model', 'heads', 'head_dim', 
                'ff', 'dropout', 'learning_rate', 'beam', 'history_len', 
                'effective_batch', 'microbatch', 'max_epochs', 'min_epochs', 
                'patience', 'weight_decay', 'gradient_clip_norm', 'init', 
                'precision', 'test_gate', 'seed', 'history_version']:
        assert t3[key] == a1[key], f"{key} differs: T3={t3[key]}, A1={a1[key]}"
    print("✓ T3 inherits A1 structure (all hyperparams match)")
    
    print()
    print("✓ Config verification passed:")
    print(f"  - T3 config differs from A1 only in sid ({t3['sid']}) and task_id ({t3['task_id']})")
    print(f"  - All model structure and training hyperparams are identical")
    print(f"  - Schema remains {t3['schema']}")
    
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
