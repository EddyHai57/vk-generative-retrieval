#!/usr/bin/env python3
"""Standalone verification script for T3 whitelist (no import dependencies).

RED phase: Verify T3 is NOT in whitelist before changes.
GREEN phase: Verify T3 IS in whitelist after changes.
Regression: Verify existing entries unchanged.
"""
import re
import sys
from pathlib import Path

# Worktree root
REPO_ROOT = Path(__file__).parent.parent.parent
SID_V2D_PATH = REPO_ROOT / 'src' / 'vkgr' / 'sid_v2d.py'


def extract_constant(source, pattern):
    """Extract a Python constant using regex."""
    match = re.search(pattern, source, re.MULTILINE | re.DOTALL)
    if not match:
        raise ValueError(f'Pattern not found: {pattern}')
    return match.group(1)


def parse_tuple(s):
    """Parse a Python tuple literal."""
    s = s.strip()
    if not (s.startswith('(') and s.endswith(')')):
        raise ValueError(f'Not a tuple: {s}')
    # Simple split by comma (assumes no nested structures)
    items = [item.strip().strip("'\"") for item in s[1:-1].split(',') if item.strip()]
    return tuple(items)


def parse_dict(s):
    """Parse a Python dict literal (simple case)."""
    s = s.strip()
    if not (s.startswith('{') and s.endswith('}')):
        raise ValueError(f'Not a dict: {s}')
    result = {}
    # Extract key: value pairs
    for match in re.finditer(r"'([^']+)':\s*\(([^)]+)\)", s):
        key = match.group(1)
        value_str = match.group(2)
        # Parse tuple value
        parts = [p.strip().strip("'\"") for p in value_str.split(',')]
        value = (parts[0], int(parts[1]), int(parts[2]))
        result[key] = value
    return result


def main():
    source = SID_V2D_PATH.read_text()
    
    # Extract constants
    sid_versions_str = extract_constant(source, r"SID_VERSIONS\s*=\s*(\([^)]+\))")
    engineering_versions_str = extract_constant(
        source, r"ENGINEERING_SID_VERSIONS\s*=\s*SID_VERSIONS\s*\+\s*(\([^)]+\))"
    )
    version_spec_str = extract_constant(source, r"VERSION_SPEC\s*=\s*(\{[^}]+\})")
    
    # Parse
    sid_versions = parse_tuple(sid_versions_str)
    engineering_extra = parse_tuple(engineering_versions_str)
    engineering_versions = sid_versions + engineering_extra
    version_spec = parse_dict(version_spec_str)
    
    print(f"SID_VERSIONS: {sid_versions}")
    print(f"ENGINEERING_SID_VERSIONS extra: {engineering_extra}")
    print(f"ENGINEERING_SID_VERSIONS full: {engineering_versions}")
    print(f"VERSION_SPEC keys: {sorted(version_spec.keys())}")
    print()
    
    # --- Regression checks (always pass) ---
    print("=== Regression checks ===")
    
    # Existing formal versions
    expected_formal = ('SID_BHKM_B16D4_S42', 'SID_RQVAE_B256D4_S42')
    assert sid_versions == expected_formal, f"SID_VERSIONS changed: {sid_versions}"
    print("✓ SID_VERSIONS unchanged")
    
    # Existing engineering-only (B16D3)
    assert 'SID_V2D_REPLAY_BHKM_B16D3' in engineering_versions, "B16D3 missing"
    print("✓ B16D3 still in ENGINEERING_SID_VERSIONS")
    
    # Existing VERSION_SPEC entries
    assert version_spec['SID_BHKM_B16D4_S42'] == ('bhkm', 16, 4), "B16D4 spec changed"
    assert version_spec['SID_RQVAE_B256D4_S42'] == ('rqvae', 256, 4), "RQ256D4 spec changed"
    assert version_spec['SID_V2D_REPLAY_BHKM_B16D3'] == ('bhkm', 16, 3), "B16D3 spec changed"
    print("✓ Existing VERSION_SPEC entries unchanged")
    
    # --- Phase-dependent checks ---
    t3_in_formal = 'SID_BHKM_B256D3_S42' in sid_versions
    t3_in_engineering = 'SID_BHKM_B256D3_S42' in engineering_versions
    t3_in_spec = 'SID_BHKM_B256D3_S42' in version_spec
    
    print()
    print("=== T3 status ===")
    print(f"T3 in SID_VERSIONS: {t3_in_formal}")
    print(f"T3 in ENGINEERING_SID_VERSIONS: {t3_in_engineering}")
    print(f"T3 in VERSION_SPEC: {t3_in_spec}")
    
    # Determine phase
    if not t3_in_engineering and not t3_in_spec:
        # RED phase
        print()
        print("✓ RED phase verified: T3 NOT in whitelist (as expected before changes)")
        assert not t3_in_formal, "T3 should not be in formal versions"
        print("✓ T3 not in formal versions (筛选候选，非正式生产)")
        return 0
        
    elif t3_in_engineering and t3_in_spec:
        # GREEN phase
        print()
        print("=== GREEN phase checks ===")
        
        # T3 should be engineering-only, not formal
        assert not t3_in_formal, "T3 should NOT be in formal SID_VERSIONS"
        print("✓ T3 not in formal versions (engineering-only)")
        
        # T3 should be in engineering
        assert t3_in_engineering, "T3 should be in ENGINEERING_SID_VERSIONS"
        print("✓ T3 in ENGINEERING_SID_VERSIONS")
        
        # T3 spec should be correct
        assert version_spec['SID_BHKM_B256D3_S42'] == ('bhkm', 256, 3), \
            f"T3 spec wrong: {version_spec['SID_BHKM_B256D3_S42']}"
        print("✓ T3 spec correct: ('bhkm', 256, 3)")
        
        print()
        print("✓ GREEN phase verified: T3 correctly added to whitelist")
        return 0
        
    else:
        # Inconsistent state
        print()
        print("✗ INCONSISTENT: T3 partially added")
        print(f"  - in ENGINEERING_SID_VERSIONS: {t3_in_engineering}")
        print(f"  - in VERSION_SPEC: {t3_in_spec}")
        return 1


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
