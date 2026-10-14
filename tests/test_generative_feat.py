"""GFEAT-01: FEAT-01 F2/F3 side features in G's encoder (new entry only).

Run inside the INFRA-01 6.1 frozen-base harness with OMP_NUM_THREADS=1 (bare ``from test_generative_b1 import ...``
needs ``discover -s tests``). Classes marked "server-safe" do not touch the G1 engineering contract.
"""
import copy
import hashlib
import itertools
import random
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src' / 'vkgr'
P_COMMIT = 'd70a993a800aa4f37a9db55d6133383b95436e35'
BASE_COMMIT = '2c8882b4084a41d2862332a794477ba1d812fae3'
HISTORY_V2_SHA = '1c8d8718ab7527524df9d47dfbb518ee7f5b419ca34dfbb8971aca7efc31f06e'
VENDORED = {
    'history_side': '8faf06bb4dead2a83311bec1edfc84aaea1959fbbbde88871a1697de29cdea73',
    'retrieval_features': 'acc60664458a439823daa99edf1b9e44348ab6ca8783dd284a595404856878fa',
    'retrieval_hp': 'fcc5e8ac18f5fb9278556414de331a1c19ccca7432ec8282cbc657bdc0febadc',
    'retrieval_sasrec': '301b7086af2a2ada58e28c4440d18c880bdd50ddf7ccafc19fd470dcfb80f00c',
    'retrieval_models': 'a482cfde4ec0b6bf5525fa51d3bf226d8c10a4dc4aac785c1440721ee68f30f2',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_blob(commit, rel):
    """Bytes of ``rel`` at ``commit``; None when this checkout's object store lacks the commit."""
    try:
        return subprocess.run(['git', '-C', str(ROOT), 'show', f'{commit}:{rel}'], capture_output=True, check=True).stdout
    except (subprocess.CalledProcessError, OSError):
        return None


# Independent reference of the frozen card (FEAT-01 section 4): written from the card's tables, not from the P code.
CARD_TIMESPENT_BUCKETS = ((1, 3, 1), (4, 9, 2), (10, 30, 3), (31, 60, 4), (61, 134, 5), (135, 254, 6), (255, 255, 7))


def card_timespent_bucket(ts):
    for low, high, bucket in CARD_TIMESPENT_BUCKETS:
        if low <= ts <= high:
            return bucket
    raise AssertionError(ts)


def card_ratio_bucket(ts, dur):
    """Integer comparisons exactly as the card table: r<0.1 | [0.1,0.5) | [0.5,1) | r=1 | r>1."""
    if 10 * ts < dur:
        return 1
    if 2 * ts < dur:
        return 2
    if ts < dur:
        return 3
    if ts == dur:
        return 4
    return 5


class SharedDefinitionTests(unittest.TestCase):
    """server-safe. The feature definitions are ONE source shared with the P line (feat01/p d70a993)."""

    def test_history_v2_identical_to_p_line(self):
        self.assertEqual(sha(SRC / 'history_v2.py'), HISTORY_V2_SHA)
        for commit in (P_COMMIT, BASE_COMMIT):
            blob = git_blob(commit, 'src/vkgr/history_v2.py')
            if blob is not None:
                self.assertEqual((SRC / 'history_v2.py').read_bytes(), blob, commit)

    def test_vendored_modules_identical_to_p_d70a993(self):
        for name, expected in VENDORED.items():
            with self.subTest(module=name):
                self.assertEqual(sha(SRC / f'{name}.py'), expected)
                blob = git_blob(P_COMMIT, f'src/vkgr/{name}.py')
                if blob is not None:
                    self.assertEqual((SRC / f'{name}.py').read_bytes(), blob)

    def test_bucket_tables_match_frozen_card(self):
        from vkgr import history_side
        from vkgr import retrieval_features as rf
        self.assertEqual(rf.TIMESPENT_EDGES, (4, 10, 31, 61, 135, 255))
        self.assertEqual((rf.TIMESPENT_BUCKETS, rf.RATIO_BUCKETS, rf.FEEDBACK_DIM), (7, 5, 6))
        self.assertEqual(history_side.FEEDBACK_BITS, ('like', 'share', 'bookmark', 'open_comments', 'click_on_author', 'dislike'))
        ts = torch.arange(1, 256)
        valid = torch.ones_like(ts, dtype=torch.bool)
        self.assertEqual(rf.timespent_bucket(ts, valid).tolist(), [card_timespent_bucket(v) for v in range(1, 256)])
        self.assertEqual(rf.timespent_bucket(ts, ~valid).tolist(), [0] * 255)
        grid_ts, grid_dur = torch.meshgrid(ts, ts, indexing='ij')
        ones = torch.ones_like(grid_ts, dtype=torch.bool)
        got = rf.ratio_bucket(grid_ts, grid_dur, ones).flatten().tolist()
        want = [card_ratio_bucket(a, b) for a in range(1, 256) for b in range(1, 256)]
        self.assertEqual(got, want)
        self.assertEqual(rf.ratio_bucket(grid_ts, grid_dur, ~ones).flatten().tolist(), [0] * (255 * 255))
        self.assertEqual(sorted(set(want)), [1, 2, 3, 4, 5])


if __name__ == '__main__':
    unittest.main()
