"""GTOK-02: the old entry points stay bit-for-bit unchanged.

PINS lists the git blob id of every tracked file under src/vkgr and configs at base 7a3b815 (64 paths, README included).
The table is a literal on purpose: it never calls git, so it also runs on the server. Only the listed paths are checked,
so files added later cannot cause false alarms. The mutation suite proves that this test goes red when a pin or a byte moves.
"""
import hashlib
import unittest
from pathlib import Path

from vkgr import sid_v2

ROOT = Path(__file__).resolve().parents[1]
PINS = {
    'configs/batch1/g-g4-rand-s2024.json': '26136f48cdfd6d5a987ca8ca66d6bda076998661',
    'configs/batch1/g-g4-rand-s2025.json': 'b8f2673c56420d4b3e9a80efdca8ea8d1d9b2d42',
    'configs/batch1/g-g4-rand-s42.json': '4891eb36cdf6a86df3587086a04c3fd0852a6ad8',
    'configs/batch1/g-hist-len-l100-s42.json': 'c4bc338f2c5520728e03e9c6ccbb908b2204c320',
    'configs/batch1/g-hist-len-l20-s42.json': 'b186ca3d7bd385cd8abba778f74b74f6941cfeee',
    'configs/batch1/g-hist-v2-l50-s2024.json': 'e588172f8a6a195f7f845743b1757cd62916af9f',
    'configs/batch1/g-hist-v2-l50-s2025.json': '602b5b4273111ac502ff74131e0387c6b33ea8ad',
    'configs/batch1/g-hist-v2-l50-s42.json': 'eda64cb4461c93f573959bcb0d2c3f5748ee7d2c',
    'configs/generative-core-v1.json': '9b166d62f496e58f42bc764e413ad839094c0e01',
    'configs/gtok01/g-tok-01.json': '5a5e2fe8dddaac77949533173fcada52152fd506',
    'configs/gtok01/g-tok-t1-rqvae-b256-s42.json': 'fb5d4c1d7de474910d6d0cb20f527ca640f7e2c3',
    'configs/gtok01/g-tok-t2-rqvae-b16-s42.json': '80ce84f22e3a5d707dbed8be22f768d1b0f98ef5',
    'configs/gtok01/g-tok-t3-bhkm-b256-s42.json': '63f71338b35603065ee27ea2beb61f128a2e7335',
    'configs/mvp.yaml': 'cc1d73ca877142d38489d50c5a2b391795db1bac',
    'configs/p3.yaml': '25bdd6e3810f63bdd8572d8b2e81d9b7e09414dd',
    'configs/p4.json': 'dcfbd67567be0e2327893f67e11a11979fb7202c',
    'configs/p4d-core-01-v1.json': '68c6adfc5292cafa496237dfd2f48f4bc86dcceb',
    'configs/p4d-core-runtime-v1.json': '16db34c9533496d346a05b588d416879f072adef',
    'configs/p5.json': '2c18158768133681d022f72664957cb287aac0e6',
    'configs/p6.json': 'eaff3a0f3fdac5e0ea132406ded70606e3294721',
    'configs/scale.yaml': '8ec6920d06f7fb999661746a4d2d7bff4a09a2c7',
    'configs/smoke.yaml': 'f66f698a1caf4631fe42c38e61edc1bd95f62fa6',
    'configs/traditional.yaml': '5513bb475b5db3fa6f435fc2b9ea2d4d92d9a048',
    'src/vkgr/README.md': 'fa7c880638e5652e60886de1a2a944b046bc43ab',
    'src/vkgr/__init__.py': 'e785608b91fee2dbd46a925838c9d0dfd3d1a8f8',
    'src/vkgr/__main__.py': 'e1fe524ea2d175fbf91895c757789ced9d00f707',
    'src/vkgr/baseline_run.py': '99fc6e9931551fd6103dd9b644cd26db9671557c',
    'src/vkgr/baselines.py': '5aed2069ea7bc636c9d3bc9fe51c998d72f49dae',
    'src/vkgr/config.py': '2ed5bd247727df8ea8016aaea7f077f07e31ca26',
    'src/vkgr/data_audit.py': 'cf645c0f84468807df685eb8e66fb1497ff79f18',
    'src/vkgr/generative_b1_batched.py': '8c1e8e408256e82cbe1b707ed0f0b2b56832422e',
    'src/vkgr/generative_b1_run.py': '252ba1ee2b7dda814af41cbb5c76a8fd2189dcab',
    'src/vkgr/generative_batched.py': '0a59e57152d357c7ecd83b2118cd585551ab7191',
    'src/vkgr/generative_data.py': 'a267967aa7cf8aa4c2bb6886cd218dd4079d0551',
    'src/vkgr/generative_data_v2.py': '98f9fa92409503cf467feae01101afda2c63e42e',
    'src/vkgr/generative_diagnostics.py': 'a95cf0cf70ecd235f56d3eb5be5e486a24822d35',
    'src/vkgr/generative_evaluation.py': 'f4c02ff3aea8ef1df2332b6cca2c459e9d77e6da',
    'src/vkgr/generative_experiment.py': '1748d6e558341f411ca06b6392da8cae0bbeda41',
    'src/vkgr/generative_foundation.py': 'b85a0d5cfbde1650d7d20d37b918ed5e087dd9bf',
    'src/vkgr/generative_hp.py': '94c5bb0ddf69cdece651dfecf68f50eecf995c04',
    'src/vkgr/generative_pilot.py': 'a340fd83f7e0bc780604d37e48c2b768c32f8cec',
    'src/vkgr/generative_real.py': '712b724e2a1a1a678eeb34ecb282a07e5425f777',
    'src/vkgr/generative_sharded.py': 'b583b45b37b7fc7f03a4a381d345a322fe91db90',
    'src/vkgr/generative_tok_data.py': '933563d80ead447ba10baec245a5ab003d4332d7',
    'src/vkgr/generative_tok_run.py': 'aea175689a51a9b86a0e2029808d33a770798f54',
    'src/vkgr/generative_training.py': 'd5536f8d1d7bbce267f06985dd937d95ff8c8770',
    'src/vkgr/history_v2.py': '638e5e3e948245f8207cc826328c888de8c5911d',
    'src/vkgr/manifest.py': '18445f85c8ebe101c5f2d1b5bd091f90e894f574',
    'src/vkgr/retrieval.py': 'f474688ac2c978c7f85fd20f88381fa326ca7bbe',
    'src/vkgr/retrieval_baselines.py': '6bb047cec1910c4c1a2ae2023f44806bbe94da6a',
    'src/vkgr/retrieval_data.py': '4523b8e40dc12c6ccd365376ac6f3dc4989ab896',
    'src/vkgr/retrieval_experiment.py': 'd5649ffba61e82f9d7c32fafa47a2d084bc9aa51',
    'src/vkgr/retrieval_models.py': '8e34618bc7a4dd459db89c41861b10fdcdd53fda',
    'src/vkgr/retrieval_training.py': '6e5dfd305566f731b04191c786c32db5af2a1ed8',
    'src/vkgr/runtime.py': '3ca6188d9c88e83932d7b25c7742322ec9b0baa3',
    'src/vkgr/sid.py': '7053ea10bdc09b7aac6ac2b75e3b36978ca89f5b',
    'src/vkgr/sid_rqvae.py': 'b02f978e604282f0427e2f195a33b86f58a4da7c',
    'src/vkgr/sid_run.py': '2d5443c4b4fc0a45d99e4c6dbf893fc6b3184c04',
    'src/vkgr/sid_tok_audit.py': '593dcf53b7f0012f0b034baf10b314b23d6180f5',
    'src/vkgr/sid_tok_run.py': 'a7467fcaa7ef7495aa63c166afbf8bc79aa4813e',
    'src/vkgr/sid_v2.py': 'df3cb34450fbe498beb93e4b0b65187601da86dd',
    'src/vkgr/tiger.py': '612839198285cd1cf0b0734d31248cfec5bc4d60',
    'src/vkgr/tiger_run.py': '3e9511b51650be49498ab7534dc44eb6054302f8',
    'src/vkgr/training.py': '9aa25c4708145145d4f39f91008837856127b67c',
}


def read_bytes(path):
    return Path(path).read_bytes()


def blob_id(data):
    return hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()


class OldEntryPins(unittest.TestCase):
    def test_table_has_the_64_base_paths(self):
        self.assertEqual(len(PINS), 64)
        self.assertEqual(sum(p.startswith('src/vkgr/') for p in PINS), 41)
        self.assertEqual(sum(p.startswith('configs/') for p in PINS), 23)
        self.assertIn('src/vkgr/README.md', PINS)

    def test_every_pinned_file_is_byte_identical_to_the_base(self):
        for path, expected in sorted(PINS.items()):
            with self.subTest(path):
                self.assertEqual(blob_id(read_bytes(ROOT / path)), expected)

    def test_old_registry_literals_are_unchanged(self):
        self.assertEqual(sid_v2.SID_VERSIONS, ('SID_RQVAE_B256D3_S42', 'SID_RQVAE_B16D3_S42', 'SID_BHKM_B256D3_S42'))
        self.assertEqual(sid_v2.ENGINEERING_SID_VERSIONS, sid_v2.SID_VERSIONS + ('SID_V2_REPLAY_BHKM_B16D3',))
        self.assertEqual(sid_v2.VERSION_SPEC, {
            'SID_RQVAE_B256D3_S42': ('rqvae', 256), 'SID_RQVAE_B16D3_S42': ('rqvae', 16),
            'SID_BHKM_B256D3_S42': ('bhkm', 256), 'SID_V2_REPLAY_BHKM_B16D3': ('bhkm', 16)})


if __name__ == '__main__':
    unittest.main()
