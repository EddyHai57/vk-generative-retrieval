import json
import subprocess
import sys
import unittest
from pathlib import Path
import tempfile
from vkgr.throughput_cgroups import effective_limits
from vkgr.throughput_probe import recommend


class ProbeTests(unittest.TestCase):
    def test_nested_cgroup_caps_root_and_unlimited_ancestors(self):
        with tempfile.TemporaryDirectory()as t:
            root=Path(t);cg=root/'cg';nested=cg/'rented'/'instance';nested.mkdir(parents=True)
            member=root/'membership';member.write_text('0::/rented/instance\n')
            (cg/'memory.max').write_text('max');(nested.parent/'memory.max').write_text('128000000000');(nested/'memory.max').write_text('64000000000')
            (cg/'cpu.max').write_text('max 100000');(nested/'cpu.max').write_text('1200000 100000')
            self.assertEqual(effective_limits(cg,member),{'cpu_cores':12.,'memory_bytes':64000000000})

    def test_minimum_recommended_and_no_science_fields(self):
        for cores,ram,vram in ((12,64*10**9,24576),(24,128*10**9,49140)):
            record={'cpu_cores':cores,'memory_total_bytes':ram,'disk_total_bytes':500*10**9,'disk_free_bytes':480*10**9,'gpu_total_mib':vram}
            config=recommend(record)
            self.assertEqual(config.decode_workers,1 if vram<45000 else 2)
            self.assertNotIn('batch_size',config.__dict__)
            self.assertLessEqual(config.memory_limit_bytes,ram*.6)
            self.assertTrue(config.use_mmap)

    def test_cli_dry_and_invalid(self):
        base=[sys.executable,'-B','-m','analysis.throughput_probe']
        good=subprocess.run(base+['dry'],capture_output=True,text=True)
        self.assertEqual(good.returncode,0,good.stderr)
        self.assertFalse(json.loads(good.stdout)['probe_executed'])
        bad=subprocess.run(base+['invalid'],capture_output=True,text=True)
        self.assertNotEqual(bad.returncode,0)


if __name__=='__main__':unittest.main()
