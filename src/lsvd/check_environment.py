"""Check pinned local versions and the CUDA wheel on the actual machine."""

import argparse
import json
from pathlib import Path
import platform

import duckdb
import faiss
import numpy as np
import threadpoolctl
import torch


def check_environment(*, allow_cpu=False):
    expected = json.loads((Path(__file__).parent / 'deployment/versions.json').read_text())
    versions = dict(python=platform.python_version(), torch=torch.__version__.split('+')[0],
                    numpy=np.__version__, faiss=faiss.__version__, duckdb=duckdb.__version__,
                    threadpoolctl=threadpoolctl.__version__)
    if any(versions[name] != expected[name] for name in versions):
        raise ValueError('installed versions differ from deployment/versions.json')
    available = torch.cuda.is_available()
    if torch.version.cuda is not None and torch.version.cuda != '12.8':
        raise ValueError('CUDA wheel differs from the pinned cu128 build')
    if not available and not allow_cpu:
        raise ValueError('CUDA is unavailable on this machine')
    result = dict(versions=versions, versions_match=True, cuda_available=available,
                  cuda_runtime=torch.version.cuda, platform=platform.system(),
                  architecture=platform.machine(), gpu_check='skipped_cpu', gpu_name=None,
                  arch_list=[], gpu_vram_bytes=None)
    if available:
        runtime = tuple(int(part) for part in torch.version.cuda.split('.')[:2])
        if runtime < (12, 8):
            raise ValueError('CUDA 12.8 or newer is required')
        arches = torch.cuda.get_arch_list()
        if 'sm_120' not in arches:
            raise ValueError('installed torch wheel does not list sm_120')
        if platform.system() != 'Linux' or platform.machine() not in ('x86_64', 'AMD64'):
            raise ValueError('deployment target must be Linux x86_64')
        result.update(gpu_check='verified', gpu_name=torch.cuda.get_device_name(),
                       arch_list=arches,
                       capability=list(torch.cuda.get_device_capability()),
                       gpu_vram_bytes=torch.cuda.get_device_properties(0).total_memory)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-cpu', action='store_true')
    args = parser.parse_args(argv)
    print(json.dumps(check_environment(allow_cpu=args.allow_cpu), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
