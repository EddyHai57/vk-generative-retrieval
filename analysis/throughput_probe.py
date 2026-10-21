"""Local-only resource/config dry entrypoint; never connects to another machine."""
import argparse
import json
from pathlib import Path
from vkgr.throughput_probe import resource_manifest
from vkgr.throughput_config import Engineering,manifest


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('dry');probe=sub.add_parser('probe');probe.add_argument('--directory',required=True);probe.add_argument('--output',required=True);probe.add_argument('--config')
    config=sub.add_parser('config');config.add_argument('--path',required=True)
    a=p.parse_args(argv)
    if a.command=='dry':value={'status':'ready','probe_executed':False,'network':False,'science_config_changed':False}
    elif a.command=='config':value=manifest({},Engineering.load(a.path))
    else:
        value=resource_manifest(Path(a.directory),Engineering.load(a.config)if a.config else None)
        with Path(a.output).open('x')as f:json.dump(value,f,sort_keys=True,indent=2)
    print(json.dumps(value,sort_keys=True));return 0


if __name__=='__main__':raise SystemExit(main())
