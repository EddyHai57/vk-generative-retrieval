"""P1 smoke and verify entry points."""
import argparse
import logging
import subprocess
import yaml
from pathlib import Path
from .runtime import run_smoke
from .manifest import verify_manifest


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(prog="vkgr")
    commands = parser.add_subparsers(dest="command", required=True)
    smoke = commands.add_parser("smoke")
    smoke.add_argument("--config", type=Path, required=True)
    smoke.add_argument("--output-dir", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "smoke":
            run_smoke(args.config, args.output_dir)
            logging.info("P1 synthetic smoke completed")
        else:
            verify_manifest(args.manifest)
            logging.info("P1 manifest verified")
    except FileExistsError:
        logging.error("output already exists; choose a new P1 run directory")
        return 2
    except OSError:
        logging.error("local input/output is unavailable")
        return 2
    except subprocess.SubprocessError:
        logging.error("Git provenance check failed; run stopped")
        return 2
    except (ValueError, TypeError, yaml.YAMLError):
        logging.error("invalid P1 config, fixture, manifest or path; run rejected")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
