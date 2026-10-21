"""Tiny artificial computation; no retrieval, training or real dataset."""
import json
import random
from pathlib import Path


def load_fixture(path: Path) -> dict:
    from .manifest import read_json
    value = read_json(path)
    if type(value) is not dict or set(value) != {"schema_version", "origin", "values"}:
        raise ValueError("invalid synthetic fixture fields")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("unsupported fixture version")
    if value["origin"] != "hand_authored_synthetic_not_vk_schema":
        raise ValueError("fixture must be explicitly artificial")
    values = value["values"]
    if type(values) is not list or not 1 <= len(values) <= 16:
        raise ValueError("fixture must contain 1 to 16 artificial integers")
    if any(type(x) is not int or not 0 <= x <= 1000 for x in values):
        raise ValueError("fixture values must be integers in [0, 1000]")
    return value


def compute_result(fixture: dict, seed: int) -> dict:
    rng = random.Random(seed)
    weights = [rng.randint(1, 9) for _ in fixture["values"]]
    return {
        "schema_version": 1,
        "operation": "synthetic_integer_weighted_sum",
        "weights": weights,
        "weighted_sum": sum(x * w for x, w in zip(fixture["values"], weights)),
        "claim_boundary": "synthetic scaffold execution; model quality not evaluated",
    }


def run_smoke(config_path: Path, output_dir: Path) -> Path:
    from datetime import datetime, timezone
    import time
    from .config import load_config
    from .manifest import ROOT, build_manifest, write_manifest, safe_path, git_context, source_fingerprint
    import re
    started = datetime.now(timezone.utc).isoformat()
    clock_start = time.monotonic()
    def confined(path):
        absolute = path if path.is_absolute() else ROOT / path
        try:
            relative = absolute.relative_to(ROOT).as_posix()
        except ValueError:
            raise ValueError("run paths must stay within repository") from None
        return safe_path(relative)
    config_path = confined(config_path)
    output_dir = confined(output_dir)
    relative_output = output_dir.relative_to(ROOT).as_posix()
    if not re.fullmatch(r"outputs/p1-[a-z0-9_-]+(?:/[a-z0-9_-]+)*", relative_output):
        raise ValueError("output must be in a uniquely named outputs/p1- run area")
    config = load_config(config_path)
    result = compute_result(load_fixture(safe_path(config.fixture_path)), config.seed)
    git_context()
    source_fingerprint()
    output_dir.mkdir(parents=True, exist_ok=False)
    result_path = output_dir / "result.json"
    with result_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, sort_keys=True, indent=2) + "\n")
    value = build_manifest(config_path, result_path, started, time.monotonic() - clock_start)
    path = output_dir / "run_manifest.json"
    write_manifest(path, value)
    return path
