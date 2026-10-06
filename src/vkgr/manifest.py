"""Versioned P1 provenance; hashes prove consistency, not authenticity."""
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import platform
import subprocess
from .config import load_config

ROOT = Path(__file__).resolve().parents[2]
CLAIM = "synthetic scaffold execution; model quality not evaluated"


def read_json(path: Path) -> object:
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate JSON key")
            value[key] = item
        return value
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_fingerprint() -> str:
    paths = [ROOT / name for name in (".python-version", "pyproject.toml", "uv.lock")]
    paths += sorted((ROOT / "src/vkgr").glob("*.py"))
    inputs = "".join(f"{p.relative_to(ROOT).as_posix()}\0{file_hash(p)}\n" for p in paths)
    return hashlib.sha256(inputs.encode()).hexdigest()


def git_context() -> tuple[str, bool]:
    def git(*args):
        return subprocess.run(["git", "-C", str(ROOT), *args], check=True,
                              capture_output=True, text=True).stdout.strip()
    return git("rev-parse", "HEAD"), bool(git("status", "--porcelain", "--untracked-files=all"))


def build_manifest(config_path: Path, result_path: Path, started_at_utc: str, elapsed_seconds: float) -> dict:
    config = load_config(config_path)
    fixture_path = safe_path(config.fixture_path)
    commit, dirty = git_context()
    return {
        "schema_version": 1, "task_id": "P1", "status": "success",
        "git_commit": commit, "git_dirty": dirty, "source_fingerprint": source_fingerprint(),
        "config_path": config_path.resolve().relative_to(ROOT).as_posix(),
        "config_hash": file_hash(config_path), "effective_config": config.effective_config,
        "fixture_path": config.fixture_path, "fixture_hash": file_hash(fixture_path), "seed": config.seed,
        "result_path": result_path.resolve().relative_to(ROOT).as_posix(), "result_hash": file_hash(result_path),
        "python_version": platform.python_version(), "dependency_versions": {"PyYAML": version("PyYAML")},
        "started_at_utc": started_at_utc, "elapsed_seconds": elapsed_seconds, "claim_boundary": CLAIM,
    }


def write_manifest(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def safe_path(value: object) -> Path:
    if type(value) is not str or not re.fullmatch(r"[A-Za-z0-9_./-]+", value):
        raise ValueError("path must be a plain repo-relative path")
    parts = value.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError("path must be canonical and repo-relative")
    path = ROOT
    for part in parts:
        path = path / part
        if path.is_symlink():
            raise ValueError("symlink paths are not allowed")
    if not path.resolve().is_relative_to(ROOT):
        raise ValueError("path escapes repository")
    return path


def verify_manifest(manifest_path: Path) -> None:
    absolute = manifest_path if manifest_path.is_absolute() else ROOT / manifest_path
    try:
        relative = absolute.relative_to(ROOT).as_posix()
    except ValueError:
        raise ValueError("manifest must be inside repository") from None
    manifest_path = safe_path(relative)
    if not re.fullmatch(r"outputs/p1-[a-z0-9_-]+(?:/[a-z0-9_-]+)*/run_manifest\.json", relative):
        raise ValueError("manifest must be run_manifest.json in a P1 output area")
    value = read_json(manifest_path)
    fields = {
        "schema_version", "task_id", "status", "git_commit", "git_dirty", "source_fingerprint",
        "config_path", "config_hash", "effective_config", "fixture_path", "fixture_hash", "seed",
        "result_path", "result_hash", "python_version", "dependency_versions", "started_at_utc",
        "elapsed_seconds", "claim_boundary",
    }
    if type(value) is not dict or set(value) != fields:
        raise ValueError("invalid manifest fields")
    for key, expected in {"schema_version": 1, "task_id": "P1", "status": "success", "claim_boundary": CLAIM}.items():
        if type(value[key]) is not type(expected) or value[key] != expected:
            raise ValueError("unsupported manifest boundary")
    if type(value["git_dirty"]) is not bool:
        raise ValueError("git_dirty must be boolean")
    if type(value["seed"]) is not int or not 0 <= value["seed"] < 2**32:
        raise ValueError("invalid manifest seed")
    for key, length in (("git_commit", 40), ("source_fingerprint", 64), ("config_hash", 64),
                        ("fixture_hash", 64), ("result_hash", 64)):
        if type(value[key]) is not str or not re.fullmatch("[0-9a-f]{" + str(length) + "}", value[key]):
            raise ValueError("invalid manifest fingerprint")
    if type(value["python_version"]) is not str or not re.fullmatch(r"3\.11\.[0-9]+", value["python_version"]):
        raise ValueError("unsupported Python version")
    if value["dependency_versions"] != {"PyYAML": "6.0.2"}:
        raise ValueError("unsupported dependencies")
    if type(value["started_at_utc"]) is not str:
        raise ValueError("invalid UTC start time")
    try:
        started = datetime.fromisoformat(value["started_at_utc"])
    except ValueError:
        raise ValueError("invalid UTC start time") from None
    if started.tzinfo is None or started.utcoffset() != timezone.utc.utcoffset(started):
        raise ValueError("start time must be UTC")
    elapsed = value["elapsed_seconds"]
    if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("invalid elapsed seconds")

    for key in ("config_path", "fixture_path", "result_path"):
        safe_path(value[key])
    for key in ("config", "fixture", "result"):
        try:
            actual = file_hash(safe_path(value[key + "_path"]))
        except OSError:
            raise ValueError("manifest input or result is missing or unreadable") from None
        if actual != value[key + "_hash"]:
            raise ValueError("manifest artifact hash mismatch")
    if value["source_fingerprint"] != source_fingerprint():
        raise ValueError("P1 source fingerprint mismatch")
    if safe_path(value["result_path"]) != manifest_path.parent / "result.json":
        raise ValueError("manifest must reference its own result.json")
    config = load_config(safe_path(value["config_path"]))
    if (json.dumps(value["effective_config"], sort_keys=True) != json.dumps(config.effective_config, sort_keys=True)
            or value["seed"] != config.seed or value["fixture_path"] != config.fixture_path):
        raise ValueError("manifest configuration is inconsistent")
    from .runtime import load_fixture, compute_result
    fixture = load_fixture(safe_path(value["fixture_path"]))
    actual_result = read_json(safe_path(value["result_path"]))
    if json.dumps(actual_result, sort_keys=True) != json.dumps(compute_result(fixture, config.seed), sort_keys=True):
        raise ValueError("result does not match the synthetic seeded computation")

    commit_check = subprocess.run(["git", "-C", str(ROOT), "cat-file", "-t", value["git_commit"]],
                                  capture_output=True, text=True)
    if commit_check.returncode != 0 or commit_check.stdout.strip() != "commit":
        raise ValueError("recorded Git commit is unavailable")
