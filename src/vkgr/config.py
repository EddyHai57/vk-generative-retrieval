"""Load the small P1 smoke configuration."""
from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass(frozen=True)
class SmokeConfig:
    effective_config: dict

    @property
    def seed(self) -> int:
        return self.effective_config["run"]["seed"]

    @property
    def fixture_path(self) -> str:
        return self.effective_config["data"]["fixture_path"]


SCHEMA = {
    "schema_version": 1, "task_id": "P1", "confidentiality": "public",
    "stage": "p1_synthetic_smoke",
    "objective": "validate_the_minimum_auditable_runtime_without_real_data",
    "data": {"dataset": "synthetic_fixture", "fixture_path": "tests/fixtures/p1_synthetic.json"},
    "resources": {"execution": "local_cpu", "network_required": False, "gpu_required": False},
    "run": {"scope": "p1_framework_dry_run_only", "model": "none", "seed": 42},
    "claims": {"model_quality": "not_evaluated", "allowed": "scaffold_execution_only"},
}


def validate_fields(value: object, schema: dict) -> None:
    if type(value) is not dict or value.keys() != schema.keys():
        raise ValueError("config fields do not match P1 schema")
    for key, expected in schema.items():
        if type(expected) is dict:
            validate_fields(value[key], expected)
        elif key == "seed":
            if type(value[key]) is not int or not 0 <= value[key] < 2**32:
                raise ValueError("seed must be an integer in [0, 2**32)")
        elif (type(value[key]) is not type(expected) or value[key] != expected):
            raise ValueError("config value violates P1 boundary")


class UniqueLoader(yaml.SafeLoader):
    """Safe YAML with duplicate mapping keys rejected, including merge keys."""


def unique_mapping(loader, node, deep=False):
    loader.flatten_mapping(node)
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError("duplicate YAML key")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping)


def load_config(path: Path) -> SmokeConfig:
    try:
        value = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueLoader)
    except (yaml.YAMLError, TypeError, RecursionError):
        raise ValueError("invalid YAML configuration") from None
    validate_fields(value, SCHEMA)
    return SmokeConfig(value)
