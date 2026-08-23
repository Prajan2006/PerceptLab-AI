"""Experiment configuration schema + validation.

An experiment binds dataset → preprocessing → model → protocol → metric
into one serializable, reproducible object. Validation consults the
dataset registry (``config/datasets.json``) and the model registry so
invalid or unimplemented references fail fast — before any training.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from data.config import DATASET_REGISTRY_PATH, PROJECT_ROOT, load_dataset_registry
from models.registry import get_model_spec

SCHEMA_VERSION = 1
_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{2,63}$")


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    root: str | None = None          # optional override; else resolved via registry/env


@dataclass(frozen=True)
class ModelConfig:
    name: str
    params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class PreprocessingConfig:
    recipe: str = "gazehub"
    params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ProtocolConfig:
    type: str                        # 'lopo'
    seed: int = 0


@dataclass(frozen=True)
class EvaluationConfig:
    metric: str                      # 'mean_angular_error_deg'


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    seed: int
    dataset: DatasetConfig
    model: ModelConfig
    preprocessing: PreprocessingConfig
    protocol: ProtocolConfig
    evaluation: EvaluationConfig
    output_dir: str

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "name": self.name,
            "seed": self.seed,
            "dataset": {"name": self.dataset.name, **({"root": self.dataset.root} if self.dataset.root else {})},
            "model": {"name": self.model.name, "params": self.model.params},
            "preprocessing": {"recipe": self.preprocessing.recipe, "params": self.preprocessing.params},
            "protocol": {"type": self.protocol.type, "seed": self.protocol.seed},
            "evaluation": {"metric": self.evaluation.metric},
            "output_dir": self.output_dir,
        }


class ExperimentConfigError(ValueError):
    def __init__(self, errors: list[str]) -> None:
        super().__init__("Invalid experiment configuration: " + "; ".join(errors))
        self.errors = errors


def _expect(condition: bool, errors: list[str], message: str) -> None:
    if not condition:
        errors.append(message)


def parse_experiment_config(payload: dict) -> ExperimentConfig:
    errors: list[str] = []

    name = payload.get("name")
    _expect(isinstance(name, str) and bool(_NAME_PATTERN.match(name)), errors,
            f"name must match {_NAME_PATTERN.pattern}, got {name!r}")

    seed = payload.get("seed", 0)
    _expect(isinstance(seed, int) and 0 <= seed <= 2**32 - 1, errors,
            f"seed must be an int in [0, 2^32), got {seed!r}")

    dataset_payload = payload.get("dataset", {})
    dataset_name = dataset_payload.get("name")
    registered_datasets = set(load_dataset_registry(DATASET_REGISTRY_PATH).get("datasets", {}))
    _expect(dataset_name in registered_datasets, errors,
            f"dataset.name {dataset_name!r} not in config/datasets.json ({sorted(registered_datasets)})")

    model_payload = payload.get("model", {})
    model_name = model_payload.get("name")
    try:
        if isinstance(model_name, str):
            get_model_spec(model_name)
        valid_model = True
    except KeyError:
        valid_model = False
    _expect(valid_model, errors,
            f"model.name {model_name!r} is not registered in models/registry.py")

    preprocess_payload = payload.get("preprocessing", {})
    _expect(preprocess_payload.get("recipe", "gazehub") == "gazehub", errors,
            f"unsupported preprocessing recipe {preprocess_payload.get('recipe')!r} "
            "(only 'gazehub' exists)")

    protocol_payload = payload.get("protocol", {})
    _expect(protocol_payload.get("type") == "lopo", errors,
            f"protocol.type must be 'lopo', got {protocol_payload.get('type')!r}")

    evaluation_payload = payload.get("evaluation", {})
    _expect(evaluation_payload.get("metric") == "mean_angular_error_deg", errors,
            f"evaluation.metric must be 'mean_angular_error_deg', "
            f"got {evaluation_payload.get('metric')!r}")

    output_dir = payload.get("output_dir", "data/experiments")
    _expect(isinstance(output_dir, str) and len(output_dir) > 0, errors,
            "output_dir must be a non-empty string")

    if errors:
        raise ExperimentConfigError(errors)

    return ExperimentConfig(
        name=name,
        seed=seed,
        dataset=DatasetConfig(name=dataset_name, root=dataset_payload.get("root")),
        model=ModelConfig(name=model_name, params=dict(model_payload.get("params", {}))),
        preprocessing=PreprocessingConfig(
            recipe=preprocess_payload.get("recipe", "gazehub"),
            params=dict(preprocess_payload.get("params", {})),
        ),
        protocol=ProtocolConfig(type="lopo", seed=int(protocol_payload.get("seed", seed))),
        evaluation=EvaluationConfig(metric=evaluation_payload.get("metric")),
        output_dir=output_dir,
    )


def load_experiment_config(path: str | os.PathLike[str]) -> ExperimentConfig:
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    payload = json.loads(path.read_text(encoding="utf-8"))
    return parse_experiment_config(payload)
