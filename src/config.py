from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class DataConfig:
    processed_dir: str = "data"
    time_stamp: int = 8
    train_frac: float = 0.8
    val_frac: float = 0.1
    batch_size: int = 64
    num_workers: int = 0
    seed: int = 42


@dataclass
class ModelConfig:
    hidden1_units: int = 40
    hidden2_units: int = 180
    embedding_num: int = 2001
    dropout: float = 0.5


@dataclass
class TrainConfig:
    epochs: int = 30
    actor_lr: float = 1e-3
    critic_lr: float = 5e-3
    lstm_lr: float = 1e-3
    weight_decay: float = 0.0
    gamma: float = 0.99
    tau: float = 0.001
    epsilon: float = 0.5
    max_reward: float = 30.0
    seed: int = 1
    device: str = "cpu"
    checkpoint_dir: str = "output_models"
    log_every: int = 5


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    experiment_name: str = "baseline"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls(
            data=DataConfig(**raw.get("data", {})),
            model=ModelConfig(**raw.get("model", {})),
            train=TrainConfig(**raw.get("train", {})),
            experiment_name=raw.get("experiment_name", "baseline"),
        )
