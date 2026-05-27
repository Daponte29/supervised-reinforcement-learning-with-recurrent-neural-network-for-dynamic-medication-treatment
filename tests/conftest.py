from __future__ import annotations

import numpy as np
import pytest

from src.config import DataConfig, ModelConfig
from src.data import MedicalDataset

# Small synthetic feature space so model/data tests run fast and need no real data.
TIME_STAMP = 8
LAB_SIZE = 6
MED_SIZE = 4
DEMO_SIZE = 3
DI_SIZE = 5
EMBEDDING_NUM = 50


def make_admissions(n: int = 20, seed: int = 0) -> dict:
    """Build per-admission dicts shaped like the real pipeline output."""
    rng = np.random.default_rng(seed)
    action, lab, demo, diag, reward, label = {}, {}, {}, {}, {}, {}
    for i in range(n):
        action[i] = rng.integers(0, 2, size=(TIME_STAMP, MED_SIZE)).astype(np.float32)
        lab[i] = rng.standard_normal((TIME_STAMP, LAB_SIZE)).astype(np.float32)
        demo[i] = rng.standard_normal(DEMO_SIZE).astype(np.float32)
        diag[i] = rng.integers(0, EMBEDDING_NUM, size=DI_SIZE)
        outcome = int(i % 2)
        label[i] = outcome
        reward[i] = [(-10.0 if outcome == 1 else 10.0)] * TIME_STAMP
    return {"action": action, "lab": lab, "demo": demo, "diag": diag, "reward": reward, "label": label}


@pytest.fixture
def admissions() -> dict:
    return make_admissions()


@pytest.fixture
def dataset(admissions) -> MedicalDataset:
    a = admissions
    return MedicalDataset(a["diag"], a["demo"], a["lab"], a["action"], a["reward"], a["label"])


@pytest.fixture
def dims() -> dict:
    return {"lab_size": LAB_SIZE, "med_size": MED_SIZE, "demo_size": DEMO_SIZE, "di_size": DI_SIZE}


@pytest.fixture
def model_cfg() -> ModelConfig:
    return ModelConfig(hidden1_units=8, hidden2_units=10, embedding_num=EMBEDDING_NUM, dropout=0.5)


@pytest.fixture
def data_cfg() -> DataConfig:
    return DataConfig(time_stamp=TIME_STAMP, batch_size=4, num_workers=0, seed=0)
