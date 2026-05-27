from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from src.config import DataConfig

# Default filenames produced by the data_preprocessing pipeline.
MATRIX_FILES = {
    "action": "treatment_matrices.pickle",
    "lab": "time_series_matrices_filtered.pickle",
    "demo": "static_matrices_filtered.pickle",
    "diag": "diagnosis_matrices.pickle",
}
LABELS_FILE = "filtered_df_admission_LABELS.pkl"


def load_matrices(processed_dir: str | Path) -> dict:
    """Load the pickled per-admission matrices and outcome labels.

    Returns a dict with keys: action, lab, demo, diag, label.
    """
    processed_dir = Path(processed_dir)

    def _load_pickle(name: str):
        with open(processed_dir / name, "rb") as handle:
            return pickle.load(handle)

    matrices = {key: _load_pickle(fname) for key, fname in MATRIX_FILES.items()}
    matrices["label"] = pd.read_pickle(processed_dir / LABELS_FILE)
    return matrices


def align_and_filter_data(action, lab, demo, label, diag):
    """Intersect HADM_IDs across all sources and build a dense reward signal.

    Reward shaping: every step of a surviving admission (label 0) gets +10, every
    step of a deceased admission (label 1) gets -10. This guides the RL agent
    toward action sequences associated with survival.
    """
    if not isinstance(label, dict):
        label = pd.Series(label.LABEL.values, index=label.HADM_ID).to_dict()

    hadm_set = set(action.keys()).intersection(
        set(lab.keys()),
        set(demo.keys()),
        set(label.keys()),
        set(diag.keys()),
    )

    def filter_dict(d, keys):
        return dict(sorted({k: d[k] for k in keys if k in d}.items()))

    action = filter_dict(action, hadm_set)
    lab = filter_dict(lab, hadm_set)
    demo = filter_dict(demo, hadm_set)
    label = filter_dict(label, hadm_set)
    diag = filter_dict(diag, hadm_set)

    reward = {}
    for hadm_id, actions_list in action.items():
        if label[hadm_id] == 1:
            reward[hadm_id] = [-10.0] * len(actions_list)
        elif label[hadm_id] == 0:
            reward[hadm_id] = [10.0] * len(actions_list)
        else:
            reward[hadm_id] = [0.0] * len(actions_list)

    return action, lab, demo, label, diag, reward


def align_lab_to_action(action: dict, lab: dict) -> dict:
    """Force each lab sequence to the same length as its action sequence.

    Longer lab sequences are truncated from the end; shorter ones are
    front-padded with zero rows.
    """
    fixed = {}
    for k, v in lab.items():
        v = np.asarray(v)
        target = len(action[k])
        if len(v) > target:
            fixed[k] = v[:target]
        elif len(v) < target:
            zeros_to_add = np.zeros((target - len(v), v.shape[1]))
            fixed[k] = np.concatenate((zeros_to_add, v), axis=0)
        else:
            fixed[k] = v
    return fixed


def truncate_to_window(action, lab, demo, label, diag, reward, time_stamp: int):
    """Keep admissions longer than ``time_stamp`` and trim sequences to the
    trailing ``time_stamp`` steps."""
    action_t = {k: v[-time_stamp:] for k, v in action.items() if np.asarray(v).shape[0] > time_stamp}
    keys = action_t.keys()
    lab_t = {k: lab[k][-time_stamp:] for k in keys}
    reward_t = {k: reward[k][-time_stamp:] for k in keys}
    demo_t = {k: demo[k] for k in keys}
    label_t = {k: label[k] for k in keys}
    diag_t = {k: diag[k] for k in keys}
    return action_t, lab_t, demo_t, label_t, diag_t, reward_t


def prepare_admissions(matrices: dict, time_stamp: int):
    """Run the full alignment → length-fix → windowing pipeline.

    Returns aligned dicts keyed by HADM_ID: action, lab, demo, label, diag, reward.
    """
    action, lab, demo, label, diag, reward = align_and_filter_data(
        matrices["action"], matrices["lab"], matrices["demo"], matrices["label"], matrices["diag"]
    )
    lab = align_lab_to_action(action, lab)
    return truncate_to_window(action, lab, demo, label, diag, reward, time_stamp)


class MedicalDataset(Dataset):
    """One sample per admission: a fixed-length window of states/actions plus
    static disease and demographic vectors and a per-step reward/done signal.
    """

    def __init__(self, disease_dic, demo_dic, lab_result, actions, rewards, label):
        self.dones = [[0] * (len(a) - 1) + [1] for a in actions.values()]
        self.diseases = list(disease_dic.values())
        self.demos = list(demo_dic.values())
        self.states = list(lab_result.values())
        self.actions = list(actions.values()) if isinstance(actions, dict) else actions
        self.rewards = list(rewards.values())

    def __len__(self) -> int:
        return len(self.demos)

    def __getitem__(self, idx: int):
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.dones[idx],
            self.diseases[idx],
            self.demos[idx],
        )


def my_collate_fn(batch):
    """Stack a batch and derive next-states by shifting each state sequence one
    step and appending a terminal zero timestep."""
    states, actions, rewards, dones, diseases, demos = zip(*batch)

    states = torch.stack([torch.as_tensor(s, dtype=torch.float32) for s in states])
    actions = torch.stack([torch.as_tensor(a, dtype=torch.float32) for a in actions])
    diseases = torch.stack([torch.as_tensor(d) for d in diseases])
    demos = torch.stack([torch.as_tensor(d) for d in demos])
    rewards = pad_sequence([torch.as_tensor(r, dtype=torch.float32) for r in rewards], batch_first=True)
    dones = pad_sequence([torch.as_tensor(d, dtype=torch.float32) for d in dones], batch_first=True)

    next_states = []
    for s in states:
        zero_timestamp = torch.zeros_like(s[0])
        next_states.append(torch.cat((s[:-1], zero_timestamp.unsqueeze(0)), dim=0))
    next_states = torch.stack(next_states)

    return states, actions, rewards, next_states, dones, diseases, demos


def infer_feature_dims(dataset: Dataset) -> dict[str, int]:
    """Read feature widths from a single sample (states, actions, _, _, diseases, demos)."""
    states, actions, _, _, diseases, demos = dataset[0]
    return {
        "lab_size": int(np.asarray(states).shape[1]),
        "med_size": int(np.asarray(actions).shape[1]),
        "di_size": int(np.asarray(diseases).shape[0]),
        "demo_size": int(np.asarray(demos).shape[0]),
    }


def split_dataset(dataset: Dataset, cfg: DataConfig) -> tuple[Subset, Subset, Subset]:
    train_size = int(cfg.train_frac * len(dataset))
    val_size = int(cfg.val_frac * len(dataset))
    test_size = len(dataset) - train_size - val_size
    generator = torch.Generator().manual_seed(cfg.seed)
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size], generator=generator)
    return train_ds, val_ds, test_ds


def build_loaders(dataset: Dataset, cfg: DataConfig):
    """Split a MedicalDataset 80/10/10 and wrap each split in a DataLoader."""
    train_ds, val_ds, test_ds = split_dataset(dataset, cfg)
    common = dict(batch_size=cfg.batch_size, collate_fn=my_collate_fn, num_workers=cfg.num_workers)
    train_loader = DataLoader(train_ds, shuffle=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)
    return train_loader, val_loader, test_loader


def build_dataset_from_dir(cfg: DataConfig) -> MedicalDataset:
    """Load matrices from disk and assemble a MedicalDataset."""
    matrices = load_matrices(cfg.processed_dir)
    action, lab, demo, label, diag, reward = prepare_admissions(matrices, cfg.time_stamp)
    return MedicalDataset(diag, demo, lab, action, reward, label)
