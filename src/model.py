from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import ModelConfig

PRELU_INIT = 0.25


def avg_emb(emb: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Average embeddings over the diagnosis axis, optionally masking padding."""
    if mask is None:
        return torch.mean(emb, dim=1)
    mask = mask.float().unsqueeze(-1)
    sum_emb = torch.sum(emb * mask, dim=1)
    sum_mask = torch.clamp(torch.sum(mask, dim=1), min=1e-9)
    return sum_emb / sum_mask


class ActorNetwork(nn.Module):
    """Policy network: maps patient state (labs + demographics + diagnoses) to a
    per-timestep multi-label medication probability."""

    def __init__(self, cfg: ModelConfig, time_stamp: int, med_size: int, lab_size: int,
                 demo_size: int, di_size: int):
        super().__init__()
        self.time_stamp = time_stamp
        h1, h2 = cfg.hidden1_units, cfg.hidden2_units
        self.lstm = nn.LSTM(input_size=lab_size, hidden_size=h2, batch_first=True)
        self.demo_dense = nn.Linear(demo_size, h1)
        self.disease_embedding = nn.Embedding(cfg.embedding_num, h1, padding_idx=0)
        self.dense_time_distributed = nn.Linear(h2 + h1 * 2, med_size)
        self.output_activation = nn.Sigmoid()

    def forward(self, lab_test: torch.Tensor, disease: torch.Tensor, demo: torch.Tensor) -> torch.Tensor:
        demo = demo.float()
        weight = torch.tensor([PRELU_INIT], device=demo.device)
        demo = F.prelu(self.demo_dense(demo), weight=weight)
        demo = demo.unsqueeze(1).repeat(1, self.time_stamp, 1)

        disease_emb = self.disease_embedding(disease.long())
        disease_repeated = avg_emb(disease_emb).unsqueeze(1).repeat(1, self.time_stamp, 1)

        lstm_out, _ = self.lstm(lab_test)

        combined = torch.cat([lstm_out, disease_repeated, demo], dim=-1)
        return self.output_activation(self.dense_time_distributed(combined))


class CriticNetwork(nn.Module):
    """Q-network: estimates the value of a (state, action) pair at each timestep."""

    def __init__(self, cfg: ModelConfig, time_stamp: int, med_size: int, lab_size: int,
                 demo_size: int, di_size: int):
        super().__init__()
        self.time_stamp = time_stamp
        h1, h2 = cfg.hidden1_units, cfg.hidden2_units
        self.dropout_lab_test = nn.Dropout(p=cfg.dropout)
        self.demo_dense = nn.Linear(demo_size, h1)
        self.demo_prelu = nn.PReLU()
        self.disease_embedding = nn.Embedding(cfg.embedding_num, h1, padding_idx=0)
        self.lstm = nn.LSTM(lab_size, h2, batch_first=True)
        self.action_dense_time_distributed = nn.Linear(med_size, h2 + 2 * h1)
        self.final_dense_time_distributed = nn.Linear(h2 + 2 * h1, 1)

    def forward(self, lab_test: torch.Tensor, action: torch.Tensor, disease: torch.Tensor,
                demo: torch.Tensor) -> torch.Tensor:
        demo = demo.float()
        demo_repeated = self.demo_prelu(self.demo_dense(demo)).unsqueeze(1).repeat(1, self.time_stamp, 1)

        disease_emb = self.disease_embedding(disease.long())
        disease_repeated = avg_emb(disease_emb).unsqueeze(1).repeat(1, self.time_stamp, 1)

        lstm_out, _ = self.lstm(self.dropout_lab_test(lab_test))

        action_transformed = self.action_dense_time_distributed(action)
        combined = torch.cat([lstm_out, disease_repeated, demo_repeated], dim=2)
        return self.final_dense_time_distributed(combined + action_transformed)


class BasicLSTM(nn.Module):
    """Supervised baseline with the same inputs as the Actor but trained purely
    with BCE imitation loss. Diagnoses are consumed as dense float features."""

    def __init__(self, cfg: ModelConfig, time_stamp: int, med_size: int, lab_size: int,
                 demo_size: int, di_size: int):
        super().__init__()
        self.time_stamp = time_stamp
        h1, h2 = cfg.hidden1_units, cfg.hidden2_units
        self.lstm = nn.LSTM(input_size=lab_size, hidden_size=h2, batch_first=True)
        self.demo_dense = nn.Linear(demo_size, h1)
        self.disease_dense = nn.Linear(di_size, h1)
        self.register_buffer("prelu_weight", torch.tensor([PRELU_INIT]))
        self.dense_time_distributed = nn.Linear(h2 + h1 * 2, med_size)
        self.output_activation = nn.Sigmoid()

    def forward(self, lab_test: torch.Tensor, disease: torch.Tensor, demo: torch.Tensor) -> torch.Tensor:
        demo = F.prelu(self.demo_dense(demo.float()), weight=self.prelu_weight)
        demo_repeated = demo.unsqueeze(1).repeat(1, self.time_stamp, 1)

        disease = F.prelu(self.disease_dense(disease.float()), weight=self.prelu_weight)
        disease_repeated = disease.unsqueeze(1).repeat(1, self.time_stamp, 1)

        lstm_out, _ = self.lstm(lab_test)

        combined = torch.cat([lstm_out, disease_repeated, demo_repeated], dim=-1)
        return self.output_activation(self.dense_time_distributed(combined))


@torch.no_grad()
def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """Polyak update: target = (1 - tau) * target + tau * source."""
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def build_actor(cfg: ModelConfig, dims: dict, time_stamp: int) -> ActorNetwork:
    return ActorNetwork(cfg, time_stamp, dims["med_size"], dims["lab_size"],
                        dims["demo_size"], dims["di_size"])


def build_critic(cfg: ModelConfig, dims: dict, time_stamp: int) -> CriticNetwork:
    return CriticNetwork(cfg, time_stamp, dims["med_size"], dims["lab_size"],
                         dims["demo_size"], dims["di_size"])


def build_basic_lstm(cfg: ModelConfig, dims: dict, time_stamp: int) -> BasicLSTM:
    return BasicLSTM(cfg, time_stamp, dims["med_size"], dims["lab_size"],
                     dims["demo_size"], dims["di_size"])
