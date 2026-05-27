from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from src.config import Config
from src.data import build_dataset_from_dir, build_loaders, infer_feature_dims
from src.loss import build_criterion
from src.metrics import compute_metrics, jaccard_samples
from src.model import build_actor, build_basic_lstm, build_critic, soft_update
from src.utils import get_device, save_model, set_seed


def _binarize(probs: torch.Tensor) -> np.ndarray:
    return (probs > 0.5).float().detach().cpu().numpy()


def train_actor_critic_epoch(actor, critic, target_actor, target_critic, actor_optimizer,
                             critic_optimizer, dataloader, device, gamma, tau, epsilon, max_reward):
    """One DDPG-style epoch: TD update for the critic, then a mixed RL + imitation
    update for the actor, followed by Polyak target updates."""
    actor.train()
    critic.train()
    target_actor.eval()
    target_critic.eval()
    criterion = build_criterion("bce")

    total_actor_loss = total_critic_loss = rl_loss = sl_loss = 0.0
    all_preds, all_targets = [], []

    for states, actions, rewards, next_states, dones, diseases, demos in dataloader:
        states = states.float().to(device)
        actions = actions.float().to(device)
        next_states = next_states.float().to(device)
        demos = demos.float().to(device)
        diseases = diseases.to(device)
        rewards = rewards.to(device)
        dones = dones.to(device)

        # --- Critic update (Bellman target from frozen target networks) ---
        with torch.no_grad():
            next_actions = target_actor(next_states, diseases, demos)
            next_q_values = target_critic(next_states, next_actions, diseases, demos)
            expected_q = rewards.unsqueeze(-1) + gamma * (1 - dones.unsqueeze(-1)) * next_q_values
            expected_q = torch.clamp(expected_q, -max_reward, max_reward)

        current_q = critic(states, actions, diseases, demos)
        critic_loss = F.mse_loss(current_q, expected_q)
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()
        total_critic_loss += critic_loss.item()

        # --- Actor update: maximise Q (RL) while imitating doctors (SL) ---
        predicted_actions = actor(states, diseases, demos)
        actor_loss = (-critic(states, predicted_actions, diseases, demos)).mean()
        supervised_loss = criterion(predicted_actions, actions)
        total_loss = (1 - epsilon) * actor_loss + epsilon * supervised_loss

        actor_optimizer.zero_grad()
        total_loss.backward()
        actor_optimizer.step()

        total_actor_loss += total_loss.item()
        rl_loss += actor_loss.item()
        sl_loss += supervised_loss.item()

        # --- Polyak target updates ---
        soft_update(target_critic, critic, tau)
        soft_update(target_actor, actor, tau)

        all_preds.extend(_binarize(predicted_actions))
        all_targets.extend(actions.detach().cpu().numpy())

    n = len(dataloader)
    jaccard = jaccard_samples(np.vstack(all_targets), np.vstack(all_preds))
    return (total_actor_loss / n, total_critic_loss / n, jaccard, rl_loss / n, sl_loss / n)


def train_basic_lstm_epoch(model, optimizer, dataloader, device):
    """One supervised (BCE) imitation epoch for the baseline."""
    model.train()
    criterion = build_criterion("bce")
    total_loss = 0.0
    all_preds, all_targets = [], []

    for states, actions, rewards, next_states, dones, diseases, demos in dataloader:
        states = states.float().to(device)
        actions = actions.float().to(device)
        diseases = diseases.float().to(device)
        demos = demos.float().to(device)

        predicted_actions = model(states, diseases, demos)
        loss = criterion(predicted_actions, actions)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        all_preds.extend(_binarize(predicted_actions))
        all_targets.extend(actions.detach().cpu().numpy())

    metrics = compute_metrics(np.vstack(all_preds), np.vstack(all_targets))
    return total_loss / len(dataloader), metrics["jaccard"], metrics["accuracy"]


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Supervised evaluation of any policy network (actor or baseline)."""
    model.eval()
    criterion = build_criterion("bce")
    total_loss = 0.0
    all_preds, all_targets = [], []

    for states, actions, rewards, next_states, dones, diseases, demos in dataloader:
        states = states.float().to(device)
        actions = actions.float().to(device)
        diseases = diseases.to(device)
        demos = demos.float().to(device)

        predicted_actions = model(states, diseases, demos).float()
        total_loss += criterion(predicted_actions, actions).item()

        all_preds.extend(_binarize(predicted_actions))
        all_targets.extend(actions.detach().cpu().numpy())

    preds, targets = np.vstack(all_preds), np.vstack(all_targets)
    metrics = compute_metrics(preds, targets)
    return total_loss / len(dataloader), metrics["jaccard"], metrics["accuracy"], preds, targets


def fit_actor_critic(cfg: Config, train_loader, val_loader, dims, device) -> dict:
    tc = cfg.train
    ckpt_dir = Path(tc.checkpoint_dir)

    actor = build_actor(cfg.model, dims, cfg.data.time_stamp).to(device)
    critic = build_critic(cfg.model, dims, cfg.data.time_stamp).to(device)
    target_actor = build_actor(cfg.model, dims, cfg.data.time_stamp).to(device)
    target_critic = build_critic(cfg.model, dims, cfg.data.time_stamp).to(device)
    target_actor.load_state_dict(actor.state_dict())
    target_critic.load_state_dict(critic.state_dict())

    actor_optimizer = optim.Adam(actor.parameters(), lr=tc.actor_lr, weight_decay=tc.weight_decay)
    critic_optimizer = optim.Adam(critic.parameters(), lr=tc.critic_lr, weight_decay=tc.weight_decay)

    history = {k: [] for k in ("actor_loss", "critic_loss", "train_jaccard",
                               "rl_loss", "sl_loss", "val_loss", "val_jaccard", "val_accuracy")}
    best_jaccard = 0.0

    for epoch in range(1, tc.epochs + 1):
        actor_loss, critic_loss, train_jaccard, rl_loss, sl_loss = train_actor_critic_epoch(
            actor, critic, target_actor, target_critic, actor_optimizer, critic_optimizer,
            train_loader, device, tc.gamma, tc.tau, tc.epsilon, tc.max_reward,
        )
        val_loss, val_jaccard, val_accuracy, _, _ = evaluate(actor, val_loader, device)

        for key, value in zip(history, (actor_loss, critic_loss, train_jaccard, rl_loss, sl_loss,
                                        val_loss, val_jaccard, val_accuracy)):
            history[key].append(value)

        if val_jaccard > best_jaccard:
            best_jaccard = val_jaccard
            save_model(actor, ckpt_dir / "best_actor.pth")
            save_model(critic, ckpt_dir / "best_critic.pth")
            save_model(target_actor, ckpt_dir / "best_target_actor.pth")
            save_model(target_critic, ckpt_dir / "best_targetcritic.pth")

        if epoch % tc.log_every == 0:
            print(f"[AC] Epoch {epoch}/{tc.epochs}  actor={actor_loss:.4f}  critic={critic_loss:.4f}  "
                  f"train_jac={train_jaccard:.4f}  val_jac={val_jaccard:.4f}  val_acc={val_accuracy:.4f}")

    print(f"[AC] Best validation Jaccard: {best_jaccard:.4f}")
    return history


def fit_basic_lstm(cfg: Config, train_loader, val_loader, dims, device) -> dict:
    tc = cfg.train
    ckpt_dir = Path(tc.checkpoint_dir)

    model = build_basic_lstm(cfg.model, dims, cfg.data.time_stamp).to(device)
    optimizer = optim.Adam(model.parameters(), lr=tc.lstm_lr, weight_decay=tc.weight_decay)

    history = {k: [] for k in ("train_loss", "train_jaccard", "train_accuracy",
                               "val_loss", "val_jaccard", "val_accuracy")}
    best_jaccard = 0.0

    for epoch in range(1, tc.epochs + 1):
        train_loss, train_jaccard, train_accuracy = train_basic_lstm_epoch(
            model, optimizer, train_loader, device)
        val_loss, val_jaccard, val_accuracy, _, _ = evaluate(model, val_loader, device)

        for key, value in zip(history, (train_loss, train_jaccard, train_accuracy,
                                        val_loss, val_jaccard, val_accuracy)):
            history[key].append(value)

        if val_jaccard > best_jaccard:
            best_jaccard = val_jaccard
            save_model(model, ckpt_dir / "best_BL.pth")

        if epoch % tc.log_every == 0:
            print(f"[LSTM] Epoch {epoch}/{tc.epochs}  loss={train_loss:.4f}  "
                  f"train_jac={train_jaccard:.4f}  val_jac={val_jaccard:.4f}  val_acc={val_accuracy:.4f}")

    print(f"[LSTM] Best validation Jaccard: {best_jaccard:.4f}")
    return history


def main(cfg: Config, model_type: str = "actor_critic") -> dict:
    set_seed(cfg.train.seed)
    device = get_device(cfg.train.device)
    print(f"Using device: {device}")

    dataset = build_dataset_from_dir(cfg.data)
    train_loader, val_loader, test_loader = build_loaders(dataset, cfg.data)
    dims = infer_feature_dims(dataset)
    print(f"Feature dims: {dims}  |  dataset size: {len(dataset)}")

    if model_type == "actor_critic":
        return fit_actor_critic(cfg, train_loader, val_loader, dims, device)
    if model_type == "basic_lstm":
        return fit_basic_lstm(cfg, train_loader, val_loader, dims, device)
    raise ValueError(f"Unknown model_type '{model_type}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment_01.yaml")
    parser.add_argument("--model", default="actor_critic", choices=["actor_critic", "basic_lstm"])
    args = parser.parse_args()
    main(Config.from_yaml(args.config), args.model)
