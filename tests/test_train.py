import math

import torch

from src.config import Config, TrainConfig
from src.data import build_loaders, infer_feature_dims
from src.model import build_actor, build_basic_lstm, build_critic
from src.train import evaluate, fit_actor_critic, train_actor_critic_epoch, train_basic_lstm_epoch
from tests.conftest import TIME_STAMP


def test_actor_critic_epoch_runs(dataset, data_cfg, model_cfg, dims):
    device = torch.device("cpu")
    train_loader, _, _ = build_loaders(dataset, data_cfg)

    actor = build_actor(model_cfg, dims, TIME_STAMP).to(device)
    critic = build_critic(model_cfg, dims, TIME_STAMP).to(device)
    target_actor = build_actor(model_cfg, dims, TIME_STAMP).to(device)
    target_critic = build_critic(model_cfg, dims, TIME_STAMP).to(device)
    target_actor.load_state_dict(actor.state_dict())
    target_critic.load_state_dict(critic.state_dict())

    actor_opt = torch.optim.Adam(actor.parameters(), lr=1e-3)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=1e-3)

    actor_loss, critic_loss, jaccard, rl, sl = train_actor_critic_epoch(
        actor, critic, target_actor, target_critic, actor_opt, critic_opt,
        train_loader, device, gamma=0.99, tau=0.001, epsilon=0.5, max_reward=30.0,
    )
    for value in (actor_loss, critic_loss, rl, sl):
        assert math.isfinite(value)
    assert 0.0 <= jaccard <= 1.0


def test_basic_lstm_epoch_and_evaluate(dataset, data_cfg, model_cfg, dims):
    device = torch.device("cpu")
    train_loader, val_loader, _ = build_loaders(dataset, data_cfg)

    model = build_basic_lstm(model_cfg, dims, TIME_STAMP).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    loss, jaccard, accuracy = train_basic_lstm_epoch(model, optimizer, train_loader, device)
    assert math.isfinite(loss)
    assert 0.0 <= jaccard <= 1.0 and 0.0 <= accuracy <= 1.0

    val_loss, val_jac, val_acc, preds, targets = evaluate(model, val_loader, device)
    assert math.isfinite(val_loss)
    assert preds.shape == targets.shape


def test_fit_actor_critic_history(dataset, data_cfg, model_cfg, dims, tmp_path):
    train_loader, val_loader, _ = build_loaders(dataset, data_cfg)
    cfg = Config(
        data=data_cfg,
        model=model_cfg,
        train=TrainConfig(epochs=2, device="cpu", checkpoint_dir=str(tmp_path), log_every=1),
    )
    history = fit_actor_critic(cfg, train_loader, val_loader, dims, torch.device("cpu"))
    assert set(history) == {"actor_loss", "critic_loss", "train_jaccard", "rl_loss",
                            "sl_loss", "val_loss", "val_jaccard", "val_accuracy"}
    assert all(len(v) == 2 for v in history.values())
    assert all(math.isfinite(x) for x in history["actor_loss"])
