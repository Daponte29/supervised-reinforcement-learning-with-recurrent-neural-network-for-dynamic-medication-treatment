import torch

from src.data import my_collate_fn
from src.model import build_actor, build_basic_lstm, build_critic, soft_update
from tests.conftest import MED_SIZE, TIME_STAMP


def _batch(dataset, n=4):
    return my_collate_fn([dataset[i] for i in range(n)])


def test_actor_forward_shape_and_range(dataset, model_cfg, dims):
    actor = build_actor(model_cfg, dims, TIME_STAMP).eval()
    states, actions, _, _, _, diseases, demos = _batch(dataset)
    with torch.no_grad():
        out = actor(states.float(), diseases, demos)
    assert out.shape == (4, TIME_STAMP, MED_SIZE)
    assert out.min() >= 0.0 and out.max() <= 1.0  # sigmoid output


def test_critic_forward_shape(dataset, model_cfg, dims):
    critic = build_critic(model_cfg, dims, TIME_STAMP).eval()
    states, actions, _, _, _, diseases, demos = _batch(dataset)
    with torch.no_grad():
        q = critic(states.float(), actions.float(), diseases, demos)
    assert q.shape == (4, TIME_STAMP, 1)


def test_basic_lstm_forward_shape_and_range(dataset, model_cfg, dims):
    model = build_basic_lstm(model_cfg, dims, TIME_STAMP).eval()
    states, actions, _, _, _, diseases, demos = _batch(dataset)
    with torch.no_grad():
        out = model(states.float(), diseases, demos)
    assert out.shape == (4, TIME_STAMP, MED_SIZE)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_soft_update_moves_target_toward_source(model_cfg, dims):
    source = build_actor(model_cfg, dims, TIME_STAMP)
    target = build_actor(model_cfg, dims, TIME_STAMP)
    with torch.no_grad():
        for p in source.parameters():
            p.add_(1.0)
    before = next(target.parameters()).clone()
    soft_update(target, source, tau=0.5)
    after = next(target.parameters())
    assert not torch.equal(before, after)  # target shifted toward source
