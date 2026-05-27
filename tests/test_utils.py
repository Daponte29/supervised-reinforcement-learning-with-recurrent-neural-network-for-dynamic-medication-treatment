import torch

from src.data import my_collate_fn
from src.model import build_actor
from src.utils import get_device, load_model, save_model, set_seed
from tests.conftest import TIME_STAMP


def test_set_seed_is_reproducible():
    set_seed(123)
    a = torch.rand(5)
    set_seed(123)
    b = torch.rand(5)
    assert torch.equal(a, b)


def test_get_device_falls_back_to_cpu():
    assert get_device("cpu").type == "cpu"


def test_save_load_model_roundtrip(tmp_path, dataset, model_cfg, dims):
    actor = build_actor(model_cfg, dims, TIME_STAMP).eval()
    path = tmp_path / "actor.pth"
    save_model(actor, path)
    assert path.exists()

    loaded = load_model(path, torch.device("cpu")).eval()
    states, _, _, _, _, diseases, demos = my_collate_fn([dataset[i] for i in range(2)])
    with torch.no_grad():
        out_a = actor(states.float(), diseases, demos)
        out_b = loaded(states.float(), diseases, demos)
    assert torch.allclose(out_a, out_b)
