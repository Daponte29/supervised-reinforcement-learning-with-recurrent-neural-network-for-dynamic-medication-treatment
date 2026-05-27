from src.config import Config, DataConfig


def test_data_config_defaults():
    cfg = DataConfig()
    assert cfg.processed_dir == "data"
    assert cfg.time_stamp == 8


def test_config_from_yaml(tmp_path):
    yaml_text = """
experiment_name: unit_test
data:
  time_stamp: 4
  batch_size: 16
model:
  hidden2_units: 64
train:
  epochs: 3
  device: cpu
"""
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml_text)

    cfg = Config.from_yaml(path)
    assert cfg.experiment_name == "unit_test"
    assert cfg.data.time_stamp == 4
    assert cfg.data.batch_size == 16
    assert cfg.model.hidden2_units == 64
    assert cfg.train.epochs == 3
    assert cfg.train.device == "cpu"


def test_repo_configs_load():
    for name in ("configs/base.yaml", "configs/experiment_01.yaml"):
        cfg = Config.from_yaml(name)
        assert cfg.data.time_stamp > 0
        assert cfg.train.epochs > 0
