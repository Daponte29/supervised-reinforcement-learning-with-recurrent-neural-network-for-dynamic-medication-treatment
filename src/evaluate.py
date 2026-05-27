from __future__ import annotations

import argparse

from src.config import Config
from src.data import build_dataset_from_dir, build_loaders
from src.train import evaluate
from src.utils import get_device, load_model


def main(cfg: Config, checkpoint: str) -> dict[str, float]:
    device = get_device(cfg.train.device)
    dataset = build_dataset_from_dir(cfg.data)
    _, _, test_loader = build_loaders(dataset, cfg.data)

    model = load_model(checkpoint, device)
    test_loss, test_jaccard, test_accuracy, _, _ = evaluate(model, test_loader, device)

    results = {"loss": test_loss, "jaccard": test_jaccard, "accuracy": test_accuracy}
    print(f"Test set — loss={test_loss:.4f}  jaccard={test_jaccard:.4f}  accuracy={test_accuracy:.4f}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment_01.yaml")
    parser.add_argument("--checkpoint", default="output_models/best_actor.pth")
    args = parser.parse_args()
    main(Config.from_yaml(args.config), args.checkpoint)
