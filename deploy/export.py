"""Export a trained checkpoint to ONNX or TorchScript."""
from __future__ import annotations

import argparse
import torch
from pathlib import Path

from src.config import Config
from src.model import build_model
from src.utils import load_checkpoint, get_device


def export_onnx(model: torch.nn.Module, dummy_input: torch.Tensor, out_path: str) -> None:
    torch.onnx.export(
        model,
        dummy_input,
        out_path,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )
    print(f"ONNX model saved to {out_path}")


def export_torchscript(model: torch.nn.Module, dummy_input: torch.Tensor, out_path: str) -> None:
    traced = torch.jit.trace(model, dummy_input)
    traced.save(out_path)
    print(f"TorchScript model saved to {out_path}")


def main(cfg: Config, checkpoint: str, format: str, out_dir: str) -> None:
    device = get_device("cpu")
    model = build_model(cfg.model).to(device)
    ckpt = load_checkpoint(checkpoint, device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    # TODO: set dummy input shape to match your model's expected input
    dummy = torch.zeros(1, 3, 224, 224)

    if format == "onnx":
        export_onnx(model, dummy, f"{out_dir}/model.onnx")
    elif format == "torchscript":
        export_torchscript(model, dummy, f"{out_dir}/model.pt")
    else:
        raise ValueError(f"Unknown format: {format}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment_01.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.ckpt")
    parser.add_argument("--format", choices=["onnx", "torchscript"], default="onnx")
    parser.add_argument("--out-dir", default="deploy/artifacts")
    args = parser.parse_args()
    main(Config.from_yaml(args.config), args.checkpoint, args.format, args.out_dir)
