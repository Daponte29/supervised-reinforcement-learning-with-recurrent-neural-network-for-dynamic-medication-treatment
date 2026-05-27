"""Lightweight inference wrapper around an ONNX or TorchScript model."""
from __future__ import annotations

import numpy as np
from pathlib import Path


class ONNXInferenceEngine:
    def __init__(self, model_path: str):
        import onnxruntime as ort
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, x: np.ndarray) -> np.ndarray:
        # x shape: (batch, ...) — TODO: adjust preprocessing as needed
        outputs = self.session.run(None, {self.input_name: x.astype(np.float32)})
        return outputs[0]


class TorchScriptInferenceEngine:
    def __init__(self, model_path: str):
        import torch
        self.model = torch.jit.load(model_path)
        self.model.eval()

    def predict(self, x: np.ndarray) -> np.ndarray:
        import torch
        with torch.no_grad():
            tensor = torch.from_numpy(x.astype(np.float32))
            return self.model(tensor).numpy()


def load_engine(model_path: str):
    p = Path(model_path)
    if p.suffix == ".onnx":
        return ONNXInferenceEngine(model_path)
    elif p.suffix == ".pt":
        return TorchScriptInferenceEngine(model_path)
    raise ValueError(f"Unsupported model format: {p.suffix}")
