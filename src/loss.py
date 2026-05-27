from __future__ import annotations

import torch.nn as nn

# Models apply a Sigmoid in their final layer, so the supervised criterion is
# plain BCELoss (not BCEWithLogitsLoss).
_REGISTRY = {
    "bce": nn.BCELoss,
    "bce_logits": nn.BCEWithLogitsLoss,
    "mse": nn.MSELoss,
    "l1": nn.L1Loss,
    "cross_entropy": nn.CrossEntropyLoss,
}


def build_criterion(name: str = "bce") -> nn.Module:
    """Return a loss module by name."""
    if name not in _REGISTRY:
        raise ValueError(f"Unknown loss '{name}'. Available: {list(_REGISTRY)}")
    return _REGISTRY[name]()
