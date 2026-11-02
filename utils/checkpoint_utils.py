"""
Checkpoint and artifact persistence utilities for deep learning models.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional
import torch


def save_checkpoint(
    state: Dict[str, Any],
    filepath: Path,
    is_best: bool = True
) -> None:
    """
    Saves PyTorch model checkpoint.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, filepath)


def load_checkpoint(
    filepath: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device = torch.device("cpu")
) -> Dict[str, Any]:
    """
    Loads PyTorch checkpoint and restores model (and optimizer) parameters.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {filepath}")

    checkpoint = torch.load(filepath, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


def save_json_history(history: Dict[str, Any], filepath: Path) -> None:
    """
    Saves training history metrics to JSON.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def load_json_history(filepath: Path) -> Dict[str, Any]:
    """
    Loads training history metrics from JSON.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        return {}
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
