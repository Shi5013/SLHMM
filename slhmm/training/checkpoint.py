"""Versioned, resumable training checkpoints."""

from __future__ import annotations

from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from slhmm.configs.training import TrainingConfig
from slhmm.training.common import atomic_torch_save


CHECKPOINT_FORMAT = "slhmm.training.v1"


def _rng_state() -> dict[str, Any]:
    value = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        value["cuda"] = torch.cuda.get_rng_state_all()
    return value


def _restore_rng_state(value: dict[str, Any]) -> None:
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"].cpu())
    if torch.cuda.is_available() and "cuda" in value:
        torch.cuda.set_rng_state_all([state.cpu() for state in value["cuda"]])


def save_checkpoint(
    path: str | Path,
    *,
    stage: str,
    epoch: int,
    global_step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any,
    scaler: Any,
    config: TrainingConfig,
    data_hashes: dict[str, str],
    metrics: dict[str, float],
) -> None:
    atomic_torch_save(
        path,
        {
            "format": CHECKPOINT_FORMAT,
            "stage": stage,
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": (
                optimizer.state_dict() if optimizer is not None else None
            ),
            "scheduler_state_dict": (
                scheduler.state_dict() if scheduler is not None else None
            ),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "config": config.to_dict(),
            "data_hashes": data_hashes,
            "metrics": metrics,
            "rng_state": _rng_state(),
        },
    )


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    scaler: Any = None,
    map_location: str | torch.device = "cpu",
    restore_rng: bool = True,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location)
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"Unsupported checkpoint format in {path}")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    if restore_rng:
        _restore_rng_state(checkpoint["rng_state"])
    return checkpoint

