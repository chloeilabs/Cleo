from __future__ import annotations

import os
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from .identity import model_identity_metadata


def atomic_torch_save(payload: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(Path(path), map_location=map_location, weights_only=False)


def stamp_checkpoint_identity(path: str | Path) -> bool:
    """Atomically add canonical identity metadata to an existing checkpoint.

    Returns ``True`` when the file changed and ``False`` when it already carried
    the current identity schema.
    """

    checkpoint = load_checkpoint(path, map_location="cpu")
    identity = model_identity_metadata()
    if checkpoint.get("identity") == identity and int(checkpoint.get("format_version", 1)) >= 2:
        return False
    checkpoint["format_version"] = max(int(checkpoint.get("format_version", 1)), 2)
    checkpoint["identity"] = identity
    atomic_torch_save(checkpoint, path)
    return True


def capture_rng_state(data_generator: torch.Generator | None = None) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if data_generator is not None:
        state["data_generator"] = data_generator.get_state()
    try:
        if torch.backends.mps.is_available():
            state["torch_mps"] = torch.mps.get_rng_state()
    except RuntimeError:
        pass
    return state


def restore_rng_state(state: dict[str, Any], data_generator: torch.Generator | None = None) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if data_generator is not None and "data_generator" in state:
        data_generator.set_state(state["data_generator"])
    if "torch_mps" in state:
        try:
            if torch.backends.mps.is_available():
                torch.mps.set_rng_state(state["torch_mps"])
        except RuntimeError:
            pass
