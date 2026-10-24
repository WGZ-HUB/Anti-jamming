"""Inference checkpoint loader for the STFT-VAE."""

from pathlib import Path

import torch

from .model import STFTVAE


def load_checkpoint(path: str | Path, map_location="cpu"):
    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=map_location)

    model = STFTVAE.from_config(checkpoint["model_config"])
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
    if state_dict is None:
        raise KeyError("Checkpoint does not contain STFT-VAE model weights.")
    cleaned = {}
    for key, value in state_dict.items():
        key = key.removeprefix("model.").removeprefix("autoencoder.")
        cleaned[key] = value
    model.load_state_dict(cleaned, strict=True)
    return model, checkpoint
