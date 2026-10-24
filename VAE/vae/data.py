"""Signal conversion helpers used during inference."""

import numpy as np
import torch
import torch.nn.functional as F


def complex_to_two_channels(signal: np.ndarray) -> torch.Tensor:
    signal = np.asarray(signal)
    return torch.from_numpy(np.stack([signal.real, signal.imag], axis=0)).float()


def pad_or_crop(signal: torch.Tensor, sample_size: int, random_crop: bool = False):
    del random_crop
    length = signal.shape[-1]
    if length < sample_size:
        return F.pad(signal, (0, sample_size - length))
    return signal[..., :sample_size]
