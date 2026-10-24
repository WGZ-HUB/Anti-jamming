"""Condition-guided network used by the released inference pipeline."""

import torch
import torch.nn as nn

from .udit import UDiT
from .vit import ViT


def create_mask(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    _, sequence_length, _ = x.shape
    lengths = lengths.to(x.device)
    return torch.arange(sequence_length, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)


class TSE(nn.Module):
    """Predict diffusion velocity from noisy, jammed, and conditional latents."""

    def __init__(self, udit_model=None, vit_model=None):
        super().__init__()
        udit_model = dict(udit_model or {})
        vit_model = dict(vit_model or {})
        out_channels = udit_model.get("out_chans")
        context_dim = udit_model.get("context_dim")
        self.model = UDiT(**udit_model)
        self.spk_proj = nn.Linear(out_channels, context_dim)
        self.spk_model = ViT(**vit_model)

    def forward(self, x, timesteps, mixture, reference, x_len=None, ref_len=None):
        x_mask = create_mask(x, x_len)
        ref_mask = create_mask(reference, ref_len)

        condition = self.spk_proj(reference)
        condition = self.spk_model(x=condition, x_mask=ref_mask)

        hidden = torch.cat([x, mixture], dim=-1).transpose(1, 2)
        hidden = self.model(
            x=hidden,
            timesteps=timesteps,
            context=condition,
            x_mask=x_mask,
            context_mask=ref_mask,
            cls_token=None,
        )
        return hidden.transpose(1, 2), x_mask
