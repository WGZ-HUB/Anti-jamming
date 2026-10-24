"""
STFT VAE

by Wang Guozheng
2026-06-29
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .stft import STFTCodec
from .tfgridnet import FrequencyTimeGridNetBlock


@dataclass
class STFTVAEOutput:
    """使用数据类封装 VAE 前向传播的输出"""
    reconstruction: torch.Tensor
    latents: torch.Tensor
    mu: torch.Tensor
    logvar: torch.Tensor
    kl_loss: torch.Tensor
    signal_std: torch.Tensor


class ResidualConvBlock(nn.Module):
    """"二维残差块 ResBlock"""
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(1, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


def create_feature_blocks(
    block_type: str,
    num_blocks: int,
    channels: int,
    n_freqs: int,
    tfgridnet_hidden_channels: int,
    tfgridnet_emb_ks: int,
    tfgridnet_emb_hs: int,
    tfgridnet_heads: int,
    tfgridnet_approx_qk_dim: int,
) -> list[nn.Module]:
    if block_type == "residual":
        return [ResidualConvBlock(channels) for _ in range(num_blocks)]
    if block_type == "tfgridnet":
        return [
            FrequencyTimeGridNetBlock(
                channels=channels,
                n_freqs=n_freqs,
                rnn_hidden_channels=tfgridnet_hidden_channels,
                emb_ks=tfgridnet_emb_ks,
                emb_hs=tfgridnet_emb_hs,
                n_head=tfgridnet_heads,
                approx_qk_dim=tfgridnet_approx_qk_dim,
            )
            for _ in range(num_blocks)
        ]
    raise ValueError(f"Unknown block_type: {block_type}")


class STFTVAE(nn.Module):
    """A compact STFT VAE for two-channel real/imaginary signal reconstruction."""

    def __init__(
        self,
        sample_size: int = 6400,
        audio_channels: int = 2,
        latent_dim: int = 64,
        hidden_channels: int = 32,
        n_fft: int = 256,
        hop_length: int = 64,
        win_length: int = 128,
        num_blocks: int = 2,
        block_type: str = "residual",
        tfgridnet_hidden_channels: int | None = None,
        tfgridnet_emb_ks: int = 1,
        tfgridnet_emb_hs: int = 1,
        tfgridnet_heads: int = 4,
        tfgridnet_approx_qk_dim: int = 512,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.sample_size = sample_size
        self.audio_channels = audio_channels
        self.latent_dim = latent_dim
        self.hidden_channels = hidden_channels
        self.block_type = block_type
        self.eps = eps

        self.stft = STFTCodec(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            channels=audio_channels,
            length=sample_size,
        )
        in_channels = audio_channels * 2
        flat_channels = hidden_channels * self.stft.n_freqs
        if tfgridnet_hidden_channels is None:
            tfgridnet_hidden_channels = max(128, hidden_channels * 2)

        encoder_blocks = create_feature_blocks(
            block_type=block_type,
            num_blocks=num_blocks,
            channels=hidden_channels,
            n_freqs=self.stft.n_freqs,
            tfgridnet_hidden_channels=tfgridnet_hidden_channels,
            tfgridnet_emb_ks=tfgridnet_emb_ks,
            tfgridnet_emb_hs=tfgridnet_emb_hs,
            tfgridnet_heads=tfgridnet_heads,
            tfgridnet_approx_qk_dim=tfgridnet_approx_qk_dim,
        )
        decoder_blocks = create_feature_blocks(
            block_type=block_type,
            num_blocks=num_blocks,
            channels=hidden_channels,
            n_freqs=self.stft.n_freqs,
            tfgridnet_hidden_channels=tfgridnet_hidden_channels,
            tfgridnet_emb_ks=tfgridnet_emb_ks,
            tfgridnet_emb_hs=tfgridnet_emb_hs,
            tfgridnet_heads=tfgridnet_heads,
            tfgridnet_approx_qk_dim=tfgridnet_approx_qk_dim,
        )

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            *encoder_blocks,
        )
        self.to_stats = nn.Sequential(
            nn.Conv1d(flat_channels, hidden_channels * 4, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(hidden_channels * 4, latent_dim * 2, kernel_size=1),
        )

        self.from_latent = nn.Sequential(
            nn.Conv1d(latent_dim, hidden_channels * 4, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(hidden_channels * 4, flat_channels, kernel_size=1),
        )
        self.decoder = nn.Sequential(
            *decoder_blocks,
            nn.GroupNorm(1, hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=3, padding=1),
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "STFTVAE":
        keys = {
            "sample_size",
            "audio_channels",
            "latent_dim",
            "hidden_channels",
            "n_fft",
            "hop_length",
            "win_length",
            "num_blocks",
            "block_type",
            "tfgridnet_hidden_channels",
            "tfgridnet_emb_ks",
            "tfgridnet_emb_hs",
            "tfgridnet_heads",
            "tfgridnet_approx_qk_dim",
        }
        return cls(**{key: config[key] for key in keys if key in config})

    def encode(
        self,
        audio: torch.Tensor,
        deterministic: bool = False,        # 是否使用确定性编码
        return_info: bool = False,          # 是否返回额外信息
    ):
        signal_std = audio.std(dim=(1, 2), keepdim=True).clamp_min(self.eps)        # 计算每个样本的标准差 【B,C,T】->【B,1,1】
        normalized = audio / signal_std                                             # 归一化
        stft_features = self.stft.encode(normalized)                                # 将归一化后的信号转换为 STFT 特征图 【B, 2*C, F, T】

        hidden = self.encoder(stft_features)                                        # 2D卷积编码 【B, hidden_channels, F, T】
        batch, channels, freqs, frames = hidden.shape
        hidden = hidden.reshape(batch, channels * freqs, frames)                    # 将二维特征图展平为一维特征图 【B, hidden_channels*F, T】
        stats = self.to_stats(hidden)                                               # 1D卷积生成均值和对数方差 【B, latent_dim*2, T】
        mu, logvar = stats.chunk(2, dim=1)                                          # 将均值和对数方差分开 【B, latent_dim, T】
        logvar = logvar.clamp(min=-20.0, max=10.0)                                  # 对数方差裁剪，防止数值不稳定

        if deterministic:                                   # 是否使用确定性编码
            latents = mu                                    # 使用均值作为潜在变量
        else:
            latents = self.reparameterize(mu, logvar)       # 使用重参数化技巧采样潜在变量

        kl_loss = self.kl_divergence(mu, logvar)            # 计算 KL 散度损失
        if return_info:
            return latents, signal_std, {"mu": mu, "logvar": logvar, "kl_loss": kl_loss}
        return latents, signal_std

    def decode(self, latents: torch.Tensor, signal_std: torch.Tensor | None = None) -> torch.Tensor:
        hidden = self.from_latent(latents)                                                      # 【B, latent_dim, T】->【B, hidden_channels*F, T】
        batch, _, frames = hidden.shape
        hidden = hidden.reshape(batch, self.hidden_channels, self.stft.n_freqs, frames)         # -> 【B, hidden_channels, F, T】
        stft_features = self.decoder(hidden)                                                    # -> 【B, 2*C, F, T】
        audio = self.stft.decode(stft_features)                                                 # -> 【B, C, T】     
        if signal_std is not None:
            audio = audio * signal_std
        return audio

    def forward(self, audio: torch.Tensor, deterministic: bool = False) -> STFTVAEOutput:
        latents, signal_std, info = self.encode(audio, deterministic=deterministic, return_info=True)
        reconstruction = self.decode(latents, signal_std)
        return STFTVAEOutput(
            reconstruction=reconstruction,
            latents=latents,
            mu=info["mu"],
            logvar=info["logvar"],
            kl_loss=info["kl_loss"],
            signal_std=signal_std,
        )

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        noise = torch.randn_like(std)
        return mu + noise * std

    @staticmethod
    def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        kl_per_latent = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
        return kl_per_latent.sum(dim=1).mean()
