"""
STFT codec for converting between waveform tensors and real/imaginary STFT feature maps.

by Wang Guozheng
2026-06-29
"""
from __future__ import annotations

import torch
from torch import nn

class STFTCodec(nn.Module):
    """Convert between waveform tensors and real/imaginary STFT feature maps."""

    def __init__(self, n_fft: int, hop_length: int, win_length: int, channels: int, length: int) -> None:
        super().__init__()
        self.n_fft = n_fft                  # 傅里叶变换点数
        self.hop_length = hop_length        # 帧移
        self.win_length = win_length        # 窗长
        self.channels = channels            # 信号通道数
        self.length = length                # 一维信号长度
        self.n_freqs = n_fft // 2 + 1
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)

    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        """
        波形 -> STFT特征图
        将形状为 [batch, channels, time] 的波形，转换为供 2D 卷积网络处理的特征图
        """
        if audio.ndim != 3:
            raise ValueError(f"Expected [batch, channels, time], got {tuple(audio.shape)}")
        batch, channels, length = audio.shape
        if channels != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {channels}")

        flat = audio.reshape(batch * channels, length)
        spec = torch.stft(
            flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
            center=True,
        )
        spec = spec.reshape(batch, channels, self.n_freqs, spec.shape[-1])
        return torch.cat([spec.real, spec.imag], dim=1)

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """
        STFT特征图 -> 波形
        作为 encode 的逆操作。VAE 生成了二维特征图后，用这个方法还原回一维雷达/音频信号。
        """
        if features.ndim != 4:
            raise ValueError(f"Expected [batch, 2 * channels, freq, frames], got {tuple(features.shape)}")
        batch, feature_channels, freq, frames = features.shape
        if feature_channels != self.channels * 2:
            raise ValueError(f"Expected {self.channels * 2} STFT channels, got {feature_channels}")
        if freq != self.n_freqs:
            raise ValueError(f"Expected {self.n_freqs} frequency bins, got {freq}")

        real, imag = features[:, : self.channels], features[:, self.channels :]
        spec = torch.complex(real, imag).reshape(batch * self.channels, freq, frames)
        audio = torch.istft(
            spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            length=self.length,
        )
        return audio.reshape(batch, self.channels, self.length)

