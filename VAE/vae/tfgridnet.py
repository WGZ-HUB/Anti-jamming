"""
TF-GridNet block adapted from the original Stable Audio VAE code.

by Wang Guozheng
2026-06-30

[B, C, F, T]
→ permute
[B, C, T, F]
→ GridNetV2Block
    → pad 时间和频率
    → intra BiLSTM: 沿频率轴建模
    → inter BiLSTM: 沿时间轴建模
    → crop 回原始 T/F
    → Q/K/V 多头注意力：沿时间轴做 self-attention
    → 残差输出
[B, C, T, F]
→ permute back
[B, C, F, T]
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter


class GridNetV2Block(nn.Module):
    """TF-GridNet block.

    Input and output shape: [batch, channels, time, frequency].
    The block alternates frequency-axis and time-axis BiLSTMs, then applies
    multi-head self-attention along the time axis.
    输入特征 → 频率轴 BiLSTM → 时间轴 BiLSTM → 时间轴多头自注意力 → 残差输出
    """

    def __getitem__(self, key: str):
        return getattr(self, key)       # 让模块支持类似字典的访问方式(访问对象的属性)

    def __init__(
        self,
        emb_dim: int,                       # 输入通道数
        emb_ks: int,                        # 输入卷积核大小
        emb_hs: int,                        # 输入卷积步长
        n_freqs: int,                       # 频率轴长度
        hidden_channels: int,               # 隐藏通道数
        n_head: int = 4,                    # 多头注意力头数
        approx_qk_dim: int = 512,           # 近似查询和键的维度
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        in_channels = emb_dim * emb_ks

        self.intra_norm = nn.LayerNorm(emb_dim, eps=eps)
        self.intra_rnn = nn.LSTM(                                                   # 单层双向 LSTM
            in_channels, hidden_channels, 1, batch_first=True, bidirectional=True
        )
        if emb_ks == emb_hs:
            self.intra_linear = nn.Linear(hidden_channels * 2, in_channels)
        else:
            self.intra_linear = nn.ConvTranspose1d(
                hidden_channels * 2, emb_dim, emb_ks, stride=emb_hs
            )

        self.inter_norm = nn.LayerNorm(emb_dim, eps=eps)
        self.inter_rnn = nn.LSTM(
            in_channels, hidden_channels, 1, batch_first=True, bidirectional=True
        )
        if emb_ks == emb_hs:
            self.inter_linear = nn.Linear(hidden_channels * 2, in_channels)
        else:
            self.inter_linear = nn.ConvTranspose1d(
                hidden_channels * 2, emb_dim, emb_ks, stride=emb_hs
            )

        qk_channels_per_head = math.ceil(approx_qk_dim / n_freqs)
        if emb_dim % n_head != 0:
            raise ValueError(f"emb_dim={emb_dim} must be divisible by n_head={n_head}")

        self.add_module("attn_conv_Q", nn.Conv2d(emb_dim, n_head * qk_channels_per_head, 1))
        self.add_module(
            "attn_norm_Q",
            AllHeadPReLULayerNormalization4DCF((n_head, qk_channels_per_head, n_freqs), eps=eps),
        )

        self.add_module("attn_conv_K", nn.Conv2d(emb_dim, n_head * qk_channels_per_head, 1))
        self.add_module(
            "attn_norm_K",
            AllHeadPReLULayerNormalization4DCF((n_head, qk_channels_per_head, n_freqs), eps=eps),
        )

        self.add_module("attn_conv_V", nn.Conv2d(emb_dim, emb_dim, 1))
        self.add_module(
            "attn_norm_V",
            AllHeadPReLULayerNormalization4DCF((n_head, emb_dim // n_head, n_freqs), eps=eps),
        )

        self.add_module(
            "attn_concat_proj",
            nn.Sequential(
                nn.Conv2d(emb_dim, emb_dim, 1),
                nn.PReLU(),
                LayerNormalization4DCF((emb_dim, n_freqs), eps=eps),
            ),
        )

        self.emb_dim = emb_dim
        self.emb_ks = emb_ks
        self.emb_hs = emb_hs
        self.n_head = n_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, old_time, old_freq = x.shape                                      # 【B, C, T, F】

        overlap = self.emb_ks - self.emb_hs                                                     # 窗口重叠量
        padded_time = (                                                                         # 填充后的时间轴长度
            math.ceil((old_time + 2 * overlap - self.emb_ks) / self.emb_hs) * self.emb_hs
            + self.emb_ks
        )
        padded_freq = (                                                                         # 填充后的频率轴长度
            math.ceil((old_freq + 2 * overlap - self.emb_ks) / self.emb_hs) * self.emb_hs
            + self.emb_ks
        )

        x = x.permute(0, 2, 3, 1)                                                                # 【B, T, F, C】
        x = F.pad(                                                                               # 【B, padded_time, padded_freq, C】
            x,
            (0, 0, overlap, padded_freq - old_freq - overlap, overlap, padded_time - old_time - overlap),
        )

        # 沿频率轴建模
        intra_input = x
        intra = self.intra_norm(intra_input)                                                    # 保存残差输入，并对输入做 LayerNorm
        if self.emb_ks == self.emb_hs:
            intra = intra.view(batch_size * padded_time, -1, self.emb_ks * channels)            # [B, padded_time, padded_freq, C] → [B * padded_time, padded_freq / emb_ks, emb_ks * C] 
            intra, _ = self.intra_rnn(intra)
            intra = self.intra_linear(intra)
            intra = intra.view(batch_size, padded_time, padded_freq, channels)                  # [B, padded_time, padded_freq, C]
        else:
            intra = intra.view(batch_size * padded_time, padded_freq, channels)                 # [B * padded_time, padded_freq, C]
            intra = intra.transpose(1, 2)                                                       # [B * padded_time, C, padded_freq]
            intra = F.unfold(intra[..., None], (self.emb_ks, 1), stride=(self.emb_hs, 1))       # unfold 成多个频率窗口
            intra = intra.transpose(1, 2)
            intra, _ = self.intra_rnn(intra)
            intra = intra.transpose(1, 2)
            intra = self.intra_linear(intra)
            intra = intra.view(batch_size, padded_time, channels, padded_freq)
            intra = intra.transpose(-2, -1)
        intra = intra + intra_input

        # 沿时间轴建模
        intra = intra.transpose(1, 2)                                                           # [B, padded_freq, padded_time, C]
        inter_input = intra
        inter = self.inter_norm(inter_input)
        if self.emb_ks == self.emb_hs:
            inter = inter.view(batch_size * padded_freq, -1, self.emb_ks * channels)
            inter, _ = self.inter_rnn(inter)
            inter = self.inter_linear(inter)
            inter = inter.view(batch_size, padded_freq, padded_time, channels)
        else:
            inter = inter.view(batch_size * padded_freq, padded_time, channels)
            inter = inter.transpose(1, 2)
            inter = F.unfold(inter[..., None], (self.emb_ks, 1), stride=(self.emb_hs, 1))
            inter = inter.transpose(1, 2)
            inter, _ = self.inter_rnn(inter)
            inter = inter.transpose(1, 2)
            inter = self.inter_linear(inter)
            inter = inter.view(batch_size, padded_freq, channels, padded_time)
            inter = inter.transpose(-2, -1)
        inter = inter + inter_input

        inter = inter.permute(0, 3, 2, 1)                                                       # -> [B, C, padded_time, padded_freq]
        inter = inter[..., overlap : overlap + old_time, overlap : overlap + old_freq]          # 裁剪掉前面加的 padding

        query = self["attn_norm_Q"](self["attn_conv_Q"](inter))                                 # [B, n_head, qk_channels_per_head, T, F]
        key = self["attn_norm_K"](self["attn_conv_K"](inter))                                   # [B, n_head, qk_channels_per_head, T, F]
        value = self["attn_norm_V"](self["attn_conv_V"](inter))                                 # [B, n_head, C / n_head, T, F]
        query = query.view(-1, *query.shape[2:])                                                # [B * n_head, qk_channels_per_head, T, F]
        key = key.view(-1, *key.shape[2:])                                                      # [B * n_head, qk_channels_per_head, T, F]
        value = value.view(-1, *value.shape[2:])                                                # [B * n_head, C / n_head, T, F]    

        query = query.transpose(1, 2).flatten(start_dim=2)                                      # [B * n_head, T, qk_channels_per_head * F]
        key = key.transpose(2, 3).contiguous().view(batch_size * self.n_head, -1, old_time)     # [B * n_head, qk_channels_per_head * F, T]
        value = value.transpose(1, 2)                                                           # [B * n_head, T, C / n_head, F]
        old_value_shape = value.shape
        value = value.flatten(start_dim=2)                                                      # [B * n_head, T, C / n_head * F]

        attn = torch.matmul(query, key) / (query.shape[-1] ** 0.5)
        attn = F.softmax(attn, dim=2)
        value = torch.matmul(attn, value)

        value = value.reshape(old_value_shape).transpose(1, 2)
        head_channels = value.shape[1]
        value = value.contiguous().view(batch_size, self.n_head * head_channels, old_time, old_freq)
        value = self["attn_concat_proj"](value)

        return value + inter


class FrequencyTimeGridNetBlock(nn.Module):
    """负责在进入和离开 GridNetV2Block 时交换 F 和 T 维度"""

    def __init__(
        self,
        channels: int,
        n_freqs: int,
        rnn_hidden_channels: int,
        emb_ks: int = 1,
        emb_hs: int = 1,
        n_head: int = 4,
        approx_qk_dim: int = 512,
    ) -> None:
        super().__init__()
        self.block = GridNetV2Block(
            emb_dim=channels,
            emb_ks=emb_ks,
            emb_hs=emb_hs,
            n_freqs=n_freqs,
            hidden_channels=rnn_hidden_channels,
            n_head=n_head,
            approx_qk_dim=approx_qk_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 1, 3, 2).contiguous()
        x = self.block(x)
        return x.permute(0, 1, 3, 2).contiguous()


class LayerNormalization4DCF(nn.Module):
    """针对 4D 卷积特征的 LayerNorm"""
    def __init__(self, input_dimension: tuple[int, int], eps: float = 1e-5) -> None:
        super().__init__()
        if len(input_dimension) != 2:
            raise ValueError("input_dimension must be (channels, frequency)")
        param_size = [1, input_dimension[0], 1, input_dimension[1]]
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected 4 dimensions, got {x.ndim}")
        stat_dim = (1, 3)
        mean = x.mean(dim=stat_dim, keepdim=True)
        std = torch.sqrt(x.var(dim=stat_dim, unbiased=False, keepdim=True) + self.eps)
        return ((x - mean) / std) * self.gamma + self.beta


class AllHeadPReLULayerNormalization4DCF(nn.Module):
    """专门给多头注意力的 Q/K/V 用的归一化层"""
    def __init__(self, input_dimension: tuple[int, int, int], eps: float = 1e-5) -> None:
        super().__init__()
        if len(input_dimension) != 3:
            raise ValueError("input_dimension must be (heads, channels_per_head, frequency)")
        heads, channels_per_head, n_freqs = input_dimension
        param_size = [1, heads, channels_per_head, 1, n_freqs]
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        self.act = nn.PReLU(num_parameters=heads, init=0.25)
        self.eps = eps
        self.heads = heads
        self.channels_per_head = channels_per_head
        self.n_freqs = n_freqs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected 4 dimensions, got {x.ndim}")
        batch_size, _, time_steps, _ = x.shape                                                      # [B, H * C_head, T, F]
        x = x.view(batch_size, self.heads, self.channels_per_head, time_steps, self.n_freqs)        # [B, H, C_head, T, F]
        x = self.act(x)
        stat_dim = (2, 4)                                                                           # 对每个 batch、每个 head、每个 time frame，单独在 C_head 和 F 上做归一化
        mean = x.mean(dim=stat_dim, keepdim=True)
        std = torch.sqrt(x.var(dim=stat_dim, unbiased=False, keepdim=True) + self.eps)
        return ((x - mean) / std) * self.gamma + self.beta                                          # [B, H, C_head, T, F]

