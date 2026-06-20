import math

import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal embedding for a scalar time t in [0, 1]."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim <= 0 or dim % 2 != 0:
            raise ValueError("embedding dim must be a positive even integer")
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        device = t.device
        # Frequencies cover slow and fast time variation. This is the same idea
        # as positional encodings, but the input coordinate is continuous time.
        # 低周波から高周波まで用意し、ゆっくりした時刻変化と速い時刻変化の両方を表す。
        # 位置埋め込みと同じ発想だが、入力座標は連続時刻tである。
        freqs = torch.exp(
            torch.linspace(
                math.log(1.0),
                math.log(1000.0),
                half,
                device=device,
                dtype=t.dtype,
            )
        )
        angles = t * freqs[None, :]
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)


class MLPVelocity(nn.Module):
    """Small velocity field v_theta(t, x) for 2D toy experiments."""

    def __init__(
        self,
        data_dim: int = 2,
        time_dim: int = 32,
        hidden_dim: int = 128,
        depth: int = 3,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        if data_dim <= 0 or hidden_dim <= 0:
            raise ValueError("data_dim and hidden_dim must be positive")
        if depth < 1:
            raise ValueError("depth must be at least 1")
        activation_factories: dict[str, type[nn.Module]] = {
            "silu": nn.SiLU,
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
        }
        if activation not in activation_factories:
            choices = ", ".join(activation_factories)
            raise ValueError(f"unknown activation {activation!r}; choose from: {choices}")
        self.time_embedding = SinusoidalTimeEmbedding(time_dim)
        layers: list[nn.Module] = []
        in_dim = data_dim + time_dim
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(activation_factories[activation]())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, data_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t[:, None]
        emb = self.time_embedding(t)
        # The velocity field is a function of both location and time:
        # v_theta(t, x). Concatenation is the simplest conditioning mechanism.
        # 速度場は場所xと時刻tの両方に依存する関数 v_theta(t, x)。
        # concatは最も単純な時刻条件付けの方法である。
        return self.net(torch.cat([x, emb], dim=1))


class RectifiedFlowMLP(MLPVelocity):
    """MLP velocity model for Rectified Flow toy experiments.

    This is architecturally the same as MLPVelocity. The separate name is useful
    in the material because Rectified Flow changes the training/evaluation
    protocol more than the small toy model architecture.

    日本語:
        2D toy用のRectified Flow速度場モデル。構造はMLPVelocityと同じだが、
        教材上は「Rectified Flow用モデル」として名前を分けて読む。
    """
