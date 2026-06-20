"""2次元 Flow Matching 実験で使う source/target 分布の sampling utilities。

source は標準正規分布、target は円周上に並ぶ eight-Gaussians mixture とし、
多峰性を保った輸送を小さなモデルでも観察できるようにする。
"""

import math

import torch


def eight_gaussian_centers(
    radius: float = 2.0,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Mode centers for the eight-Gaussians toy target distribution."""

    ids = torch.arange(8, device=device)
    angles = ids.float() * (2.0 * math.pi / 8.0)
    return torch.stack([radius * torch.cos(angles), radius * torch.sin(angles)], dim=1)


def sample_standard_normal(batch: int, dim: int = 2, device: str | torch.device = "cpu") -> torch.Tensor:
    """Source distribution p_0 for the toy Flow Matching experiments."""

    return torch.randn(batch, dim, device=device)


def sample_eight_gaussians(
    batch: int,
    radius: float = 2.0,
    std: float = 0.08,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Eight Gaussian modes on a circle, useful as a tiny target distribution."""

    # First choose which mode each sample belongs to, then add small local noise.
    # This makes mode coverage easy to evaluate later.
    # まず各サンプルが属するmodeを選び、その周囲に小さなノイズを足す。
    # あとでmode coverageを評価しやすいtoy分布になる。
    ids = torch.randint(0, 8, (batch,), device=device)
    centers = eight_gaussian_centers(radius=radius, device=device)
    return centers[ids] + std * torch.randn(batch, 2, device=device)
