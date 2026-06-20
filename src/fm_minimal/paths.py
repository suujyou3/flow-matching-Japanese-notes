from dataclasses import dataclass
from typing import Protocol

import torch


class ConditionalPath(Protocol):
    """Interface required by the conditional Flow Matching loss."""

    def sample(self, t: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor: ...

    def velocity(self, t: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor: ...


def _time_like(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Append singleton axes so one time value broadcasts over each sample."""

    if t.ndim == 0:
        return t
    if t.shape[0] != x.shape[0]:
        raise ValueError(f"time batch {t.shape[0]} does not match data batch {x.shape[0]}")
    while t.ndim < x.ndim:
        t = t.unsqueeze(-1)
    return t


@dataclass(frozen=True)
class LinearPath:
    """Straight conditional path from x0 to x1.

    Mathematical convention:
        x_t = (1 - t) x_0 + t x_1
        u_t = d x_t / dt = x_1 - x_0

    Shapes:
        x0: [batch, dim]
        x1: [batch, dim]
        t:  [batch, 1] or scalar tensor broadcastable to x0
    """

    def sample(self, t: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        # This is the conditional point x_t used in the CFM loss.
        # Broadcasting requires t to be [batch, 1] when x0/x1 are [batch, dim].
        # CFM lossで使う条件付き途中点x_tを作る。
        # x0/x1が[batch, dim]なら、broadcastのためtは[batch, 1]にする。
        t = _time_like(t, x0)
        return (1.0 - t) * x0 + t * x1

    def velocity(self, t: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        # Linear interpolation has constant velocity, so t is unused.
        # 直線補間の速度は一定なので、tは使わない。
        del t
        return x1 - x0


@dataclass(frozen=True)
class TrigGaussianPath:
    """Variance-preserving Gaussian-style path.

    Mathematical convention:
        x_t = alpha(t) x_1 + sigma(t) x_0
        alpha(t) = sin(pi t / 2)
        sigma(t) = cos(pi t / 2)

    Here x0 is noise and x1 is data. This convention keeps t=0 as source
    and t=1 as target, matching LinearPath.
    """

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sin(0.5 * torch.pi * t)

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        return torch.cos(0.5 * torch.pi * t)

    def alpha_dot(self, t: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.pi * torch.cos(0.5 * torch.pi * t)

    def sigma_dot(self, t: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.pi * torch.sin(0.5 * torch.pi * t)

    def sample(self, t: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        # Gaussian-style interpolation: data coefficient grows, noise coefficient shrinks.
        # Gaussian風の補間。data成分の係数が増え、noise成分の係数が減る。
        t = _time_like(t, x0)
        return self.alpha(t) * x1 + self.sigma(t) * x0

    def velocity(self, t: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        # Velocity is the time derivative of sample(t, x0, x1).
        # 速度は sample(t, x0, x1) を時刻tで微分したもの。
        t = _time_like(t, x0)
        return self.alpha_dot(t) * x1 + self.sigma_dot(t) * x0
