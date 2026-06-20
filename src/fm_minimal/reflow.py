"""Reflow pair の生成と、生成軌道の直線性を測る utilities。

既存 model の ODE flow map で x0 を輸送して新しい endpoint pair を作る。
straightness ratio は軌道長/端点距離で、1 に近いほど直線的である。
"""

from collections.abc import Callable

import torch
from torch import nn

from .solvers import euler_solve


@torch.no_grad()
def make_reflow_pairs(
    model: nn.Module,
    x0: torch.Tensor,
    steps: int = 64,
    solver: Callable = euler_solve,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create new training pairs by transporting source samples with a model."""

    # Reflow uses the current learned flow map to define new target endpoints.
    # The returned tensors are detached because they become training data for a
    # later model, not part of the current model's gradient graph.
    # Reflowでは、現在の学習済みflow mapで新しいtarget端点を作る。
    # 返すtensorは次のモデル用の訓練データなので、現在の勾配グラフから切り離す。
    x1_hat = solver(lambda t, x: model(t, x), x0, steps=steps)
    return x0.detach(), x1_hat.detach()


def trajectory_path_length(trajectory: torch.Tensor) -> torch.Tensor:
    """Length of each sampled trajectory.

    The expected shape is (time, batch, dim), as returned by euler_solve when
    return_trajectory=True.
    """

    # Sum the length of every small segment along each sampled trajectory.
    # 各軌道を小さな線分に分け、それらの長さを合計する。
    increments = trajectory[1:] - trajectory[:-1]
    return torch.linalg.vector_norm(increments, dim=-1).sum(dim=0)


def straightness_ratio(trajectory: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Path length divided by endpoint distance.

    A perfectly straight trajectory has ratio close to 1. Larger values indicate
    that the path bends or takes a detour.
    """

    path_length = trajectory_path_length(trajectory)
    endpoint_distance = torch.linalg.vector_norm(trajectory[-1] - trajectory[0], dim=-1)
    return path_length / endpoint_distance.clamp_min(eps)
