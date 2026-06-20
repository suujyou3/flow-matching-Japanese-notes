import torch
from torch import nn

from .paths import ConditionalPath, LinearPath
from .time_samplers import TimeSampler, sample_uniform_time


def conditional_flow_matching_loss(
    model: nn.Module,
    path: ConditionalPath,
    x0: torch.Tensor,
    x1: torch.Tensor,
    time_sampler: TimeSampler = sample_uniform_time,
) -> torch.Tensor:
    """Conditional Flow Matching loss for paired source/target samples.

    Mathematical form:
        E_{t, x0, x1} || v_theta(t, x_t) - u_t ||^2

    This is intentionally small and explicit so the code mirrors the formula.
    """

    batch = x0.shape[0]
    # One random time per sample gives an unbiased Monte Carlo estimate of the
    # expectation over t in the CFM objective.
    # 各サンプルに1つずつランダム時刻を割り当て、CFM目的関数のtに関する期待値を
    # ミニバッチで近似する。
    t = time_sampler(batch, x0.device, x0.dtype)
    if t.shape != (batch, 1):
        raise ValueError(f"time_sampler must return shape {(batch, 1)}, got {tuple(t.shape)}")

    # The path object owns the mathematical choice of x_t and teacher velocity.
    # Swapping LinearPath for TrigGaussianPath changes these two tensors only.
    # pathオブジェクトが、x_tと教師速度u_tの数式上の選び方を担当する。
    # LinearPathをTrigGaussianPathに替えると、主にこの2つのtensorだけが変わる。
    x_t = path.sample(t, x0, x1)
    u_t = path.velocity(t, x0, x1)

    # The model sees only (t, x_t), not the endpoints. This is what makes it
    # learn a location/time velocity field usable at generation time.
    # モデルには端点x0,x1を渡さず、(t, x_t)だけを渡す。
    # これにより生成時にも使える「時刻と場所の速度場」を学習する。
    pred = model(t, x_t)
    return ((pred - u_t) ** 2).flatten(1).sum(dim=1).mean()


def rectified_flow_loss(
    model: nn.Module,
    x0: torch.Tensor,
    x1: torch.Tensor,
    time_sampler: TimeSampler = sample_uniform_time,
) -> torch.Tensor:
    """1-Rectified Flow loss with a straight interpolation path.

    Mathematical form:
        x_t = (1 - t) x_0 + t x_1
        u_t = x_1 - x_0
        loss = ||v_theta(t, x_t) - u_t||^2

    日本語:
        1回目のRectified Flowで使う直線補間loss。
        最小CFMと同じ形だが、Rectified Flow章ではこの名前で読む。
    """

    return conditional_flow_matching_loss(model, LinearPath(), x0, x1, time_sampler=time_sampler)
