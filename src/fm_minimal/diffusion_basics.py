"""Flow Matching と対比するための最小 diffusion / DDIM 実装。

ここでは t=0 を clean data、t=1 を noise とする diffusion 側の時刻規約を使う。
Flow Matching の source→target 規約とは向きが異なる点に注意する。
"""

import torch
from torch import nn


def trig_alpha_sigma(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Simple continuous-time noise schedule.

    Convention:
        t = 0: clean data
        t = 1: nearly pure noise

    This is the diffusion-facing direction. The Flow Matching text later
    reverses the viewpoint when it treats noise as the source.
    """

    alpha = torch.cos(0.5 * torch.pi * t)
    sigma = torch.sin(0.5 * torch.pi * t)
    return alpha, sigma


def q_sample(
    x_data: torch.Tensor,
    t: torch.Tensor,
    noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a noised intermediate state x_t.

    Mathematical form:
        x_t = alpha(t) x_data + sigma(t) epsilon

    Shapes:
        x_data: [batch, dim]
        t:      [batch, 1]
        noise:  [batch, dim]
    """

    if noise is None:
        noise = torch.randn_like(x_data)
    alpha, sigma = trig_alpha_sigma(t)
    # This is the DDPM-style noising equation. It is intentionally written
    # beside the FM path code so the two parameterizations can be compared.
    # DDPM風のノイズ付加式。FMのpath実装と横並びで比較できるようにしている。
    x_t = alpha * x_data + sigma * noise
    return x_t, noise


def epsilon_prediction_loss(
    model: nn.Module,
    x_data: torch.Tensor,
) -> torch.Tensor:
    """Minimal diffusion-style epsilon-prediction loss.

    The model receives (t, x_t) and predicts the noise epsilon.
    """

    batch = x_data.shape[0]
    t = torch.rand(batch, 1, device=x_data.device, dtype=x_data.dtype)
    x_t, noise = q_sample(x_data, t)
    # Diffusion epsilon prediction changes only the teacher target; the model
    # still receives the same kind of input pair (t, x_t).
    # epsilon predictionでは教師ターゲットだけが変わる。
    # モデル入力はFMと同じく(t, x_t)の形で読むことができる。
    pred_noise = model(t, x_t)
    return ((pred_noise - noise) ** 2).sum(dim=1).mean()


@torch.no_grad()
def ddim_sample(
    model: nn.Module,
    initial_noise: torch.Tensor,
    steps: int = 64,
    t_start: float = 0.999,
    return_trajectory: bool = False,
) -> torch.Tensor:
    """Generate samples with a deterministic DDIM-style update.

    The model predicts epsilon at the current noisy state. We first estimate
    clean data and then rebuild the state at the next, less noisy time:

        x_data_hat = (x_t - sigma_t * epsilon_hat) / alpha_t
        x_next = alpha_next * x_data_hat + sigma_next * epsilon_hat

    Time runs from nearly pure noise (t_start) down to clean data (t=0).
    One model call is used per step, so NFE equals ``steps``.
    """

    if steps < 1:
        raise ValueError("steps must be at least 1")
    if not 0.0 < t_start < 1.0:
        raise ValueError("t_start must be between 0 and 1")

    x = initial_noise
    batch = x.shape[0]
    times = torch.linspace(t_start, 0.0, steps + 1, device=x.device, dtype=x.dtype)
    trajectory = [x.clone()] if return_trajectory else None

    for index in range(steps):
        t = times[index].expand(batch, 1)
        next_t = times[index + 1].expand(batch, 1)

        alpha, sigma = trig_alpha_sigma(t)
        next_alpha, next_sigma = trig_alpha_sigma(next_t)
        pred_noise = model(t, x)

        # t_start is below 1, so alpha is nonzero. Clamp protects custom
        # schedules or low-precision runs from division by an extremely small
        # value.
        alpha_safe = alpha.clamp_min(torch.finfo(x.dtype).eps)
        pred_data = (x - sigma * pred_noise) / alpha_safe
        x = next_alpha * pred_data + next_sigma * pred_noise

        if trajectory is not None:
            trajectory.append(x.clone())

    if trajectory is not None:
        return torch.stack(trajectory, dim=0)
    return x
