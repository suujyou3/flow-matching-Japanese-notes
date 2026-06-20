"""学習した velocity field の ODE dx/dt=v_theta(t,x) を数値積分する。

Euler・Heun・RK4 と、それぞれの NFE（model evaluation 数）の計算を提供する。
solver の step 数だけでなく NFE を揃えると、推論計算量を公平に比較できる。
"""

from collections.abc import Callable

import torch


VelocityFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def nfe_per_step(solver_name: str) -> int:
    """Number of model evaluations used by one solver step."""

    counts = {
        "euler": 1,
        "heun": 2,
        "rk4": 4,
    }
    try:
        return counts[solver_name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown solver: {solver_name}") from exc


def solver_nfe(solver_name: str, steps: int) -> int:
    """Total number of function evaluations for a fixed-step solver."""

    return nfe_per_step(solver_name) * steps


def steps_from_nfe_budget(solver_name: str, nfe_budget: int) -> int:
    """Largest integer step count that stays within an NFE budget."""

    per_step = nfe_per_step(solver_name)
    steps = nfe_budget // per_step
    if steps < 1:
        raise ValueError("nfe_budget is too small for this solver.")
    return steps


@torch.no_grad()
def euler_solve(
    velocity: VelocityFn,
    x0: torch.Tensor,
    steps: int = 64,
    t0: float = 0.0,
    t1: float = 1.0,
    return_trajectory: bool = False,
) -> torch.Tensor:
    """Solve dx/dt = v(t, x) with explicit Euler steps."""

    x = x0.clone()
    dt = (t1 - t0) / steps
    trajectory = [x.clone()]
    for i in range(steps):
        t = torch.full((x.shape[0], 1), t0 + i * dt, device=x.device, dtype=x.dtype)
        # Explicit Euler: x_{t+dt} = x_t + dt * v_theta(t, x_t).
        # 明示Euler法。現在位置に現在の速度を足して次の位置へ進む。
        x = x + dt * velocity(t, x)
        if return_trajectory:
            trajectory.append(x.clone())
    if return_trajectory:
        return torch.stack(trajectory, dim=0)
    return x


@torch.no_grad()
def heun_solve(
    velocity: VelocityFn,
    x0: torch.Tensor,
    steps: int = 64,
    t0: float = 0.0,
    t1: float = 1.0,
) -> torch.Tensor:
    """Second-order predictor-corrector solver."""

    x = x0.clone()
    dt = (t1 - t0) / steps
    for i in range(steps):
        t = torch.full((x.shape[0], 1), t0 + i * dt, device=x.device, dtype=x.dtype)
        t_next = torch.full((x.shape[0], 1), t0 + (i + 1) * dt, device=x.device, dtype=x.dtype)
        # Predictor-corrector: take an Euler prediction, then average the
        # velocity at the current and predicted next point.
        # 予測子修正子法。Eulerで仮の次点を作り、現在点と仮次点の速度を平均する。
        k1 = velocity(t, x)
        x_pred = x + dt * k1
        k2 = velocity(t_next, x_pred)
        x = x + 0.5 * dt * (k1 + k2)
    return x


@torch.no_grad()
def rk4_solve(
    velocity: VelocityFn,
    x0: torch.Tensor,
    steps: int = 64,
    t0: float = 0.0,
    t1: float = 1.0,
) -> torch.Tensor:
    """Classical fourth-order Runge-Kutta solver."""

    x = x0.clone()
    dt = (t1 - t0) / steps
    for i in range(steps):
        t = torch.full((x.shape[0], 1), t0 + i * dt, device=x.device, dtype=x.dtype)
        t_half = t + 0.5 * dt
        t_next = t + dt
        # RK4 evaluates the velocity four times per step, so its NFE is 4*steps.
        # RK4は1 stepで速度場を4回評価するため、NFEは4*stepsになる。
        k1 = velocity(t, x)
        k2 = velocity(t_half, x + 0.5 * dt * k1)
        k3 = velocity(t_half, x + 0.5 * dt * k2)
        k4 = velocity(t_next, x + dt * k3)
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return x
