"""Flow Matching ODE が定める2次元密度とNLLを評価する最小実装。

状態 x と対数密度の変化を同時に積分し、target data を t=1 から
既知のsource分布 t=0へ戻して log p_1(x) を求める。
"""

import math

import torch

from .solvers import nfe_per_step


def standard_normal_log_prob(x: torch.Tensor) -> torch.Tensor:
    """標準正規分布のサンプルごとの対数密度を返す。"""

    log_two_pi = math.log(2.0 * math.pi)
    return -0.5 * (x.square() + log_two_pi).flatten(1).sum(dim=1)


def velocity_and_exact_divergence(
    model: torch.nn.Module,
    t: torch.Tensor,
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """速度と発散 div_x v_theta(t, x) を2次元で厳密計算する。

    出力の各座標を対応する入力座標で微分し、Jacobianの対角和を取る。
    画像では高コストになるためHutchinson推定を使うことが多いが、
    この教材の2D toyでは厳密値を使ってsolver誤差だけを観察する。
    """

    with torch.enable_grad():
        x_work = x.detach().requires_grad_(True)
        velocity = model(t, x_work)
        divergence = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

        for axis in range(x.shape[1]):
            # batch内の出力を足してから微分しても、モデルが各sampleを独立に
            # 処理するため、各sampleの対応する偏微分を取り出せる。
            gradient = torch.autograd.grad(
                velocity[:, axis].sum(),
                x_work,
                retain_graph=axis + 1 < x.shape[1],
                create_graph=False,
            )[0]
            divergence = divergence + gradient[:, axis]

    return velocity.detach(), divergence.detach()


def solve_augmented_density_ode(
    model: torch.nn.Module,
    x_start: torch.Tensor,
    solver_name: str,
    steps: int,
    t0: float = 1.0,
    t1: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """状態と対数密度変化をEuler・Heun・RK4で同時に積分する。

    戻り値の第2要素は log p(t1) - log p(t0) である。NLL評価では
    target t=1からsource t=0へ逆向きに解くので、これを使って
    log p_1 = log p_0 - (log p_0 - log p_1) を復元する。
    """

    if solver_name not in {"euler", "heun", "rk4"}:
        raise ValueError(f"unknown solver: {solver_name}")
    if steps < 1:
        raise ValueError("steps must be at least 1")

    x = x_start.detach().clone()
    logp_change = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
    dt = (t1 - t0) / steps

    def dynamics(time_value: float, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.full(
            (state.shape[0], 1),
            time_value,
            device=state.device,
            dtype=state.dtype,
        )
        velocity, divergence = velocity_and_exact_divergence(model, t, state)
        # instantaneous change-of-variables: d log p_t(x_t) / dt = -div v_t
        # 点群が広がると密度が下がり、縮むと密度が上がる変化を追跡する。
        return velocity, -divergence

    for index in range(steps):
        time_value = t0 + index * dt
        if solver_name == "euler":
            k1_x, k1_logp = dynamics(time_value, x)
            x = x + dt * k1_x
            logp_change = logp_change + dt * k1_logp
        elif solver_name == "heun":
            k1_x, k1_logp = dynamics(time_value, x)
            x_predict = x + dt * k1_x
            k2_x, k2_logp = dynamics(time_value + dt, x_predict)
            x = x + 0.5 * dt * (k1_x + k2_x)
            logp_change = logp_change + 0.5 * dt * (k1_logp + k2_logp)
        else:
            k1_x, k1_logp = dynamics(time_value, x)
            k2_x, k2_logp = dynamics(time_value + 0.5 * dt, x + 0.5 * dt * k1_x)
            k3_x, k3_logp = dynamics(time_value + 0.5 * dt, x + 0.5 * dt * k2_x)
            k4_x, k4_logp = dynamics(time_value + dt, x + dt * k3_x)
            x = x + (dt / 6.0) * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x)
            logp_change = logp_change + (dt / 6.0) * (
                k1_logp + 2.0 * k2_logp + 2.0 * k3_logp + k4_logp
            )

    return x, logp_change


def estimate_nll(
    model: torch.nn.Module,
    target: torch.Tensor,
    solver_name: str,
    steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """target batchの平均NLLとサンプルごとのlog p_1を返す。"""

    source, logp0_minus_logp1 = solve_augmented_density_ode(
        model,
        target,
        solver_name=solver_name,
        steps=steps,
        t0=1.0,
        t1=0.0,
    )
    logp0 = standard_normal_log_prob(source)
    logp1 = logp0 - logp0_minus_logp1
    return -logp1.mean(), logp1


def nll_nfe(solver_name: str, steps: int) -> int:
    """NLL用augmented ODEが行う速度・発散評価の回数を返す。"""

    return nfe_per_step(solver_name) * steps
