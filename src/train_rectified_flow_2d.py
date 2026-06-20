import argparse
from pathlib import Path

import torch

from fm_minimal import (
    RectifiedFlowMLP,
    euler_solve,
    get_time_sampler,
    rectified_flow_loss,
    sample_eight_gaussians,
    sample_standard_normal,
    straightness_ratio,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--time-sampler", choices=("uniform", "center", "endpoint"), default="uniform")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--time-dim", type=int, default=32)
    parser.add_argument("--activation", choices=("silu", "relu", "tanh"), default="silu")
    parser.add_argument("--out", type=Path, default=Path("flow_matching/research_material/_outputs"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    model_config = {
        "data_dim": 2,
        "time_dim": args.time_dim,
        "hidden_dim": args.hidden_dim,
        "depth": args.depth,
        "activation": args.activation,
    }
    model = RectifiedFlowMLP(**model_config).to(args.device)
    time_sampler = get_time_sampler(args.time_sampler)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for step in range(1, args.steps + 1):
        # 1-Rectified Flow pairs source noise and target data, then learns
        # the straight velocity x1 - x0 along their interpolation.
        # 1回目のRectified Flowでは、source noiseとtarget dataをペアにし、
        # 直線補間上の速度 x1 - x0 を学習する。
        x0 = sample_standard_normal(args.batch, device=args.device)
        x1 = sample_eight_gaussians(args.batch, device=args.device)
        loss = rectified_flow_loss(model, x0, x1, time_sampler=time_sampler)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step == 1 or step % 200 == 0:
            print(f"step={step:05d} rectified_flow_loss={loss.item():.4f}")

    x0 = sample_standard_normal(2048, device=args.device)
    samples = euler_solve(model, x0, steps=64)

    trajectory_source = sample_standard_normal(128, device=args.device)
    trajectory = euler_solve(model, trajectory_source, steps=64, return_trajectory=True)
    straightness = straightness_ratio(trajectory).mean()

    torch.save(
        {
            "model": model.state_dict(),
            "samples": samples.cpu(),
            "mean_straightness_ratio": float(straightness.cpu()),
            "time_sampler": args.time_sampler,
            "seed": args.seed,
            "model_config": model_config,
        },
        args.out / "rectified_flow_2d.pt",
    )
    print(f"mean_straightness_ratio={straightness.item():.4f}")
    print(f"saved: {args.out / 'rectified_flow_2d.pt'}")


if __name__ == "__main__":
    main()
