import argparse
from pathlib import Path

import torch

from fm_minimal import (
    LinearPath,
    MLPVelocity,
    conditional_flow_matching_loss,
    euler_solve,
    get_time_sampler,
    make_reflow_pairs,
    sample_standard_normal,
    straightness_ratio,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=Path("flow_matching/research_material/_outputs/rectified_flow_2d.pt"),
    )
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pair-steps", type=int, default=64)
    parser.add_argument("--time-sampler", choices=("uniform", "center", "endpoint"), default="uniform")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("flow_matching/research_material/_outputs/reflow"),
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    base_checkpoint = torch.load(args.base_checkpoint, map_location=args.device)
    model_config = base_checkpoint.get("model_config", {})
    base_model = MLPVelocity(**model_config).to(args.device)
    base_model.load_state_dict(base_checkpoint["model"])
    base_model.eval()
    for parameter in base_model.parameters():
        parameter.requires_grad_(False)

    model = MLPVelocity(**model_config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    path = LinearPath()
    time_sampler = get_time_sampler(args.time_sampler)

    for step in range(1, args.steps + 1):
        source = sample_standard_normal(args.batch, device=args.device)
        reflow_x0, reflow_x1 = make_reflow_pairs(
            base_model,
            source,
            steps=args.pair_steps,
            solver=euler_solve,
        )
        loss = conditional_flow_matching_loss(
            model,
            path,
            reflow_x0,
            reflow_x1,
            time_sampler=time_sampler,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step % 200 == 0:
            print(f"step={step:05d} reflow_loss={loss.item():.4f}")

    source = sample_standard_normal(2048, device=args.device)
    samples = euler_solve(model, source, steps=64)
    trajectory_source = sample_standard_normal(128, device=args.device)
    trajectory = euler_solve(model, trajectory_source, steps=64, return_trajectory=True)
    mean_straightness = straightness_ratio(trajectory).mean()

    output = args.out / "reflow_2d.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "model_config": model_config,
            "samples": samples.cpu(),
            "mean_straightness_ratio": float(mean_straightness.cpu()),
            "base_checkpoint": str(args.base_checkpoint),
            "pair_steps": args.pair_steps,
            "time_sampler": args.time_sampler,
            "seed": args.seed,
        },
        output,
    )
    print(f"mean_straightness_ratio={mean_straightness.item():.4f}")
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
