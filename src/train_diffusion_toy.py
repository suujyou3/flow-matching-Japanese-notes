"""Flow Matching と比較するための連続時刻 diffusion baseline を学習する。

clean data を時刻 t に応じて加ノイズし、model に混入した noise epsilon を
予測させる。データ分布と MLP 規模は Flow Matching 実験に揃えている。
"""

import argparse
from pathlib import Path

import torch

from fm_minimal import MLPVelocity, epsilon_prediction_loss, sample_eight_gaussians


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
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
    model = MLPVelocity(**model_config).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for step in range(1, args.steps + 1):
        # Diffusion training starts from clean data x_data and adds noise.
        # Diffusion学習ではclean data x_dataにノイズを足してx_tを作る。
        x_data = sample_eight_gaussians(args.batch, device=args.device)
        # epsilon_prediction_loss samples t, creates x_t, and uses epsilon as teacher.
        # epsilon_prediction_loss内部でtとx_tを作り、epsilonを教師信号にする。
        loss = epsilon_prediction_loss(model, x_data)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step == 1 or step % 200 == 0:
            print(f"step={step:05d} diffusion_epsilon_loss={loss.item():.4f}")

    torch.save(
        {
            "model": model.state_dict(),
            "seed": args.seed,
            "model_config": model_config,
        },
        args.out / "diffusion_toy.pt",
    )
    print(f"saved: {args.out / 'diffusion_toy.pt'}")


if __name__ == "__main__":
    main()
