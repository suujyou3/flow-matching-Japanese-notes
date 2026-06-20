"""2次元 toy problem で Conditional Flow Matching の最小実験を学習する。

source noise x0 と target data x1 を結び、条件付き経路上の点 x_t において
model velocity v_theta(t, x_t) を teacher velocity u_t へ回帰する。
path・coupling・time sampler を切り替え、各設計選択の影響を観察できる。
"""

import argparse
from pathlib import Path

import torch

from fm_minimal import (
    LinearPath,
    MLPVelocity,
    TrigGaussianPath,
    conditional_flow_matching_loss,
    euler_solve,
    get_time_sampler,
    greedy_minibatch_coupling,
    random_coupling,
    sample_eight_gaussians,
    sample_standard_normal,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--time-sampler", choices=("uniform", "center", "endpoint"), default="uniform")
    parser.add_argument("--path", choices=("linear", "trig"), default="linear")
    parser.add_argument("--coupling", choices=("independent", "greedy"), default="independent")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--time-dim", type=int, default=32)
    parser.add_argument("--activation", choices=("silu", "relu", "tanh"), default="silu")
    parser.add_argument("--out", type=Path, default=Path("flow_matching/research_material/_outputs"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    path = LinearPath() if args.path == "linear" else TrigGaussianPath()
    time_sampler = get_time_sampler(args.time_sampler)
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
        # Source is easy noise p_0; target is the toy data distribution p_1.
        # sourceは簡単にサンプルできるnoise分布p_0、targetはtoyデータ分布p_1。
        x0 = sample_standard_normal(args.batch, device=args.device)
        x1 = sample_eight_gaussians(args.batch, device=args.device)
        if args.coupling == "independent":
            x0, x1, _ = random_coupling(x0, x1)
        else:
            x0, x1, _ = greedy_minibatch_coupling(x0, x1)

        # The selected time sampler decides which part of the path is trained
        # more often. Uniform sampling remains the default baseline.
        # time samplerは、pathのどの区間を多く学ぶかを決める。
        # 既定値uniformなら、従来どおり全区間を一様に標本化する。
        loss = conditional_flow_matching_loss(model, path, x0, x1, time_sampler=time_sampler)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step == 1 or step % 200 == 0:
            print(f"step={step:05d} loss={loss.item():.4f}")

    # Save both the model weights and a quick generated sample cloud. The plot
    # script recomputes samples from the model, but this sample is useful for
    # quick checkpoint inspection.
    # モデル重みと簡易生成サンプルを保存する。plotスクリプトは再生成するが、
    # checkpointをすぐ確認するには保存済みsamplesが便利。
    x0 = sample_standard_normal(2048, device=args.device)
    samples = euler_solve(model, x0, steps=64)
    torch.save(
        {
            "model": model.state_dict(),
            "samples": samples.cpu(),
            "time_sampler": args.time_sampler,
            "path": args.path,
            "coupling": args.coupling,
            "seed": args.seed,
            "model_config": model_config,
        },
        args.out / "minimal_2d.pt",
    )
    print(f"saved: {args.out / 'minimal_2d.pt'}")


if __name__ == "__main__":
    main()
