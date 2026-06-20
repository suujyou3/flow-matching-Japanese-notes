"""学習済み2次元 Flow Matching model を複数の ODE solver で評価する。

生成品質だけでなく NFE（速度場の評価回数）も記録し、Euler・Heun・RK4 を
同一計算予算で比較できるようにする。結果は後段の表や図で使える形式に保存する。
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

from fm_minimal import (
    MLPVelocity,
    eight_gaussian_centers,
    endpoint_error,
    euler_solve,
    gaussian_kernel_mmd,
    heun_solve,
    nearest_mode_coverage,
    rk4_solve,
    sample_eight_gaussians,
    sample_standard_normal,
    solver_nfe,
    steps_from_nfe_budget,
)


SOLVERS = {
    "euler": euler_solve,
    "heun": heun_solve,
    "rk4": rk4_solve,
}

FIELDNAMES = [
    "solver",
    "steps",
    "nfe",
    "mmd",
    "coverage",
    "endpoint_error",
    "mode_counts",
    "samples",
    "target_samples",
    "seed",
    "bandwidth",
    "coverage_radius",
]


def set_seed(seed: int) -> None:
    """Fix CPU/CUDA seeds so evaluation conditions are reproducible.

    CPUとCUDAの乱数seedを固定し、評価条件を再現可能にする。
    """

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(
    model: torch.nn.Module,
    source: torch.Tensor,
    target: torch.Tensor,
    centers: torch.Tensor,
    solver_names: list[str],
    step_counts: list[int],
    nfe_budgets: list[int] | None,
    bandwidth: float,
    coverage_radius: float,
    seed: int,
) -> list[dict[str, object]]:
    """Evaluate every condition on the same source and target samples.

    すべての条件を、同じsource noiseとtarget標本で評価する。
    """

    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for solver_name in solver_names:
            if solver_name not in SOLVERS:
                raise ValueError(f"unknown solver: {solver_name}")

            solve = SOLVERS[solver_name]
            if nfe_budgets is None:
                solver_steps = step_counts
            else:
                solver_steps = [steps_from_nfe_budget(solver_name, budget) for budget in nfe_budgets]

            for steps in solver_steps:
                generated = solve(model, source.clone(), steps=steps)

                # These metrics detect distribution mismatch, missing modes,
                # and distance from target samples, respectively.
                # 各指標は、分布のずれ、mode落ち、targetからの距離を検出する。
                mmd = gaussian_kernel_mmd(generated, target, bandwidth=bandwidth)
                coverage, counts = nearest_mode_coverage(
                    generated,
                    centers,
                    radius=coverage_radius,
                )
                err = endpoint_error(generated, target)

                rows.append(
                    {
                        "solver": solver_name,
                        "steps": steps,
                        "nfe": solver_nfe(solver_name, steps),
                        "mmd": float(mmd.item()),
                        "coverage": float(coverage.item()),
                        "endpoint_error": float(err.item()),
                        "mode_counts": json.dumps(
                            [int(v) for v in counts.detach().cpu()],
                            separators=(",", ":"),
                        ),
                        "samples": source.shape[0],
                        "target_samples": target.shape[0],
                        "seed": seed,
                        "bandwidth": bandwidth,
                        "coverage_radius": coverage_radius,
                    }
                )
    return rows


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    """Save results as a standards-compliant CSV file.

    mode_counts内のcommaも正しくquoteしたCSVとして評価結果を保存する。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def print_csv(rows: list[dict[str, object]]) -> None:
    """Keep terminal-friendly CSV output in addition to file output.

    ファイル保存に加え、ターミナルにもCSV形式で結果を表示する。
    """

    writer = csv.DictWriter(sys.stdout, fieldnames=FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)


def plot_rows(rows: list[dict[str, object]], path: Path) -> None:
    """Plot biased MMD^2, mode coverage, and endpoint error against NFE.

    biased MMD^2、mode coverage、endpoint errorをNFEに対して描画する。
    """

    import matplotlib.pyplot as plt

    metrics = [
        ("mmd", "Biased MMD^2 (lower is better)"),
        ("coverage", "Mode coverage (higher is better)"),
        ("endpoint_error", "Endpoint error (lower is better)"),
    ]
    colors = {"euler": "#2F6FA3", "heun": "#C47A2C", "rk4": "#4F8F57"}
    markers = {"euler": "o", "heun": "s", "rk4": "^"}
    solver_names = list(dict.fromkeys(str(row["solver"]) for row in rows))
    nfe_ticks = sorted({int(row["nfe"]) for row in rows})

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    for ax, (key, title) in zip(axes, metrics):
        for solver_name in solver_names:
            solver_rows = sorted(
                (row for row in rows if row["solver"] == solver_name),
                key=lambda row: int(row["nfe"]),
            )
            ax.plot(
                [int(row["nfe"]) for row in solver_rows],
                [float(row[key]) for row in solver_rows],
                label=solver_name.capitalize(),
                color=colors.get(solver_name, "#59636F"),
                marker=markers.get(solver_name, "o"),
                linewidth=2,
                markersize=6,
            )
        ax.set_title(title)
        ax.set_xlabel("NFE")
        ax.set_xscale("log", base=2)
        ax.set_xticks(nfe_ticks)
        ax.set_xticklabels([str(value) for value in nfe_ticks])
        ax.grid(alpha=0.25)

    axes[1].set_ylim(-0.02, 1.02)
    axes[0].set_ylabel("Metric value")
    axes[-1].legend(frameon=False, loc="best")
    fig.suptitle("2D Flow Matching: solver comparison under NFE budget", fontsize=14)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("flow_matching/research_material/_outputs/minimal_2d.pt"),
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--target-samples", type=int, default=2048)
    parser.add_argument("--steps", type=int, nargs="+", default=None)
    parser.add_argument("--nfe-budgets", type=int, nargs="+", default=None)
    parser.add_argument("--solvers", type=str, nargs="+", default=["euler", "heun"])
    parser.add_argument("--bandwidth", type=float, default=0.5)
    parser.add_argument("--coverage-radius", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=Path("flow_matching/research_material/_outputs/minimal_2d_evaluation.csv"),
    )
    parser.add_argument(
        "--plot-out",
        type=Path,
        default=Path("flow_matching/research_material/figures/minimal_2d_solver_comparison.png"),
    )
    args = parser.parse_args()

    if args.steps is not None and args.nfe_budgets is not None:
        parser.error("use either --steps or --nfe-budgets, not both")
    step_counts = args.steps if args.steps is not None else [8, 16, 32, 64]

    set_seed(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model = MLPVelocity(**checkpoint.get("model_config", {})).to(args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # Reuse the same random samples for every condition so the comparison is
    # focused on solver and NFE differences.
    # 全条件で同じ乱数を再利用し、solverとNFE以外の差を抑える。
    source = sample_standard_normal(args.samples, device=args.device)
    target = sample_eight_gaussians(args.target_samples, device=args.device)
    centers = eight_gaussian_centers(device=args.device)

    rows = evaluate(
        model=model,
        source=source,
        target=target,
        centers=centers,
        solver_names=args.solvers,
        step_counts=step_counts,
        nfe_budgets=args.nfe_budgets,
        bandwidth=args.bandwidth,
        coverage_radius=args.coverage_radius,
        seed=args.seed,
    )
    print_csv(rows)
    write_csv(rows, args.csv_out)
    plot_rows(rows, args.plot_out)
    print(f"saved CSV: {args.csv_out}")
    print(f"saved plot: {args.plot_out}")


if __name__ == "__main__":
    main()
