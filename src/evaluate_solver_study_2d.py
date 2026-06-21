"""第9部のsolver・NFE比較実験をまとめて実行する。

次の4種類を同じcheckpointと乱数条件で評価する。
1. NFE sweep
2. 同一step数と同一NFEの比較
3. 品質–計算量曲線
4. NLLのsolver精度依存性
"""

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import torch

from fm_minimal import (
    MLPVelocity,
    eight_gaussian_centers,
    endpoint_error,
    estimate_nll,
    euler_solve,
    gaussian_kernel_mmd,
    heun_solve,
    nearest_mode_coverage,
    nfe_per_step,
    nll_nfe,
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

# 実行時のcurrent directoryに左右されず、教材rootを基準に入出力する。
MATERIAL_ROOT = Path(__file__).resolve().parents[1]

QUALITY_FIELDS = [
    "comparison",
    "solver",
    "requested_nfe_budget",
    "steps",
    "nfe",
    "mmd",
    "coverage",
    "endpoint_error",
    "wall_time_ms",
    "throughput_samples_per_sec",
    "peak_extra_memory_mb",
    "mode_counts",
    "samples",
    "target_samples",
    "seed",
]

NLL_FIELDS = [
    "solver",
    "requested_nfe_budget",
    "steps",
    "nfe",
    "nll_nats",
    "reference_nll_nats",
    "absolute_error_from_reference",
    "wall_time_ms",
    "samples",
    "seed",
    "reference_solver",
    "reference_steps",
    "reference_nfe",
]


def set_seed(seed: int) -> None:
    """CPUとCUDAの乱数seedを固定する。"""

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synchronize(device: torch.device) -> None:
    """CUDAの非同期実行を待ち、実時間を正しく測れるようにする。"""

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_generate(
    model: torch.nn.Module,
    source: torch.Tensor,
    solver_name: str,
    steps: int,
    warmup: int,
    repeats: int,
) -> tuple[torch.Tensor, float, float, float]:
    """生成結果、平均実時間、throughput、追加peak memoryを返す。"""

    solve = SOLVERS[solver_name]
    device = source.device

    # 初回だけ発生するkernel初期化などを本計測へ混ぜない。
    for _ in range(warmup):
        solve(model, source.clone(), steps=steps)
    synchronize(device)

    baseline_memory = 0
    if device.type == "cuda":
        baseline_memory = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)

    generated = source
    elapsed_values: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        generated = solve(model, source.clone(), steps=steps)
        synchronize(device)
        elapsed_values.append(time.perf_counter() - start)

    # OS schedulingやbackground処理による外れ値を抑えるため中央値を使う。
    elapsed_per_repeat = statistics.median(elapsed_values)

    peak_extra_memory_mb = 0.0
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(device)
        peak_extra_memory_mb = max(0, peak - baseline_memory) / (1024.0**2)

    throughput = source.shape[0] / elapsed_per_repeat
    return generated, elapsed_per_repeat * 1000.0, throughput, peak_extra_memory_mb


def quality_row(
    model: torch.nn.Module,
    source: torch.Tensor,
    target: torch.Tensor,
    centers: torch.Tensor,
    comparison: str,
    solver_name: str,
    steps: int,
    requested_budget: int | None,
    bandwidth: float,
    coverage_radius: float,
    warmup: int,
    repeats: int,
    seed: int,
) -> dict[str, object]:
    """一つのsolver条件について品質と計算量を測る。"""

    generated, wall_time_ms, throughput, peak_memory = timed_generate(
        model,
        source,
        solver_name,
        steps,
        warmup,
        repeats,
    )
    with torch.no_grad():
        # 同じ生成標本から複数指標を計算し、指標ごとの乱数差を避ける。
        mmd = gaussian_kernel_mmd(generated, target, bandwidth=bandwidth)
        coverage, counts = nearest_mode_coverage(generated, centers, radius=coverage_radius)
        error = endpoint_error(generated, target)

    return {
        "comparison": comparison,
        "solver": solver_name,
        "requested_nfe_budget": "" if requested_budget is None else requested_budget,
        "steps": steps,
        "nfe": solver_nfe(solver_name, steps),
        "mmd": float(mmd.item()),
        "coverage": float(coverage.item()),
        "endpoint_error": float(error.item()),
        "wall_time_ms": wall_time_ms,
        "throughput_samples_per_sec": throughput,
        "peak_extra_memory_mb": peak_memory,
        "mode_counts": json.dumps([int(value) for value in counts.cpu()], separators=(",", ":")),
        "samples": source.shape[0],
        "target_samples": target.shape[0],
        "seed": seed,
    }


def evaluate_quality_and_compute(
    model: torch.nn.Module,
    source: torch.Tensor,
    target: torch.Tensor,
    centers: torch.Tensor,
    solver_names: list[str],
    nfe_budgets: list[int],
    same_steps: list[int],
    bandwidth: float,
    coverage_radius: float,
    warmup: int,
    repeats: int,
    seed: int,
) -> list[dict[str, object]]:
    """NFE sweepと同一step数比較を同じ条件で実行する。"""

    rows: list[dict[str, object]] = []
    condition_cache: dict[tuple[str, int], dict[str, object]] = {}

    def cached_quality_row(
        solver_name: str,
        steps: int,
        comparison: str,
        requested_budget: int | None,
    ) -> dict[str, object]:
        """同じsolver・steps条件は一度だけ計測し、protocol間で再利用する。"""

        key = (solver_name, steps)
        if key not in condition_cache:
            condition_cache[key] = quality_row(
                model,
                source,
                target,
                centers,
                comparison=comparison,
                solver_name=solver_name,
                steps=steps,
                requested_budget=requested_budget,
                bandwidth=bandwidth,
                coverage_radius=coverage_radius,
                warmup=warmup,
                repeats=repeats,
                seed=seed,
            )
        row = dict(condition_cache[key])
        row["comparison"] = comparison
        row["requested_nfe_budget"] = "" if requested_budget is None else requested_budget
        return row

    for solver_name in solver_names:
        if solver_name not in SOLVERS:
            raise ValueError(f"unknown solver: {solver_name}")

        # NFEが1 step分に満たないsolverは、その予算点を欠測として扱う。
        for budget in nfe_budgets:
            if budget < nfe_per_step(solver_name):
                continue
            steps = steps_from_nfe_budget(solver_name, budget)
            rows.append(cached_quality_row(solver_name, steps, "same_nfe", budget))

        for steps in same_steps:
            rows.append(cached_quality_row(solver_name, steps, "same_steps", None))
    return rows


def timed_nll(
    model: torch.nn.Module,
    target: torch.Tensor,
    solver_name: str,
    steps: int,
) -> tuple[float, float]:
    """平均NLLとNLL計算の実時間を返す。"""

    synchronize(target.device)
    start = time.perf_counter()
    nll, _ = estimate_nll(model, target, solver_name=solver_name, steps=steps)
    synchronize(target.device)
    return float(nll.item()), (time.perf_counter() - start) * 1000.0


def evaluate_nll_sensitivity(
    model: torch.nn.Module,
    target: torch.Tensor,
    solver_names: list[str],
    nfe_budgets: list[int],
    reference_solver: str,
    reference_steps: int,
    seed: int,
) -> list[dict[str, object]]:
    """高精度基準からのNLL差をsolver・NFEごとに測る。"""

    reference_nll, _ = timed_nll(model, target, reference_solver, reference_steps)
    rows: list[dict[str, object]] = []

    for solver_name in solver_names:
        for budget in nfe_budgets:
            if budget < nfe_per_step(solver_name):
                continue
            steps = steps_from_nfe_budget(solver_name, budget)
            nll, wall_time_ms = timed_nll(model, target, solver_name, steps)
            rows.append(
                {
                    "solver": solver_name,
                    "requested_nfe_budget": budget,
                    "steps": steps,
                    "nfe": nll_nfe(solver_name, steps),
                    "nll_nats": nll,
                    "reference_nll_nats": reference_nll,
                    "absolute_error_from_reference": abs(nll - reference_nll),
                    "wall_time_ms": wall_time_ms,
                    "samples": target.shape[0],
                    "seed": seed,
                    "reference_solver": reference_solver,
                    "reference_steps": reference_steps,
                    "reference_nfe": nll_nfe(reference_solver, reference_steps),
                }
            )
    return rows


def write_csv(rows: list[dict[str, object]], fieldnames: list[str], path: Path) -> None:
    """評価結果をUTF-8 CSVとして保存する。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_csv(rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    """保存内容と同じ列をターミナルにも表示する。"""

    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)


def plot_quality_compute(rows: list[dict[str, object]], path: Path) -> None:
    """同一NFE条件の品質–計算量曲線を保存する。"""

    import matplotlib.pyplot as plt

    sweep_rows = [row for row in rows if row["comparison"] == "same_nfe"]
    metrics = [
        ("mmd", "Biased MMD^2 (lower is better)"),
        ("coverage", "Mode coverage (higher is better)"),
        ("endpoint_error", "Endpoint error (lower is better)"),
        ("wall_time_ms", "Wall time ms (lower is better)"),
    ]
    colors = {"euler": "#2F6FA3", "heun": "#C47A2C", "rk4": "#4F8F57"}
    markers = {"euler": "o", "heun": "s", "rk4": "^"}
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), constrained_layout=True)

    for axis, (metric, title) in zip(axes.flat, metrics):
        for solver_name in dict.fromkeys(str(row["solver"]) for row in sweep_rows):
            solver_rows = sorted(
                (row for row in sweep_rows if row["solver"] == solver_name),
                key=lambda row: int(row["nfe"]),
            )
            axis.plot(
                [int(row["nfe"]) for row in solver_rows],
                [float(row[metric]) for row in solver_rows],
                label=solver_name.capitalize(),
                color=colors.get(solver_name, "#59636F"),
                marker=markers.get(solver_name, "o"),
                linewidth=2,
            )
        axis.set_title(title)
        axis.set_xlabel("Actual NFE")
        axis.set_xscale("log", base=2)
        axis.grid(alpha=0.25)

    axes[0, 1].set_ylim(-0.02, 1.02)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("2D Flow Matching: quality-compute curves")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_nll(rows: list[dict[str, object]], path: Path) -> None:
    """NLL推定値と高精度基準からの誤差をNFEに対して描く。"""

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    for solver_name in dict.fromkeys(str(row["solver"]) for row in rows):
        solver_rows = sorted(
            (row for row in rows if row["solver"] == solver_name),
            key=lambda row: int(row["nfe"]),
        )
        nfe = [int(row["nfe"]) for row in solver_rows]
        axes[0].plot(nfe, [float(row["nll_nats"]) for row in solver_rows], marker="o", label=solver_name)
        axes[1].plot(
            nfe,
            # 基準値と一致した点でもlog軸を描けるよう、表示時だけ下限を置く。
            [max(float(row["absolute_error_from_reference"]), 1e-12) for row in solver_rows],
            marker="o",
            label=solver_name,
        )

    reference = float(rows[0]["reference_nll_nats"])
    axes[0].axhline(reference, color="#333333", linestyle="--", label="reference")
    axes[0].set_title("Estimated NLL (nats, lower is better)")
    axes[1].set_title("Absolute error from reference")
    for axis in axes:
        axis.set_xlabel("Actual NFE")
        axis.set_xscale("log", base=2)
        axis.grid(alpha=0.25)
    axes[1].set_yscale("log")
    axes[0].legend(frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=MATERIAL_ROOT / "_outputs" / "minimal_2d.pt",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--target-samples", type=int, default=2048)
    parser.add_argument("--nll-samples", type=int, default=64)
    parser.add_argument("--solvers", nargs="+", default=["euler", "heun", "rk4"])
    parser.add_argument("--nfe-budgets", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128])
    parser.add_argument("--same-steps", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--nll-nfe-budgets", type=int, nargs="+", default=[4, 8, 16, 32, 64])
    parser.add_argument("--nll-reference-solver", choices=tuple(SOLVERS), default="rk4")
    parser.add_argument("--nll-reference-steps", type=int, default=128)
    parser.add_argument("--skip-nll", action="store_true")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument("--bandwidth", type=float, default=0.5)
    parser.add_argument("--coverage-radius", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=MATERIAL_ROOT / "_outputs" / "solver_study",
    )
    args = parser.parse_args()

    if args.warmup < 0 or args.timing_repeats < 1:
        parser.error("warmup must be non-negative and timing-repeats must be positive")
    if any(value < 1 for value in args.nfe_budgets + args.same_steps + args.nll_nfe_budgets):
        parser.error("steps and NFE budgets must be positive")

    set_seed(args.seed)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = MLPVelocity(**checkpoint.get("model_config", {})).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # 全solver・全予算で同じ標本を再利用し、比較対象以外の乱数差を抑える。
    source = sample_standard_normal(args.samples, device=device)
    target = sample_eight_gaussians(args.target_samples, device=device)
    centers = eight_gaussian_centers(device=device)

    quality_rows = evaluate_quality_and_compute(
        model=model,
        source=source,
        target=target,
        centers=centers,
        solver_names=args.solvers,
        nfe_budgets=args.nfe_budgets,
        same_steps=args.same_steps,
        bandwidth=args.bandwidth,
        coverage_radius=args.coverage_radius,
        warmup=args.warmup,
        repeats=args.timing_repeats,
        seed=args.seed,
    )

    quality_csv = args.out_dir / "solver_quality_compute.csv"
    quality_plot = args.out_dir / "solver_quality_compute.png"
    write_csv(quality_rows, QUALITY_FIELDS, quality_csv)
    plot_quality_compute(quality_rows, quality_plot)
    print_csv(quality_rows, QUALITY_FIELDS)
    print(f"saved: {quality_csv}")
    print(f"saved: {quality_plot}")

    if not args.skip_nll:
        # NLL用targetは計算量を抑えるため別batchにし、全条件で固定する。
        nll_target = sample_eight_gaussians(args.nll_samples, device=device)
        nll_rows = evaluate_nll_sensitivity(
            model=model,
            target=nll_target,
            solver_names=args.solvers,
            nfe_budgets=args.nll_nfe_budgets,
            reference_solver=args.nll_reference_solver,
            reference_steps=args.nll_reference_steps,
            seed=args.seed,
        )
        nll_csv = args.out_dir / "solver_nll_sensitivity.csv"
        nll_plot = args.out_dir / "solver_nll_sensitivity.png"
        write_csv(nll_rows, NLL_FIELDS, nll_csv)
        plot_nll(nll_rows, nll_plot)
        print_csv(nll_rows, NLL_FIELDS)
        print(f"saved: {nll_csv}")
        print(f"saved: {nll_plot}")


if __name__ == "__main__":
    main()
