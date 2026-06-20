import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from fm_minimal import (
    MLPVelocity,
    ddim_sample,
    eight_gaussian_centers,
    endpoint_error,
    gaussian_kernel_mmd,
    nearest_mode_coverage,
    sample_eight_gaussians,
    sample_standard_normal,
)


FIELDNAMES = [
    "sampler",
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("flow_matching/research_material/_outputs/diffusion_toy.pt"),
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--target-samples", type=int, default=2048)
    parser.add_argument("--steps", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--bandwidth", type=float, default=0.5)
    parser.add_argument("--coverage-radius", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=Path("flow_matching/research_material/_outputs/diffusion_toy_evaluation.csv"),
    )
    parser.add_argument(
        "--plot-out",
        type=Path,
        default=Path("flow_matching/research_material/figures/diffusion_toy_evaluation.png"),
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model = MLPVelocity(**checkpoint.get("model_config", {})).to(args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    source = sample_standard_normal(args.samples, device=args.device)
    target = sample_eight_gaussians(args.target_samples, device=args.device)
    centers = eight_gaussian_centers(device=args.device)

    rows: list[dict[str, object]] = []
    for steps in args.steps:
        generated = ddim_sample(model, source.clone(), steps=steps)
        mmd = gaussian_kernel_mmd(generated, target, bandwidth=args.bandwidth)
        coverage, counts = nearest_mode_coverage(generated, centers, radius=args.coverage_radius)
        err = endpoint_error(generated, target)
        rows.append(
            {
                "sampler": "ddim",
                "steps": steps,
                "nfe": steps,
                "mmd": float(mmd.item()),
                "coverage": float(coverage.item()),
                "endpoint_error": float(err.item()),
                "mode_counts": json.dumps(
                    [int(value) for value in counts.detach().cpu()],
                    separators=(",", ":"),
                ),
                "samples": args.samples,
                "target_samples": args.target_samples,
                "seed": args.seed,
                "bandwidth": args.bandwidth,
                "coverage_radius": args.coverage_radius,
            }
        )

    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    metrics = [
        ("mmd", "Biased MMD^2 (lower is better)"),
        ("coverage", "Mode coverage (higher is better)"),
        ("endpoint_error", "Endpoint error (lower is better)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    nfe = [int(row["nfe"]) for row in rows]
    for ax, (key, title) in zip(axes, metrics):
        ax.plot(nfe, [float(row[key]) for row in rows], marker="o", linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("NFE")
        ax.set_xscale("log", base=2)
        ax.set_xticks(nfe)
        ax.set_xticklabels([str(value) for value in nfe])
        ax.grid(alpha=0.25)
    axes[1].set_ylim(-0.02, 1.02)
    axes[0].set_ylabel("Metric value")
    fig.suptitle("2D Diffusion: DDIM-style sampling under NFE budget", fontsize=14)
    args.plot_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.plot_out, dpi=180)
    plt.close(fig)

    print(f"saved: {args.csv_out}")
    print(f"saved: {args.plot_out}")


if __name__ == "__main__":
    main()
