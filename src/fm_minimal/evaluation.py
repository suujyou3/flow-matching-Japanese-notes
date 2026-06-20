import torch


def pairwise_squared_distances(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Squared Euclidean distances between two point clouds."""

    return torch.cdist(x, y, p=2).pow(2)


def gaussian_kernel_mmd(
    x: torch.Tensor,
    y: torch.Tensor,
    bandwidth: float = 1.0,
) -> torch.Tensor:
    """Biased MMD^2 estimate with an RBF kernel for small toy experiments."""

    gamma = 1.0 / (2.0 * bandwidth * bandwidth)
    # Compare smoothed point clouds. Lower is better, but the value depends on
    # bandwidth, so always log the bandwidth with the metric.
    # ぼかした点群どうしを比較する。小さいほどよいがbandwidth依存なので、
    # metricと一緒にbandwidthも必ず記録する。
    k_xx = torch.exp(-gamma * pairwise_squared_distances(x, x)).mean()
    k_yy = torch.exp(-gamma * pairwise_squared_distances(y, y)).mean()
    k_xy = torch.exp(-gamma * pairwise_squared_distances(x, y)).mean()
    return k_xx + k_yy - 2.0 * k_xy


def nearest_mode_coverage(
    samples: torch.Tensor,
    centers: torch.Tensor,
    radius: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fraction of modes that receive at least one nearby generated sample."""

    distances = torch.cdist(samples, centers, p=2)
    nearest_distance, nearest_mode = distances.min(dim=1)
    # A mode counts as covered if at least one generated sample falls within
    # the chosen radius of that mode center.
    # mode中心から指定半径内に生成サンプルが1つでも入れば、そのmodeはcoveredと数える。
    inside = nearest_distance <= radius
    counts = torch.bincount(
        nearest_mode[inside],
        minlength=centers.shape[0],
    )
    coverage = (counts > 0).float().mean()
    return coverage, counts


def endpoint_error(generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean nearest-neighbor distance from generated samples to target samples."""

    # This measures closeness to the target cloud, but not whether all modes are
    # represented. Use it together with nearest_mode_coverage.
    # target点群への近さを見る指標。ただし全modeを覆っているかは分からないため、
    # nearest_mode_coverageと併用する。
    distances = torch.cdist(generated, target, p=2)
    return distances.min(dim=1).values.mean()
